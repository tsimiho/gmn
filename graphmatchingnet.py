import torch
import torch.nn as nn


def unsorted_segment_sum(data, segment_ids, num_segments):
    assert all(
        [i in data.shape for i in segment_ids.shape]
    ), "segment_ids.shape should be a prefix of data.shape"

    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], device=data.device)).long()
        segment_ids = segment_ids.repeat_interleave(s).view(
            segment_ids.shape[0], *data.shape[1:]
        )

    assert (
        data.shape == segment_ids.shape
    ), "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape, device=data.device).scatter_add(0, segment_ids, data)
    tensor = tensor.type(data.dtype)
    return tensor


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        node_hidden_sizes=None,
        edge_hidden_sizes=None,
        name="graph-encoder",
    ):
        super(GraphEncoder, self).__init__()
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes
        self._build_model()

    def _build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
        for i in range(1, len(self._node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(
                nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i])
            )
        self.MLP1 = nn.Sequential(*layer)

        if self._edge_hidden_sizes is not None:
            layer = []
            layer.append(nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._edge_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(
                    nn.Linear(
                        self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]
                    )
                )
            self.MLP2 = nn.Sequential(*layer)
        else:
            self.MLP2 = None

    def forward(self, node_features, edge_features=None):
        if self._node_hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(edge_features)

        return node_outputs, edge_outputs


def graph_prop_once(
    node_states,
    from_idx,
    to_idx,
    message_net,
    aggregation_module=None,
    edge_features=None,
):
    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]

    if edge_features is not None:
        edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)
    messages = message_net(edge_inputs)

    tensor = unsorted_segment_sum(messages, to_idx, node_states.shape[0])
    return tensor


class GraphPropLayer(nn.Module):
    def __init__(
        self,
        node_state_dim,
        edge_state_dim,
        edge_hidden_sizes,  # int
        node_hidden_sizes,  # int
        edge_net_init_scale=0.1,
        node_update_type="residual",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
        prop_type="embedding",
        name="graph-net",
    ):
        super(GraphPropLayer, self).__init__()

        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]

        # output size is node_state_dim
        self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type

        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different

        self._layer_norm = layer_norm
        self._prop_type = prop_type
        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()

    def build_model(self):
        layer = []
        layer.append(
            nn.Linear(
                self._node_state_dim * 2 + self._edge_state_dim,
                self._edge_hidden_sizes[0],
            )
        )
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(
                nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i])
            )
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(
                    nn.Linear(
                        self._node_state_dim * 2 + self._edge_state_dim,
                        self._edge_hidden_sizes[0],
                    )
                )
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append(
                        nn.Linear(
                            self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]
                        )
                    )
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net

        if self._node_update_type == "gru":
            if self._prop_type == "embedding":
                self.GRU = torch.nn.GRU(self._node_state_dim * 2, self._node_state_dim)
            elif self._prop_type == "matching":
                self.GRU = torch.nn.GRU(self._node_state_dim * 3, self._node_state_dim)
        else:
            layer = []
            if self._prop_type == "embedding":
                layer.append(
                    nn.Linear(self._node_state_dim * 3, self._node_hidden_sizes[0])
                )
            elif self._prop_type == "matching":
                layer.append(
                    nn.Linear(self._node_state_dim * 4, self._node_hidden_sizes[0])
                )
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(
                    nn.Linear(
                        self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]
                    )
                )
            self.MLP = nn.Sequential(*layer)

    def _compute_aggregated_messages(
        self, node_states, from_idx, to_idx, edge_features=None
    ):
        aggregated_messages = graph_prop_once(
            node_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            edge_features=edge_features,
        )

        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(
                node_states,
                to_idx,
                from_idx,
                self._reverse_message_net,
                aggregation_module=None,
                edge_features=edge_features,
            )

            aggregated_messages = aggregated_messages + reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages

    def _compute_node_update(self, node_states, node_state_inputs, node_features=None):
        if self._node_update_type in ("mlp", "residual"):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == "gru":
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(node_state_inputs)
            if self._layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self._node_update_type == "mlp":
                return mlp_output
            elif self._node_update_type == "residual":
                return node_states + mlp_output
            else:
                raise ValueError("Unknown node update type %s" % self._node_update_type)

    def forward(
        self, node_states, from_idx, to_idx, edge_features=None, node_features=None
    ):
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features
        )

        return self._compute_node_update(
            node_states, [aggregated_messages], node_features=node_features
        )


class GraphAggregator(nn.Module):
    def __init__(
        self,
        node_hidden_sizes,
        graph_transform_sizes=None,
        input_size=None,
        gated=True,
        aggregation_type="sum",
        name="graph-aggregator",
    ):
        super(GraphAggregator, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = input_size
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = self._node_hidden_sizes
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        layer.append(nn.Linear(self._input_size[0], node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if (
            self._graph_transform_sizes is not None
            and len(self._graph_transform_sizes) > 0
        ):
            layer = []
            layer.append(
                nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0])
            )
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.ReLU())
                layer.append(
                    nn.Linear(
                        self._graph_transform_sizes[i - 1],
                        self._graph_transform_sizes[i],
                    )
                )
            MLP2 = nn.Sequential(*layer)

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        node_states_g = self.MLP1(node_states)

        if self._gated:
            gates = torch.sigmoid(node_states_g[:, : self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim :] * gates

        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)

        if self._aggregation_type == "max":
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)

        if (
            self._graph_transform_sizes is not None
            and len(self._graph_transform_sizes) > 0
        ):
            graph_states = self.MLP2(graph_states)

        return graph_states


class GraphEmbeddingNet(nn.Module):
    def __init__(
        self,
        encoder,
        aggregator,
        node_state_dim,
        edge_state_dim,
        edge_hidden_sizes,
        node_hidden_sizes,
        n_prop_layers,
        share_prop_params=False,
        edge_net_init_scale=0.1,
        node_update_type="residual",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
        layer_class=GraphPropLayer,
        prop_type="embedding",
        name="graph-embedding-net",
    ):
        super(GraphEmbeddingNet, self).__init__()

        self._encoder = encoder
        self._aggregator = aggregator
        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        self._n_prop_layers = n_prop_layers
        self._share_prop_params = share_prop_params
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._layer_norm = layer_norm
        self._prop_layers = []
        self._prop_layers = nn.ModuleList()
        self._layer_class = layer_class
        self._prop_type = prop_type
        self.build_model()

    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            self._node_state_dim,
            self._edge_state_dim,
            self._edge_hidden_sizes,
            self._node_hidden_sizes,
            edge_net_init_scale=self._edge_net_init_scale,
            node_update_type=self._node_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm,
            prop_type=self._prop_type,
        )

    def _apply_layer(
        self, layer, node_states, from_idx, to_idx, graph_idx, n_graphs, edge_features
    ):
        del graph_idx, n_graphs
        node_states = layer(node_states, from_idx, to_idx, edge_features=edge_features)
        return node_states

    def build_model(self):
        if len(self._prop_layers) < self._n_prop_layers:
            # build the layers
            for i in range(self._n_prop_layers):
                if i == 0 or not self._share_prop_params:
                    layer = self._build_layer(i)
                else:
                    layer = self._prop_layers[0]
                self._prop_layers.append(layer)

    def forward(
        self, node_features, edge_features, from_idx, to_idx, graph_idx, n_graphs
    ):
        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]

        for layer in self._prop_layers:
            node_states = self._apply_layer(
                layer, node_states, from_idx, to_idx, graph_idx, n_graphs, edge_features
            )
            layer_outputs.append(node_states)

        self._layer_outputs = layer_outputs

        return self._aggregator(node_states, graph_idx, n_graphs)

    def reset_n_prop_layers(self, n_prop_layers):
        self._n_prop_layers = n_prop_layers

    @property
    def n_prop_layers(self):
        return self._n_prop_layers

    def get_layer_outputs(self):
        if hasattr(self, "_layer_outputs"):
            return self._layer_outputs
        else:
            raise ValueError("No layer outputs available.")


def pairwise_euclidean_similarity(x, y):
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x**2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y**2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    "euclidean": pairwise_euclidean_similarity,
    "dotproduct": pairwise_dot_product_similarity,
    "cosine": pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim, k=3):
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)

    top_values_x, top_indices_x = torch.topk(a, k, dim=1)
    top_values_y, top_indices_y = torch.topk(a, k, dim=0)

    return (
        attention_x,
        attention_y,
        a_x,
        a_y,
        (
            top_values_x,
            top_indices_x,
            top_values_y,
            top_indices_y,
        ),
    )


def batch_block_pair_attention(data, block_idx, n_blocks, similarity="dotproduct"):
    if not isinstance(n_blocks, int):
        raise ValueError("n_blocks (%s) has to be an integer." % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError("n_blocks (%d) must be a multiple of 2." % n_blocks)

    sim = get_pairwise_similarity(similarity)

    results = []
    topk_cross_attentions = []
    a_x_s = []
    a_y_s = []

    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        (
            attention_x,
            attention_y,
            a_x,
            a_y,
            topk_cross_attention,
        ) = compute_cross_attention(x, y, sim)
        a_x_s.append(a_x)
        a_y_s.append(a_y)
        results.append(attention_x)
        results.append(attention_y)
        topk_cross_attentions.append(topk_cross_attention)
    results = torch.cat(results, dim=0)

    return results, topk_cross_attentions, a_x_s, a_y_s


class GraphPropMatchingLayer(GraphPropLayer):
    def forward(
        self,
        node_states,
        from_idx,
        to_idx,
        graph_idx,
        n_graphs,
        similarity="dotproduct",
        edge_features=None,
        node_features=None,
    ):
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features
        )

        (
            cross_graph_attention,
            topk_cross_attentions,
            a_x_s,
            a_y_s,
        ) = batch_block_pair_attention(
            node_states, graph_idx, n_graphs, similarity=similarity
        )
        attention_input = node_states - cross_graph_attention
        return self._compute_node_update(
            node_states,
            [aggregated_messages, attention_input],
            node_features=node_features,
        )


class GraphMatchingNet(GraphEmbeddingNet):
    def __init__(
        self,
        encoder,
        aggregator,
        node_state_dim,
        edge_state_dim,
        edge_hidden_sizes,
        node_hidden_sizes,
        n_prop_layers,
        share_prop_params=False,
        edge_net_init_scale=0.1,
        node_update_type="residual",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
        layer_class=GraphPropLayer,
        similarity="dotproduct",
        prop_type="embedding",
    ):
        super(GraphMatchingNet, self).__init__(
            encoder,
            aggregator,
            node_state_dim,
            edge_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            n_prop_layers,
            share_prop_params=share_prop_params,
            edge_net_init_scale=edge_net_init_scale,
            node_update_type=node_update_type,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            layer_class=GraphPropMatchingLayer,
            prop_type=prop_type,
        )
        self._similarity = similarity

    def _apply_layer(
        self, layer, node_states, from_idx, to_idx, graph_idx, n_graphs, edge_features
    ):
        return layer(
            node_states,
            from_idx,
            to_idx,
            graph_idx,
            n_graphs,
            similarity=self._similarity,
            edge_features=edge_features,
        )
