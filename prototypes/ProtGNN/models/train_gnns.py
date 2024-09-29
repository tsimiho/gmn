import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from Configures import data_args, model_args, train_args
from load_dataset import get_dataloader, get_dataset
from models import GnnNets, GnnNets_NC
from my_mcts import mcts


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def append_record(info):
    f = open("./log/hyper_search", "a")
    f.write(info)
    f.write("\n")
    f.close()


def concrete_sample(log_alpha, beta=1.0, training=True):
    """Sample from the instantiation of concrete distribution when training
    \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
    """
    if training:
        random_noise = torch.rand(log_alpha.shape)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        gate_inputs = (random_noise + log_alpha) / beta
        gate_inputs = gate_inputs.sigmoid()
    else:
        gate_inputs = log_alpha.sigmoid()

    return gate_inputs


def edge_mask(inputs, training=None):
    x, embed, edge_index, prot, tmp = inputs
    nodesize = embed.shape[0]
    feature_dim = embed.shape[1]
    f1 = embed.unsqueeze(1).repeat(1, nodesize, 1).reshape(-1, feature_dim)
    f2 = embed.unsqueeze(0).repeat(nodesize, 1, 1).reshape(-1, feature_dim)
    f3 = prot.unsqueeze(0).repeat(nodesize * nodesize, 1)
    # using the node embedding to calculate the edge weight
    f12self = torch.cat([f1, f2, f3], dim=-1)
    h = f12self
    for elayer in elayers:
        h = elayer(h)
    values = h.reshape(-1)
    values = concrete_sample(values, beta=tmp, training=training)
    mask_sigmoid = values.reshape(nodesize, nodesize)

    sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
    edge_mask = sym_mask[edge_index[0], edge_index[1]]

    return edge_mask


def clear_masks(model):
    """clear the edge weights to None"""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def set_masks(model, edgemask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edgemask


def prototype_subgraph_similarity(x, prototype):
    distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
    similarity = torch.log((distance + 1) / (distance + 1e-4))
    return distance, similarity


elayers = nn.ModuleList()
elayers.append(nn.Sequential(nn.Linear(128 * 3, 64), nn.ReLU()))
elayers.append(nn.Sequential(nn.Linear(64, 8), nn.ReLU()))
elayers.append(nn.Linear(8, 1))


# train for graph classification
def train_GC(clst, sep):
    # attention the multi-task here
    print(clst)
    print(sep)

    # Load the dataset splits
    train_set = torch.load("ba2motifs_train_split.pt", weights_only=False)
    test_set = torch.load("ba2motifs_test_split.pt", weights_only=False)

    input_dim = train_set[0].x.shape[1]  # Get input dimension from node features
    output_dim = int(
        torch.max(torch.tensor([graph.y.item() for graph in train_set])) + 1
    )  # Number of classes

    print("start training model==================")
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        gnnNets.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
    )

    # Calculate average number of nodes and edges
    avg_nodes = sum(graph.x.shape[0] for graph in train_set) / len(train_set)
    avg_edges = sum(graph.edge_index.shape[1] // 2 for graph in train_set) / len(
        train_set
    )
    print(
        f"graphs {len(train_set)}, avg_nodes {avg_nodes:.4f}, avg_edges {avg_edges:.4f}"
    )

    best_acc = 0.0
    early_stop_count = 0

    # Training loop
    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        # Prototype projection (if applicable)
        if epoch >= train_args.proj_epochs and epoch % 10 == 0:
            gnnNets.eval()
            for i in range(output_dim * model_args.num_prototypes_per_class):
                count = 0
                best_similarity = 0
                label = i // model_args.num_prototypes_per_class
                for graph in train_set:
                    if graph.y == label:
                        count += 1
                        coalition, similarity, prot = mcts(
                            graph, gnnNets, gnnNets.model.prototype_vectors[i]
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= 10:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print("Projection of prototype completed")
                        break

        # Training the model
        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for graph in train_set:  # Instead of a dataloader, use train_set directly
            logits, probs, _, _, min_distances = gnnNets(graph)
            loss = criterion(logits, graph.y)

            # Cluster, separation, sparsity, and diversity losses
            prototypes_of_correct_class = torch.t(
                gnnNets.model.prototype_class_identity[:, graph.y].bool()
            )
            cluster_cost = torch.mean(
                torch.min(
                    min_distances[prototypes_of_correct_class].reshape(
                        -1, model_args.num_prototypes_per_class
                    ),
                    dim=1,
                )[0]
            )
            separation_cost = -torch.mean(
                torch.min(
                    min_distances[~prototypes_of_correct_class].reshape(
                        -1, (output_dim - 1) * model_args.num_prototypes_per_class
                    ),
                    dim=1,
                )[0]
            )
            l1_mask = 1 - torch.t(gnnNets.model.prototype_class_identity)
            l1 = (gnnNets.model.last_layer.weight * l1_mask).norm(p=1)

            # Diversity loss
            ld = 0
            for k in range(output_dim):
                p = gnnNets.model.prototype_vectors[
                    k
                    * model_args.num_prototypes_per_class : (k + 1)
                    * model_args.num_prototypes_per_class
                ]
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]) - 0.3
                matrix2 = torch.zeros(matrix1.shape)
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            # Total loss
            loss = (
                loss
                + clst * cluster_cost
                + sep * separation_cost
                + 5e-4 * l1
                + 0.00 * ld
            )

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            # Record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(graph.y).cpu().numpy())

        # Report train msg
        print(
            f"Train Epoch:{epoch} | Loss: {np.mean(loss_list):.3f} | Ld: {np.mean(ld_loss_list):.3f} | Acc: {np.mean(acc):.3f}"
        )

        # Evaluate the model on the test set (instead of eval dataloader)
        eval_state = evaluate_GC(test_set, gnnNets, criterion)
        print(
            f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}"
        )

        # Early stopping and checkpoint saving logic
        if eval_state["acc"] > best_acc:
            early_stop_count = 0
            best_acc = eval_state["acc"]
            save_best(
                ckpt_dir, epoch, gnnNets, model_args.model_name, best_acc, is_best=True
            )
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

    print(f"The best validation accuracy is {best_acc}.")


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {
            "loss": np.average(loss_list),
            "acc": np.concatenate(acc, axis=0).mean(),
        }

    return eval_state


def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {
        "loss": np.average(loss_list),
        "acc": np.average(np.concatenate(acc, axis=0).mean()),
    }

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def predict_GC(test_dataloader, gnnNets):
    """
    return: pred_probs --  np.array : the probability of the graph class
            predictions -- np.array : the prediction class for each graph
    """
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)

            ## record
            _, prediction = torch.max(logits, -1)
            predictions.append(prediction)
            pred_probs.append(probs)

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return pred_probs, predictions


# train for node classification task
def train_NC():
    print("start loading data====================")
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(
        f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}"
    )

    # save path for model
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.isdir(os.path.join("checkpoint", f"{data_args.dataset_name}")):
        os.mkdir(os.path.join("checkpoint", f"{data_args.dataset_name}"))
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"

    data = dataset[0]
    gnnNets_NC = GnnNets_NC(input_dim, output_dim, model_args)
    gnnNets_NC
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        gnnNets_NC.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
    )

    best_val_loss = float("inf")
    best_acc = 0
    val_loss_history = []
    early_stop_count = 0
    for epoch in range(1, train_args.max_epochs + 1):
        gnnNets_NC.train()
        logits, prob, _, min_distances = gnnNets_NC(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        # cluster loss
        prototypes_of_correct_class = torch.t(
            gnnNets_NC.model.prototype_class_identity[:, data.y]
        )
        cluster_cost = torch.mean(
            torch.min(min_distances * prototypes_of_correct_class, dim=1)[0]
        )

        # seperation loss
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        separation_cost = -torch.mean(
            torch.min(min_distances * prototypes_of_wrong_class, dim=1)[0]
        )

        # sparsity loss
        l1_mask = 1 - torch.t(gnnNets_NC.model.prototype_class_identity)
        l1 = (gnnNets_NC.model.last_layer.weight * l1_mask).norm(p=1)

        # diversity loss
        ld = 0
        for k in range(output_dim):
            p = gnnNets_NC.model.prototype_vectors[
                k
                * model_args.num_prototypes_per_class : (k + 1)
                * model_args.num_prototypes_per_class
            ]
            p = F.normalize(p, p=2, dim=1)
            matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]) - 0.3
            matrix2 = torch.zeros(matrix1.shape)
            ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

        loss = loss + 0.1 * cluster_cost + 0.1 * separation_cost + 0.001 * ld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_info = evaluate_NC(data, gnnNets_NC, criterion)
        eval_info["epoch"] = epoch

        if eval_info["val_loss"] < best_val_loss:
            best_val_loss = eval_info["val_loss"]
            val_acc = eval_info["val_acc"]

        val_loss_history.append(eval_info["val_loss"])

        # only save the best model
        is_best = eval_info["val_acc"] > best_acc

        if eval_info["val_acc"] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_info["val_acc"]
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(
                ckpt_dir,
                epoch,
                gnnNets_NC,
                model_args.model_name,
                eval_info["val_acc"],
                is_best,
            )
            print(
                f'Epoch {epoch}, Train Loss: {eval_info["train_loss"]:.4f}, '
                f'Train Accuracy: {eval_info["train_acc"]:.3f}, '
                f'Val Loss: {eval_info["val_loss"]:.3f}, '
                f'Val Accuracy: {eval_info["val_acc"]:.3f}'
            )

    # report test msg
    checkpoint = torch.load(
        os.path.join(ckpt_dir, f"{model_args.model_name}_best.pth"), weights_only=False
    )
    gnnNets_NC.update_state_dict(checkpoint["net"])
    eval_info = evaluate_NC(data, gnnNets_NC, criterion)
    print(
        f'Test Loss: {eval_info["test_loss"]:.4f}, Test Accuracy: {eval_info["test_acc"]:.3f}'
    )


def evaluate_NC(data, gnnNets_NC, criterion):
    eval_state = {}
    gnnNets_NC.eval()

    with torch.no_grad():
        for key in ["train", "val", "test"]:
            mask = data["{}_mask".format(key)]
            logits, probs, _, _ = gnnNets_NC(data)
            loss = criterion(logits[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            ## record
            eval_state["{}_loss".format(key)] = loss
            eval_state["{}_acc".format(key)] = acc

    return eval_state


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print("saving....")
    gnnNets.to("cpu")
    state = {"net": gnnNets.state_dict(), "epoch": epoch, "acc": eval_acc}
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f"{model_name}_best.pth"
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        # best_pth_name = f"{model_name}_best_clst_{args.clst}_sep_{args.sep}.pth"
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch implementation of ProtGNN")
    parser.add_argument("--clst", type=float, default=0.01, help="cluster")
    parser.add_argument("--sep", type=float, default=0.0, help="separation")
    args = parser.parse_args()
    train_GC(args.clst, args.sep)
