from load_dataset import *

dataset = MUTAGDataset(root="datasets", name="MUTAG")
syn_dataset = CustomGraphDataset(root="my_data")

print((syn_dataset[0]))
