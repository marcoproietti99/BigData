import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import networkx as nx
import pandas as pd
import ast
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Example load function for args
def load():
    class Args:
        pass
    args = Args()
    args.DATASET_PATH = ''
    args.COMPLETE_IGS_NAME = 'balanced_WorstTimeCustomerRemoval'
    return args

args = load()

def custom_mapping(valore):
    if valore < 0:
        return 0
    elif valore == 0:
        return 1
    elif valore > 0 and valore <= 10:
        return 2
    elif valore > 10:
        return 3

# Leggi il dataset Excel
df = pd.read_csv("balanced_WorstTimeCustomerRemoval.csv",sep=",")
df = df[["Instance's Name","Initial Solution", "OF_Diff"]]
df['Results'] = df['OF_Diff'].apply(custom_mapping)
"""
block1 = shuffle(df.iloc[0:5, :])
block2 = shuffle(df.iloc[54330:54335, :])
block3 = shuffle(df.iloc[108660:108665, :])
block4 = shuffle(df.iloc[162990:162995, :])
df = pd.concat([block1, block2, block3, block4])
"""
df = shuffle(df)
df.reset_index(drop=True, inplace=True)
df = df.drop(["OF_Diff"],axis=1)

def create_ordered_graph_from_nodes(nodes):
    G = nx.DiGraph()
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    return G

type_map = {'d': [1, 0, 0], 'f': [0, 1, 0], 'c': [0, 0, 1]}

def get_feature_vector(df,node):
    row = df[df['StringID'] == node]
    if not row.empty:
        node_type = row['Type'].values[0]
        x, y = row['x_normalized'].values[0], row['y_normalized'].values[0]
        return type_map[node_type] + [x, y]
    else:
        return [0, 0, 0, 0, 0] 

# Estrarre le liste di grafi per ciascuna riga
all_graphs = []
for idx, row in df.iterrows():
    row_paths = []
    lol = ast.literal_eval(row["Initial Solution"])
    graphs = create_ordered_graph_from_nodes([node for path in lol for node in path])
    """
    print(graphs)
    pos = nx.spring_layout(graphs)
    nx.draw(graphs, pos, with_labels=True, node_size=700, node_color='lightblue', arrows=True, arrowsize=20)
    plt.show()
    """
    all_graphs.append(graphs)

class TraceDataset(InMemoryDataset):
    def __init__(self, root, graphs, transform=None, pre_transform=None):
        self.graphs = graphs
        super(TraceDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.labels = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{args.COMPLETE_IGS_NAME}_par.pt']

    def process(self):
        i = 0
        data_list = []
        labels_list = []

        for graph_set in self.graphs:

            instanceName = df.iloc[i,0]
            file_path = f"instances\\{instanceName.replace('.txt', '_normalized.csv')}"
            instance = pd.read_csv(file_path)
            graph = ast.literal_eval(df.iloc[i,1])

            x1 = []
            """
            for node_id in graph_set.nodes:
                node_features = get_feature_vector(instance,node_id)
                x1.append(node_features)
                print(node_id)
                #print(node_features)
            """
            for sublist in graph:
                for node_id in sublist:
                    node_features = get_feature_vector(instance,node_id)
                    x1.append(node_features)
            x = torch.tensor(x1, dtype=torch.float)
            adj = nx.to_scipy_sparse_array(graph_set)
            adj = adj.tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([col, row], dim=0)
            y = torch.tensor([i])
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
            labels_list.append(df.iloc[i,2])
            i = i+1
        data, slices = self.collate(data_list)
        torch.save((data, slices, labels_list), self.processed_paths[0])

if __name__ == '__main__':
    dataset = TraceDataset(root=args.DATASET_PATH, graphs=all_graphs)
