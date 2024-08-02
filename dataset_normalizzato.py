import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import networkx as nx
import pandas as pd
import ast
import os
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

df = shuffle(df)
df.reset_index(drop=True, inplace=True)
df = df.drop(["OF_Diff"],axis=1)

# funzione per creare un grafo NetworkX dalle righe del dataset 
def create_ordered_graph_from_nodes(nodes):
    G = nx.DiGraph()
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    return G

# one-hot encoding per ogni tipo di nodo
type_map = {'d': [1, 0, 0], 'f': [0, 1, 0], 'c': [0, 0, 1]}

# funzione per creare le feature per ogni nodo, il vettore delle feature è formato da 5 righe: le prime 3 rappresentano il tipo di nodo (d, f o c) 
# mentre le restanti due rappresentano le coordinate x e y normalizzate 
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

            instanceName = df.iloc[i,0] # ogni riga fa parte di un'istanza diversa
            file_path = os.path.join("instances", instanceName.replace(".txt", "_normalized.csv"))
            instance = pd.read_csv(file_path)
            graph = ast.literal_eval(df.iloc[i,1])

            x1 = []
            
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
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
            labels_list.append(df.iloc[i,2])
            i = i+1
        data, slices = self.collate(data_list)
        torch.save((data, slices, labels_list), self.processed_paths[0])

if __name__ == '__main__':
    dataset = TraceDataset(root=args.DATASET_PATH, graphs=all_graphs)
