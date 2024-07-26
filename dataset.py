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
    args.COMPLETE_IGS_NAME = 'balanced_final_WorstTimeCustomerRemoval'
    return args
 
#dataset elaborati: balanced_ZoneCustomerRemoval, balanced_WorstTimeCustomerRemoval, balanced_WorstDistanceCustomerRemoval, balanced_TimeBasedCustomerRemoval, balanced_ShawCustomerRemoval, balanced_RandomRouteCustomerRemoval, balanced_RandomCustomerRemoval, balanced_GreedyRouteRemoval, balanced_DemandBasedCustomerRemoval
args = load()
 
def custom_mapping(valore):
    if valore == 'Negativo':
        return 0
    elif valore == 'Neutro':
        return 1
    elif valore == 'Positivo':
        return 2
    elif valore == 'Molto Positivo':
        return 3
 
# Leggi il dataset Excel
df = pd.read_csv("balanced_final_WorstTimeCustomerRemoval.csv",sep=",")
 
df['Results'] = df['Results'].apply(custom_mapping)

"""
block1 = shuffle(df.iloc[0:5, :])
block2 = shuffle(df.iloc[54330:54335, :])
block3 = shuffle(df.iloc[108660:108665, :])
block4 = shuffle(df.iloc[145074:145079, :])
df = pd.concat([block1, block2, block3, block4])
"""
 
df = shuffle(df)
df.reset_index(drop=True, inplace=True)
df = df.drop(["Unnamed: 0"], axis=1)
 
def create_ordered_graph_from_nodes(nodes):
    G = nx.DiGraph()
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    return G
 
# Estrarre le liste di grafi per ciascuna riga
all_graphs = []
for idx, row in df.iterrows():
    row_paths = []
    lol = ast.literal_eval(row["Initial Solution"])
    graphs = create_ordered_graph_from_nodes([node for path in lol for node in path])
    all_graphs.append(graphs)
 
def dict_attr(graphs):
    # Estrarre tutti i nodi unici da tutti i grafi
    unique_nodes = set()
    for graph in graphs:
        unique_nodes.update(graph.nodes)
   
    unique_nodes_list = list(unique_nodes)
    max_nodes = len(unique_nodes_list)
    attr = {}
   
    # Generare gli attributi one-hot per tutti i nodi unici
    for node in unique_nodes_list:
        one_hot = [0] * max_nodes
        one_hot[unique_nodes_list.index(node)] = 1
        attr[node] = one_hot
   
    return attr
 
def dict_target(num_graphs):
    return {f'graph_{i}': i for i in range(num_graphs)}
 
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
        attr_event = dict_attr(self.graphs)
        labels_list = []
        for graph_set in self.graphs:
            graph = ast.literal_eval(df.iloc[i,1])
            x1 = []
            for sublist in graph:
                for node_id in sublist:
                    #print(node_id)
                    node_features = attr_event[node_id]
                    x1.append(node_features)
                    #print(node_features)
            x = torch.tensor(x1, dtype=torch.float)
            #print(x)
            """
            pos = nx.spring_layout(graph_set)
            nx.draw(graph_set, pos, with_labels=True, node_size=700, node_color='lightblue', arrows=True, arrowsize=20)
            plt.show()
            """
            adj = nx.to_scipy_sparse_array(graph_set)
            adj = adj.tocoo()
            #print(adj)
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([col, row], dim=0)
            #print(edge_index)
            y = torch.tensor([i])
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
            labels_list.append(df.iloc[i,0])
            i = i+1
        data, slices = self.collate(data_list)
        torch.save((data, slices, labels_list), self.processed_paths[0])
 
if __name__ == '__main__':
    dataset = TraceDataset(root=args.DATASET_PATH, graphs=all_graphs)
