import torch
from torch.nn import Linear, Conv1d, BatchNorm1d, ReLU, Softmax, ModuleList, Dropout, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SortAggregation


class DGCNN(torch.nn.Module):
    def __init__(self, dataset, num_layers, dropout, num_neurons, k):
        super(DGCNN, self).__init__()
        # SAGEConv rappresenta il layer di convoluzione spaziale su grafi
        self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=num_neurons)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(in_channels=num_neurons, out_channels=num_neurons))
        self.conv1d = Conv1d(in_channels=num_neurons, out_channels=32, kernel_size=num_layers)

        # Linear rappresenta il layer che applica una trasformazione lineare
        self.lin1 = Linear(in_features=(32 * (k - num_layers + 1)),
                           out_features=num_neurons)
        self.dropout = Dropout(p=dropout)
        self.lin2 = Linear(in_features=num_neurons, out_features=4)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, k):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        input_data = self.conv1(x, edge_index)
        x = F.relu(input=input_data)

        for conv in self.convs:
            conv_input = conv(x, edge_index)
            x = F.relu(input=conv_input)
        # layer di pooling
        sort_aggr = SortAggregation(k=k)
        x = sort_aggr(x, batch)

        # modifica la struttura del vettore per passarlo al layer conv1d (devono avere n°nodi=k)
        x = x.view(len(x), k, -1).permute(dims=[0, 2, 1])
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    