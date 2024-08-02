import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from DGCNN import DGCNN
from pandas import DataFrame
import numpy as np
from itertools import product
from time import time
import random

from os import makedirs
from os.path import join, exists
from datetime import datetime

# Importa le configurazioni e le costanti
from config import load

COMPLETE_IGS_NAME = 'balanced_final_WorstTimeCustomerRemoval' # nel caso di dataset elaborato dal file "dataset" è da sostituire con balanced_final_WorstTimeCustomerRemoval
DATASET_PATH = ''
NET_RESULTS_PATH = 'results'

# Carica le configurazioni
args = load()
plt.ioff()

# Fissiamo il seed per la riproducibilità
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Definizione del dataset personalizzato
class TraceDataset(InMemoryDataset):
    def __init__(self):
        super(TraceDataset, self).__init__(DATASET_PATH)
        self.data, self.slices, self.labels = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{COMPLETE_IGS_NAME}_par.pt']

    def get(self, idx):
        graph_data = self.get_example(idx)
        graph_data.y = torch.tensor([self.labels[idx]], dtype=torch.long) # aggiungo un attributo y a graph data per salvare la classe della riga "idx"
        return graph_data

    def get_example(self, idx): # funzione per ottenere la riga "idx" del dataset
        x = self.data.x[self.slices['x'][idx]:self.slices['x'][idx+1]] # x rappresenta le feature dei nodi
        edge_index = self.data.edge_index[:, self.slices['edge_index'][idx]:self.slices['edge_index'][idx+1]] # edge index rappresenta gli archi
        return Data(x=x, edge_index=edge_index)

# Funzione per dividere il dataset in train e test
def split_target(G, per):
    dict = {i: [] for i in range(4)}  # 4 è il numero delle classi

    for idx in range(len(G)):
        graph = G.get(idx)
        label = int(graph.y.item())
        dict[label].append(graph)

    train, test = [], []
    for graphs in dict.values():
        l = int(len(graphs) * per / 100)
        train.extend(graphs[:l])
        test.extend(graphs[l:])

    return train, test

# Funzione per plot della matrice di confusione
def plot_confusion_matrix(reals, predictions, keyword, epoch, path):
    classes = ['0', '1', '2', '3']  # Identificatore delle classi (0: Negativo, 1: Neutro, 2: Positivo, 3: Molto Positivo)

    cmt = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    for p in zip(reals, predictions):
        tl, pl = p
        cmt[int(tl), int(pl)] += 1

    plt.figure(figsize=(15, 15))
    plt.imshow(cmt, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in product(range(cmt.shape[0]), range(cmt.shape[1])):
        plt.text(j, i, format(cmt[i, j], 'd'), horizontalalignment="center",
                 color="white" if cmt[i, j] > cmt.max() / 2. else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(path, f'cm_{keyword}_{epoch}.png'))
    plt.close()

# Funzione per plot delle metriche combinate
def plot_comb_metrics(result_df, path):
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(result_df['train_loss'].tolist(), label='Loss train')
    plt.plot(result_df['test_loss'].tolist(), label='Loss test')
    plt.legend()
    plt.savefig(join(path, 'losses.png'))
    plt.close('all')

    plt.xlabel('Epochs')
    plt.ylabel('Weighted F1-score')
    plt.plot(result_df['train_f1'].tolist(), label='Weighted F1-score train')
    plt.plot(result_df['test_f1'].tolist(), label='Weighted F1-score test')
    plt.legend()
    plt.savefig(join(path, 'f1.png'))
    plt.close('all')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(result_df['train_accuracy'].tolist(), label='Accuracy train')
    plt.plot(result_df['test_accuracy'].tolist(), label='Accuracy test')
    plt.legend()
    plt.savefig(join(path, 'acc.png'))
    plt.close('all')

if __name__ == '__main__':
    # Impostazione del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G = TraceDataset()

    dropout, patience, perc_split = args.dropout, args.patience, args.per
    train, test = split_target(G, perc_split)
    criterion = torch.nn.CrossEntropyLoss()

    # Impostazione dei valori di ricerca
    if args.grid_search:
        epochs = [100]
        k_values = [3]
        num_layers_values = [3, 5]
        lr_values = [10e-4]
        batch_size_values = [64]
        num_neurons = [64]
        n_of_comb = (len(epochs) * len(k_values) * len(num_layers_values)
                     * len(lr_values) * len(batch_size_values) * len(num_neurons))
    else:
        batch_size_values = [args.batch_size]
        epochs = [args.num_epochs]
        k_values = [args.k]
        num_layers_values = [args.num_layers]
        lr_values = [args.learning_rate]
        num_neurons = [args.num_neurons]
        n_of_comb = 1

    actual_comb = 1
    current_timestamp = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    for batch_size in batch_size_values:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

        for total_epochs in epochs:
            for k in k_values:
                for num_layers in num_layers_values:
                    for num_neuron in num_neurons:
                        for lr in lr_values:
                            if k < num_layers:
                                continue

                            data_comb = f"epochs_{total_epochs}_batchsize_{batch_size}_k_{k}"\
                                        f"_numlayers_{num_layers}"\
                                        f"_numneurons_{num_neuron}_lr_{lr}"

                            results_df = DataFrame(columns=['combination', 'best', 'total_epochs', 'epoch', 'batchsize',
                                                            'k', 'num_layers', 'num_neurons', 'learning_rate',
                                                            'train_loss', 'train_accuracy', 'train_f1', 'test_loss',
                                                            'test_accuracy', 'test_f1'])
                            prefix_results_df = DataFrame(columns=['prefix_len', 'y_pred', 'y_true'])
                            comb_path = join(NET_RESULTS_PATH, current_timestamp, data_comb)

                            if not exists(comb_path):
                                makedirs(comb_path)

                            print(f"\n\nStarting combination ({actual_comb}/{n_of_comb}):\n"
                                  f"-> EPOCHS: {total_epochs} | PATIENCE: {patience}\n"
                                  f"-> NUM_NEURONS: {num_neuron} | K: {k} | NUM_LAYERS: {num_layers}\n"
                                  f"-> BATCH_SIZE: {batch_size} | LEARNING RATE: {lr}\n")

                            # Creazione del modello DGCNN
                            model = DGCNN(dataset=G, num_layers=num_layers, dropout=dropout, num_neurons=num_neuron, k=k).to(device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                            best_loss_test, trigger_times = 0, 0

                            for epoch in range(total_epochs):
                                print(f'\nStarting epoch: {epoch + 1}/{total_epochs}')
                                start_time = time()

                                epoch_path = join(comb_path, f'{epoch}')
                                if not exists(epoch_path):
                                    makedirs(epoch_path)

                                model.train(True)
                                train_loss, correct_predictions = 0, 0
                                predictions, reals = torch.tensor([]).to(device), torch.tensor([]).to(device)
                                for batch in train_loader:
                                    batch = batch.to(device)  # Sposta il batch sulla GPU
                                    optimizer.zero_grad()
                                    out = model(batch, k)
                                    loss = criterion(out, batch.y.view(-1))  # Batch.y dovrebbe essere un tensore di etichette
                                    pred = out.argmax(dim=1)
                                    predictions = torch.cat((predictions, pred), dim=0)
                                    reals = torch.cat((reals, batch.y.view(-1)), dim=0)
                                    loss.backward()
                                    optimizer.step()
                                    train_loss += loss.item() * batch.num_graphs
                                    correct_predictions += int((pred == batch.y).sum())
                                end_time = time()

                                train_acc = correct_predictions / len(train_loader.dataset)
                                train_loss = train_loss / len(train_loader.dataset)
                                train_f1 = f1_score(reals.cpu(), predictions.cpu(), average='weighted')
                                plot_confusion_matrix(reals.cpu(), predictions.cpu(), 'train', epoch, epoch_path)

                                print(f'-> Training done in {end_time - start_time} s'
                                      f'\n\t-> Loss: {train_loss:.4f}'
                                      f'\n\t-> Accuracy: {(train_acc * 100):.4f}%'
                                      f'\n\t-> Weighted F1: {(train_f1 * 100):.4f}%')

                                start_time = time()
                                model.eval()
                                test_loss, correct_predictions = 0, 0
                                with torch.no_grad():
                                    predictions, reals = torch.tensor([]).to(device), torch.tensor([]).to(device)
                                    for batch in test_loader:
                                        batch = batch.to(device)  # Sposta il batch sulla GPU
                                        out = model(batch, k)
                                        loss = criterion(out, batch.y)
                                        pred = out.argmax(dim=1)
                                        predictions = torch.cat((predictions, pred), dim=0)
                                        reals = torch.cat((reals, batch.y), dim=0)
                                        test_loss += loss.item() * batch.num_graphs
                                        correct_predictions += int((pred == batch.y).sum())

                                end_time = time()
                                test_acc = correct_predictions / len(test_loader.dataset)
                                test_loss = test_loss / len(test_loader.dataset)
                                test_f1 = f1_score(reals.cpu(), predictions.cpu(), average='weighted')
                                plot_confusion_matrix(reals.cpu(), predictions.cpu(), 'test', epoch, epoch_path)

                                print(f'-> Test done in {end_time - start_time} s'
                                      f'\n\t-> Loss: {test_loss:.4f}'
                                      f'\n\t-> Accuracy: {(test_acc * 100):.4f}%'
                                      f'\n\t-> Weighted F1: {(test_f1 * 100):.4f}%')

                                if epoch == 0 or test_loss <= best_loss_test:
                                    best_loss_test = test_loss
                                    print(f'**** BEST TEST LOSS:{best_loss_test:.4f} ****')
                                    trigger_times = 0
                                else:
                                    trigger_times += 1

                                # saving infos..
                                results_df.loc[len(results_df)] = [data_comb, str(test_loss <= best_loss_test),
                                                                  total_epochs, epoch, batch_size, k,
                                                                  num_layers, num_neuron, lr, train_loss,
                                                                  train_acc, train_f1, test_loss, test_acc,
                                                                  test_f1]
                                results_df.to_csv(join(comb_path, 'results.csv'), header=True, index=False, sep=',')

                                # saving epoch summary..
                                with open(join(epoch_path, f'result_epoch_{epoch}.txt'), 'w') as file:
                                    file.write(f'\n\nMetrics of model ({epoch}/{total_epochs}):'
                                               f'\n\t-> Train loss: {train_loss:.4f}'
                                               f'\n\t-> Train accuracy: {(train_acc * 100):.4f}%'
                                               f'\n\t-> Train weighted F1: {(train_f1 * 100):.4f}%'
                                               f'\n\t-> Test loss: {test_loss:.4f}'
                                               f'\n\t-> Test accuracy: {(test_acc * 100):.4f}%'
                                               f'\n\t-> Test weighted F1: {(test_f1 * 100):.4f}%')

                                if trigger_times >= patience:
                                    print(f'Early stopping!\nBest test loss = {best_loss_test}')
                                    break

                            # plotting combination metrics..
                            plot_comb_metrics(results_df, comb_path)

                            actual_comb += 1




