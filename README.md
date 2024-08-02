# Addestramento modello per classificazione mosse da effettuare da veicoli elettrici per consegne

In questo progetto si punta a utilizzare una rete neurale a grafo per addestrare un modello che classifichi in 4 classi (negativa, neutra, positiva, molto positiva), le mosse effettuabili da veicoli elettrici predisposti a consegne di pacchi, in cui i posto dove partono, sostano e consegnano rappresentano i nodi del grafo e la strada percorsa tra questi nodi gli archi.

## Utilizzo
Scaricare la cartella github e aggiungere a posteriori i file con dataset da elaborare a seconda dei casi: balanced_final_WorstTimeCustomerRemoval.csv è da usare col file "dataset.py", balanced_WorstTimeCustomerRemoval.csv con "dataset_normalizzato.py". I file presenti nella cartella sono invece i due file per l'elaborazione del dataset: scrivendo nel terminale entrando nella cartella di progetto
```sh
python dataset.py
```

Viene elaborata una lista contenente i nodi elaborati come tensori one-hot encoded, dove l'1 equivale al nodo univoco mentre le posizioni di tutti gli altri sono a 0. Invece con
```sh
python dataset_normalizzato.py
```
 crea sempre una lista contenente tensori per ogni nodo ma ogni tensore ha prima tre valori che possono essere true o false a seconda del tipo di nodo (deposito, stazione, cliente) e i due successivi elementi del tensore sono i valori x e y dello specifico nodo normalizzati. I file con i valori di x e y normalizzati sono già tutti contenuti nella cartella "instances" ma è stato anche creato il file "normalization.py" se si volesse ripetere l'operazione dato che prende i valori di x e y contenuti nella cartella "data"(dove non sono ancora normalizzati), trova i massimi e minimi locali per ogni file poi prende i massimi e minimi globali e associa quindi un valore tra 0 ed 1 ad ogni coordinata. Dopo aver elaborato i dataset si può eseguire il comando 
 ```sh
python TRAINING2.py
```
 che è il file che serve per l'addestramento del modello. A seconda di quale dataset si è elaborato su "COMPLETE_IGS_NAME" va scritto o "balanced_final_WorstTimeCustomerRemoval" o "balanced_WorstTimeCustomerRemoval". Il file contiene una grid search dove è possibile cambiare i parametri per l'addestramento. Oltre questi file, gli altri contenuti sono "DGCNN.py" che contiene l'effettiva rete neurale a grafo con i suoi strati e "config.py" che contiene la configurazione dei path e dei parametri di base della rete.


