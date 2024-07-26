import pandas as pd
import os
import io

# Funzione per leggere i valori x e y da un file
def read_xy_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    start_info_idx = next(i for i, line in enumerate(lines) if line.startswith("Vehicle"))
    table_lines = lines[:start_info_idx]
    columns = ["StringID", "Type", "x", "y", "demand", "ReadyTime", "DueDate", "ServiceTime"]
    
    table_data = io.StringIO(''.join(table_lines))
    data = pd.read_csv(table_data, delim_whitespace=True, skiprows=1, names=columns)
    
    return data[['x', 'y']]

# Passaggio 1: Trovare i valori minimi e massimi globali di x e y
global_min_x = float('inf')
global_max_x = float('-inf')
global_min_y = float('inf')
global_max_y = float('-inf')

input_dir = 'C:\\Users\\marco\\Downloads\\EVRPTW 1\\EVRPTW\\data'  # Sostituisci con il percorso della cartella di input
output_dir = 'instances'  # Sostituisci con il percorso della cartella di output

for file_name in os.listdir(input_dir):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_dir, file_name)
        xy_values = read_xy_values(file_path)
        global_min_x = min(global_min_x, xy_values['x'].min())
        global_max_x = max(global_max_x, xy_values['x'].max())
        global_min_y = min(global_min_y, xy_values['y'].min())
        global_max_y = max(global_max_y, xy_values['y'].max())

print(f"Global min x: {global_min_x}, Global max x: {global_max_x}")
print(f"Global min y: {global_min_y}, Global max y: {global_max_y}")

# Funzione per normalizzare i dati e salvarli in un nuovo file
def normalize_and_save(file_path, output_dir, global_min_x, global_max_x, global_min_y, global_max_y):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    start_info_idx = next(i for i, line in enumerate(lines) if line.startswith("Vehicle"))
    table_lines = lines[:start_info_idx]
    columns = ["StringID", "Type", "x", "y", "demand", "ReadyTime", "DueDate", "ServiceTime"]
    
    table_data = io.StringIO(''.join(table_lines))
    data = pd.read_csv(table_data, delim_whitespace=True, skiprows=1, names=columns)
    
    data['x_normalized'] = (data['x'] - global_min_x) / (global_max_x - global_min_x)
    data['y_normalized'] = (data['y'] - global_min_y) / (global_max_y - global_min_y)
    
    normalized_data = data[['StringID', 'Type', 'x_normalized', 'y_normalized']]
    
    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_normalized.csv'
    output_file_path = os.path.join(output_dir, output_file_name)
    normalized_data.to_csv(output_file_path, index=False)
    print(f"File {file_path} normalizzato e salvato come {output_file_path}")

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Passaggio 2: Normalizzare e salvare i file utilizzando i valori globali
for file_name in os.listdir(input_dir):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_dir, file_name)
        normalize_and_save(file_path, output_dir, global_min_x, global_max_x, global_min_y, global_max_y)
