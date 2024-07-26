from os import mkdir
from os.path import join, abspath, exists, dirname
from argparse import ArgumentParser
import sys

# the name of the xes file
XES_NAME = 'helpdesk_SE'
if len(sys.argv) < 2:
    print(f"\n** INFO: No XES_NAME provided, using default: '{XES_NAME}' **\n")
else:
    XES_NAME = str(sys.argv[1])

# the name of the igs file
G_NAME = f'{XES_NAME}_instance_graphs'

# the name of the petri net generated from BIG
PETRI_NET_NAME = f'{XES_NAME}_petri_net'

# the name of the igs file which contains only timestamps
PARTIAL_IGS_NAME = f'{G_NAME}_partial'

# the name of the igs file which contains all selected features
COMPLETE_IGS_NAME = f'{G_NAME}_complete'

# list where to insert xes names which not contains 'start'/'end' activities for BIG
LOGS_WITH_NO_START_END = ['BPI12', 'BPI12W', 'helpdesk', 'PrepaidTravelCost', 'PermitLog',
                          'RequestForPayment', 'InternationalDeclarations']

BASE_PATH = dirname((abspath(__file__)))

INPUT_PATH = join(BASE_PATH, 'input')
XES_PATH = join(INPUT_PATH, 'xes')
CSV_PATH = join(INPUT_PATH, 'csv')
G_PATH = join(INPUT_PATH, 'g')
OUTPUT_PATH = join(BASE_PATH, 'output')
DATASET_PATH = join(OUTPUT_PATH, 'dataset')
DATASET_PROCESSED_PATH = join(DATASET_PATH, 'processed')
DATASET_G_PATH = join(DATASET_PATH, 'g')
NET_PATH = join(OUTPUT_PATH, 'net')
NET_RESULTS_PATH = join(NET_PATH, 'results')
TMP_STATE_PATH = join(DATASET_G_PATH, 'TMP')
XES_FILE_PATH = join(XES_PATH, f'{XES_NAME}.xes')
PETRI_NET_FILE_PATH = join(XES_PATH, f'{PETRI_NET_NAME}.pnml')
G_FILE_PATH = join(G_PATH, f'{G_NAME}.g')
CASES_TIMESTAMP_FILE_PATH = join(TMP_STATE_PATH, f'{XES_NAME}_case_intervals.csv')
G_WITH_STATE_FILE_PATH = join(DATASET_G_PATH, f'{XES_NAME}_with_state_igs.g')
G_WITH_STATE_PARTIAL_FILE_PATH = join(TMP_STATE_PATH, f'TMP_{XES_NAME}_with_state_igs.g')
RESOURCES_COUNTER_FILE_PATH = join(TMP_STATE_PATH, f'TMP_{XES_NAME}_resources_counter.txt')

ATTRIBUTES_FILE_PATH = join(DATASET_PATH, f'{XES_NAME}_attributes.txt')
TARGET_STD_FILE_PATH = join(DATASET_PATH, f'{XES_NAME}_target_std.txt')
TARGET_PAR_FILE_PATH = join(DATASET_PATH, f'{XES_NAME}_target_par.txt')
MAX_VALUE_PREFIXES_FILE_PATH = join(DATASET_PATH, f'{XES_NAME}_max_value_prefixes.txt')
PARTIAL_IGS_FILE_PATH = join(DATASET_G_PATH, f'{PARTIAL_IGS_NAME}.g')
COMPLETE_IGS_FILE_PATH = join(DATASET_G_PATH, f'{COMPLETE_IGS_NAME}.g')
NUMERICAL_ATTRIBUTES_FILE_PATH = join(DATASET_PATH, f'{XES_NAME}_numerical_attributes.txt')
CATEGORICAL_ATTRIBUTES_FILE_PATH = join(DATASET_PATH, f'{XES_NAME}_categorical_attributes.txt')


def create_directories(confirm=False):
    if confirm:
        folders_to_create = [
            INPUT_PATH,
            XES_PATH,
            CSV_PATH,
            G_PATH,
            OUTPUT_PATH,
            DATASET_PATH,
            DATASET_PROCESSED_PATH,
            DATASET_G_PATH,
            NET_PATH,
            NET_RESULTS_PATH,
            TMP_STATE_PATH,
        ]
        for to_create in folders_to_create:
            if exists(to_create):
                continue
            mkdir(to_create)


def load():
    parser = ArgumentParser()
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--per', default=80, type=int)
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_neurons', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--grid_search', default=True, type=bool)

    args = parser.parse_args()
    return args
