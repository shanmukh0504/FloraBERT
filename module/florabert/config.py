from pathlib import Path
import yaml
import random

import numpy as np
import torch


root = Path(__file__).parent.parent.parent
data = root / 'data'
models = root / 'models'
notebooks = root / 'notebooks'
scripts = root / 'scripts'
output = root / 'output'
docs = root / 'docs'

# Data specific paths
data_raw = data / 'raw'
data_processed = data / 'processed'
data_final = data / 'final'

# Location of tools
libs = root / 'libs'
samtools = libs / 'samtools'
bedtools = libs / 'bedtools'
dnabert = root / 'DNABERT'

# Locations of specific files
bpe_tokenizer = data_final / 'tokenizer' / 'maize_bpe_full.tokenizer.json'

# Loading settings
settings = yaml.full_load((root / 'config.yaml').open('r'))

# Setting random seeds across the whole project
random_seed = settings['random_seed']
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def reload_settings():
    global settings
    settings = yaml.full_load((root / 'config.yaml').open('r'))


# Tissue labels in data/final/nam_data/merged_seq_genex.csv and
# data/final/transformer/genex/nam/*.tsv.
tissues = [
    'tassel',
    'base',
    'anther',
    'middle',
    'ear',
    'shoot',
    'tip',
    'root',
]

# OLD
# tissues = [
#     'anther',
#     'ear',
#     'embryo',
#     'endosperm',
#     'leaf',
#     'leafbase',
#     'leaftip',
#     'root',
#     'shoot',
#     'tassel'
# ]
