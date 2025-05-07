# These are all the parameters. 
import argparse

# parameters (training)
parser = argparse.ArgumentParser()
parser.add_argument('-data', '--train_dataset', default='mz10', help='choose the training dataset')
args = parser.parse_args()
# dataset for training
train_dataset = args.train_dataset
dev_dataset = train_dataset
test_dataset = train_dataset

# Active collection
random_mode = False
feature_mode = True

# Limited number of expert predictions
limit_human_pred = [2, 3, 4, 5, 6, 7, 8, 3000]

# Number of repeated experiments: num_repeats*num_human_repeats
num_repeats = 10
num_human_repeats = 5

# median-window
win_l = 5
win_s = 50

# sample size
num_al = 100


# check validity
ds_range = ['dxy','mz10', 'mz4', 'MDD', 'GMD']

assert train_dataset in ds_range
assert test_dataset in ds_range

device_num = 0

# train/test data path
train_path = []
if train_dataset != 'all':
    train_path.append('../../data/MZ-10/train_set.json'.format(train_dataset))
else:
    for ds in ds_range[1:]:
        train_path.append('../../data/MZ-10/train_set.json'.format(ds))

dev_path = []
if dev_dataset != 'all':
    dev_path.append('../../data/MZ-10/dev_set.json'.format(dev_dataset))
else:
    for ds in ds_range[1:]:
        dev_path.append('../../data/MZ-10/dev_set.json'.format(ds))

test_path = []
if test_dataset != 'all':
    test_path.append('../../data/MZ-10/test_set.json'.format(test_dataset))
else:
    for ds in ds_range[1:]:
        test_path.append('../../data/MZ-10/test_set.json'.format(ds))
dev_path = test_path
best_pt_path = 'saved/MZ-10/best_pt_exe_model.pt'.format(train_dataset)
last_pt_path = 'saved/MZ-10/last_pt_exe_model.pt'.format(train_dataset)
best_ai_path = 'saved/MZ-10/ai_'.format(train_dataset)

# global settings
suffix = {'0': '-Negative', '1': '-Positive', '2': '-Negative'}
min_sx_freq = None
max_voc_size = None
keep_unk = True
digits = 4

# model hyperparameter setting

num_attrs = 5
num_executor = 2


# group 3: transformer encoder
enc_emb_dim = 512 
enc_dim_feedforward = 2 * enc_emb_dim

enc_num_heads = 16 
enc_num_layers = {'dxy': 2, 'mz10': 2, 'mz4': 2, 'MDD': 10, 'GMD': 1}.get(train_dataset)
enc_dropout = {'dxy': 0.0, 'mz10': 0.0, 'mz4': 0.0, 'MDD': 10, 'GMD': 0.45}.get(train_dataset)

# training
num_workers = 0

cos_lr_max = 10
train_bsz = 64
test_bsz = 128

alpha = 0.2
verbose = True


num_dx_s = {'dxy': 4, 'mz10': 8, 'mz4': 3, 'MDD': 10, 'GMD': 10}.get(train_dataset)
num_dx = {'dxy': 5, 'mz10': 10, 'mz4': 4, 'MDD': 12, 'GMD': 12}.get(train_dataset)
