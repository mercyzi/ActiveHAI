
# These are all the parameters. 
import torch

# Limited number of expert predictions
limit_human_pred = [2, 3, 4, 5, 6, 7, 8, 3000]

# Number of repeated experiments: num_repeats*num_human_repeats
num_repeats = 5
num_human_repeats = 10

batch_size = 64
model_lr = 3e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 30

# Active collection
random_mode = False
feature_mode = True
lambda_ = 0.05

# median-window
win_l = 5
win_s = 55

# sample size
num_al = 100
