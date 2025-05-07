# These are the main classes and methods.
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from scipy.stats import truncnorm
from collections import deque
import operator
import math
import random
import os
from torch.nn import functional
from torch.distributions import Categorical
import copy
import itertools
from torch.distributions.log_normal import LogNormal
from torchvision.datasets import ImageFolder
from torchvision import transforms
import json
import torch.nn as nn
from conf import *
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_np_seed(seed):
    np.random.seed(seed)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data
    
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def to_numpy_(tensor):
    return tensor.cpu().numpy()

def kl_loss(human_probs, machine_probs, h_labels, labels):
    labels_onehot = F.one_hot(h_labels, num_classes=3).float()
    mask_ = ~h_labels.eq(labels)
    kl_div = F.kl_div(human_probs.log(), labels_onehot, reduction='none')
    return -(kl_div[mask_].mean())* lambda_

data_transform_ai = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
    }


class HumanAgent:

    def __init__(self, labels, h_labels):
        self.n_cls = 3
        self.tr_true = labels
        self.tr_pred = h_labels
    
    def int_random_indices(self):
        category_indices = {i.item(): (self.tr_true == i).nonzero(as_tuple=True)[0] for i in self.tr_true.unique()}
        min_len = min(len(lst) for lst in category_indices.values())
        random_indices = {i.item(): np.random.permutation(min_len) for i in self.tr_true.unique()}
        self.category_indices = category_indices
        self.random_indices = random_indices

    def make_human_limit(self, nums_limit):
        self.nums_limit = nums_limit
        self.limit_ex_pred = []
        
        for category in self.category_indices.keys():
            samples = self.category_indices[category]
            random_ids = self.random_indices[category][:nums_limit]
            self.limit_ex_pred.extend(samples[random_ids].tolist())
        self.tr_pred_limit = self.tr_pred[self.limit_ex_pred]
        posterior_matr = 1. * confusion_matrix(self.tr_true[self.limit_ex_pred].cpu().numpy(), self.tr_pred[self.limit_ex_pred].cpu().numpy(), labels=np.arange(self.n_cls)) +0.01
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
        self.posterior_matr = torch.tensor(posterior_matr, device=device)
        # all
        posterior_matr = 1. * confusion_matrix(self.tr_true.cpu().numpy(), self.tr_pred.cpu().numpy(), labels=np.arange(self.n_cls)) +0.01
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
        self.posterior_matr_all = torch.tensor(posterior_matr, device=device)
    
    def make_alh_limit(self, category_indices, nums_limit):
        self.limit_ex_pred = []
        for category in category_indices:
            self.limit_ex_pred.extend(category_indices[category][:nums_limit])
        self.tr_pred_limit = self.tr_pred[self.limit_ex_pred]
        posterior_matr = 1. * confusion_matrix(self.tr_true[self.limit_ex_pred].cpu().numpy(), self.tr_pred[self.limit_ex_pred].cpu().numpy(), labels=np.arange(self.n_cls)) +0.01
        
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
        self.posterior_matr = torch.tensor(posterior_matr, device=device)

    def make_alh_limit_plusone(self, category_indices, nums_limit):
        limit_dict = {}
        n = len(self.limit_ex_pred)
        chunk_size = n // 3
        parts = [self.limit_ex_pred[i * chunk_size:(i + 1) * chunk_size] for i in range(3)]
        
        self.limit_ex_pred = []
        for category in category_indices:
            samples = self.category_indices[category]
            random_ids = self.random_indices[category]
            self.limit_ex_pred.extend(parts[category])
            increase_id = category_indices[category][win_s: win_s+win_l]
            np.random.shuffle((increase_id))
            self.limit_ex_pred.extend(increase_id[:(nums_limit-chunk_size)])
        self.tr_pred_limit = self.tr_pred[self.limit_ex_pred]
        posterior_matr = 1. * confusion_matrix(self.tr_true[self.limit_ex_pred].cpu().numpy(), self.tr_pred[self.limit_ex_pred].cpu().numpy(), labels=np.arange(self.n_cls)) +0.01
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
        self.posterior_matr = torch.tensor(posterior_matr, device=device)
    
    def revise_cm(self,matrix, row_id, acc):
        matrix = matrix
        row_sums = matrix.sum(axis=1)

        new_matrix = matrix.copy()
        new_diag_value = acc * row_sums[row_id]
        non_diag_sum = row_sums[row_id] - matrix[row_id, row_id]
        if non_diag_sum > 0:
            scale_factor = (row_sums[row_id] - new_diag_value) / non_diag_sum
            new_matrix[row_id, :] *= scale_factor
            new_matrix[row_id, row_id] = new_diag_value
        else:
            new_matrix[row_id, row_id] = new_diag_value
        return new_matrix
    def alh_choose_sample(self, num_al, new_len):
        update_sampele_dict = {}
        n = len(self.limit_ex_pred)
        chunk_size = n // 3
        new_num_al = (new_len - chunk_size) * num_al
        for category in self.category_indices.keys():
            samples = self.category_indices[category]
            random_ids = [i for i in self.random_indices[category]]
            np.random.shuffle((random_ids))
            update_sampele_dict[category] = [num for num in samples[random_ids].tolist() if num not in self.limit_ex_pred][:new_num_al]
            
        return update_sampele_dict

    def pm_choose_sample(self, num_al):
        update_sampele_dict = {}
        new_num_al = num_al
        for category in self.category_indices.keys():
            samples = self.category_indices[category]
            random_ids = [i for i in self.random_indices[category]]
            np.random.shuffle((random_ids))
            update_sampele_dict[category] = [num for num in samples[random_ids].tolist() if num not in self.limit_ex_pred][:new_num_al]
            
        return update_sampele_dict

    def create_new_hlabel(self, category_indices):
        self.limit_ex_pred = []
        for category in category_indices:
            self.limit_ex_pred.extend(category_indices[category][:self.k])
        self.tr_pred = self.all_tr_pred[self.limit_ex_pred]

    def create_new_conf(self, human_preds, labels):
        human_preds = np.concatenate((human_preds.cpu().numpy(), self.tr_pred[self.limit_ex_pred].cpu().numpy()))
        labels = np.concatenate((labels.cpu().numpy(), self.tr_true[self.limit_ex_pred].cpu().numpy()))
        posterior_matr = 1. * confusion_matrix(labels, human_preds, labels=np.arange(self.n_cls)) +0.01#+ prior_matr
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
        self.posterior_matr = torch.tensor(posterior_matr, device=device)

    def combine_hm(self, machine_probs, h_preds):
        combined_matrix = self.posterior_matr[h_preds]
        combined_matrix_all = self.posterior_matr_all[h_preds]
        y_comb = machine_probs * combined_matrix
        y_comb_norm = y_comb / torch.sum(y_comb, dim=1, keepdim=True)
        return combined_matrix / combined_matrix.sum(dim=1, keepdim=True), y_comb_norm, combined_matrix_all / combined_matrix_all.sum(dim=1, keepdim=True)

    def combine_hm_gamma(self, machine_probs, norm_hp_outputs, h_preds, hm_gamma):
        h_preds_onehot = F.one_hot(h_preds, num_classes=3).float()
        hm_gamma_expanded = hm_gamma.unsqueeze(2)
        combined_matrix = h_preds_onehot * hm_gamma_expanded[:, 0, :] + norm_hp_outputs * hm_gamma_expanded[:, 1, :]
        y_comb = machine_probs * combined_matrix
        y_comb_norm = y_comb / torch.sum(y_comb, dim=1, keepdim=True)
        return y_comb_norm


    def combine_hm_eva(self, machine_probs, norm_hp_outputs):
        y_comb = machine_probs * norm_hp_outputs
        y_comb_norm = y_comb / torch.sum(y_comb, dim=1, keepdim=True)
        return y_comb_norm

class TripletImageFolder(ImageFolder):
    def __init__(self, root, human_labels_file=None, transform=None):
        super(TripletImageFolder, self).__init__(root, transform=transform)
        
        self.human_labels = self._load_human_labels_from_json(human_labels_file)
        

    def _load_human_labels_from_json(self, json_file):
        df = load_json(json_file)
        filename_to_label = {item['name']: item['h_label'] for item in df}
        
        human_labels = []
        for sample_path, _ in self.samples:
            filename = 'train/'+os.path.basename(sample_path)
            human_labels.append(filename_to_label[filename])
        return human_labels

    def __getitem__(self, index):
        image, true_label = super(TripletImageFolder, self).__getitem__(index)
        human_label = self.human_labels[index]
        
        return image, true_label, human_label



class BaseCalibrator:
    """ Abstract calibrator class
    """
    def __init__(self):
        self.n_classes = None

    def fit(self, logits, y):
        raise NotImplementedError

    def calibrate(self, probs):
        raise NotImplementedError

class TSCalibratorMAP(BaseCalibrator):
    """ MAP Temperature Scaling
    """

    def __init__(self, temperature=1., prior_mu=0.5, prior_sigma=0.5):
        super().__init__()
        self.temperature = temperature
        self.loss_trace = None

        self.prior_mu = torch.tensor(prior_mu)
        self.prior_sigma = torch.tensor(prior_sigma)

    def fit(self, model_logits, y):
        """ Fits temperature scaling using hard labels.
        """
        # Pre-processing
        _model_logits = torch.from_numpy(model_logits)
        _y = torch.from_numpy(y)
        _temperature = torch.tensor(self.temperature, requires_grad=True)

        prior = LogNormal(self.prior_mu, self.prior_sigma)
        # Optimization parameters
        nll = torch.nn.CrossEntropyLoss()  # Supervised hard-label loss
        num_steps = 17500
        learning_rate = 0.05
        grad_tol = 1e-3  # Gradient tolerance for early stopping
        min_temp, max_temp = 1e-2, 1e4  # Upper / lower bounds on temperature

        optimizer = torch.optim.Adam([_temperature], lr=learning_rate)

        loss_trace = []  # Track loss over iterations
        step = 0
        converged = False
        while not converged:

            optimizer.zero_grad()
            loss = nll(_model_logits / _temperature, _y)
            
            loss += -1 * prior.log_prob(_temperature)  # This step adds the prior
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())
            with torch.no_grad():
                _temperature.clamp_(min=min_temp, max=max_temp)
            step += 1
            if step > num_steps:
                warnings.warn('Maximum number of steps reached -- may not have converged (TS)')
            converged = (step > num_steps) or (np.abs(_temperature.grad) < grad_tol)

        self.loss_trace = loss_trace
        self.temperature = _temperature.item()

    def calibrate(self, probs):
        calibrated_probs = probs ** (1. / self.temperature)  # Temper
        calibrated_probs /= torch.sum(calibrated_probs, axis=1, keepdims=True) # Normalize
        return calibrated_probs

class Evaluator(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        emb_dim = 512
        self.num_dis = 3
        self.human_embedding = nn.Embedding(self.num_dis, emb_dim)
        
        self.dis_fc1 = nn.Linear(emb_dim, self.num_dis, bias=True)
        self.dis_fc2 = nn.Linear(emb_dim, 1, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.Sigmoid()

    def forward(self, human_input, feature_input=None, combined_matrix = None):
        human_emb = self.human_embedding(human_input)
        if feature_mode != True:
            hp_outputs = self.dis_fc1(self.dropout(human_emb))
            norm_h_outputs = torch.softmax(hp_outputs, dim=-1)
        else:
            h_outputs = self.dis_fc1(self.dropout(human_emb+feature_input))
            norm_h_outputs = torch.softmax(h_outputs, dim=-1)

        return norm_h_outputs



# [ECCV 2024] Learning to Complement or Defer to Multiple Users (LECODU) 
class CollaborationNet(nn.Module):
    def __init__(self, channels=2, hidden_size=512, dim=3):
        super(CollaborationNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels * dim, hidden_size)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PredictNet(nn.Module):
    def __init__(self, dim):
        super(PredictNet, self).__init__()
        self.fc1 = nn.Linear(dim, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x

def make_bin_labels(h_preds, labels):
    return torch.eq(h_preds, labels).long()

def make_multi_labels(h_preds, labels, num_classes=3):
    output = torch.clone(labels)
    redistribution_count = torch.zeros(num_classes, device=h_preds.device)

    for i in range(h_preds.size(0)):
        if h_preds[i] == 0:
            current_label = labels[i]
            choices = torch.tensor(
                [x for x in range(num_classes) if x != current_label], device=h_preds.device
            )

            min_count_label = torch.argmin(redistribution_count[choices])
            chosen_label = choices[min_count_label]

            output[i] = chosen_label
            redistribution_count[chosen_label] += 1

    return output

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        logits_str, true_label, predicted_label, feat_str = self.data.iloc[idx]

        logits = np.array([float(x) for x in logits_str.split(',')])
        feat = np.array([float(x) for x in feat_str.split(',')])

        true_label = int(true_label)
        predicted_label = int(predicted_label)

        logits = torch.tensor(logits, dtype=torch.float32)
        true_label = torch.tensor(true_label, dtype=torch.long)
        predicted_label = torch.tensor(predicted_label, dtype=torch.long)
        feat = torch.tensor(feat, dtype=torch.float32)

        if self.transform:
            logits = self.transform(logits)

        return logits, true_label, predicted_label, feat

def compute_two_acc(human_pred, ai_pred, combined_pred, true_pred):
    human_pred = np.array(human_pred)
    ai_pred = np.array(ai_pred)
    combined_pred = np.array(combined_pred)
    true_pred = np.array(true_pred)
    complementary_condition = (human_pred != true_pred) #& (ai_pred == true_pred)
    complementary_accuracy = (complementary_condition & (combined_pred == true_pred))
    len_data = len(human_pred)
    complementary_accuracy_rate = complementary_accuracy.sum().item() / len_data #complementary_condition.sum().item() if complementary_condition.sum() > 0 else 0
    correction_condition = (human_pred == true_pred) #& (ai_pred != true_pred)
    correction_accuracy = (correction_condition & (combined_pred == true_pred))

    correction_accuracy_rate = correction_accuracy.sum().item() / len_data #correction_condition.sum().item() if correction_condition.sum() > 0 else 0
    return complementary_accuracy_rate, correction_accuracy_rate #complementary_accuracy.sum(), complementary_condition.sum(),  correction_accuracy.sum(), correction_condition.sum()
