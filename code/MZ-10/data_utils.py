# These are the main classes and methods.
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import os
from torch.nn import functional
from torch.distributions import Categorical
from torch.distributions.log_normal import LogNormal
import torch.nn.functional as F
import json
from conf import *
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset, DataLoader
import pandas as pd
import torch.nn as nn
device = torch.device('cuda:{}'.format(device_num) if torch.cuda.is_available() else 'cpu')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SymptomVocab:

    def __init__(self, samples: list = None, add_special_sxs: bool = False,
                 min_sx_freq: int = None, max_voc_size: int = None, prior_feat: list = None):

        # sx is short for symptom
        self.sx2idx = {}  # map from symptom to symptom id
        self.idx2sx = {}  # map from symptom id to symptom
        self.sx2count = {}  # map from symptom to symptom count
        self.num_sxs = 0  # number of symptoms
        self.prior_sx_attr = {} # key symptoms with True
        self.prior_sx_attr_2 = {} # key symptoms with False
        self.no_key_sx = []


        # symptom attrs
        self.SX_ATTR_PAD_IDX = 0  # symptom attribute id for PAD
        self.SX_ATTR_POS_IDX = 1  # symptom attribute id for YES
        self.SX_ATTR_NEG_IDX = 2  # symptom attribute id for NO
        self.SX_ATTR_NS_IDX = 3  # symptom attribute id for NOT SURE
        self.SX_ATTR_NM_IDX = 4  # symptom attribute id for NOT MENTIONED

        # symptom askers
        self.SX_EXE_PAD_IDX = 0  # PAD
        self.SX_EXE_AI_IDX = 1  # AI
        self.SX_EXE_DOC_IDX = 2  # Human

        self.SX_ATTR_MAP = {  # map from symptom attribute to symptom attribute id
            '0': self.SX_ATTR_NEG_IDX,
            '1': self.SX_ATTR_POS_IDX,
            '2': self.SX_ATTR_NS_IDX,
        }

        self.SX_ATTR_MAP_INV = {
            self.SX_ATTR_NEG_IDX: '0',
            self.SX_ATTR_POS_IDX: '1',
            self.SX_ATTR_NS_IDX: '2',
        }

        # special symptoms
        self.num_special = 0  # number of special symptoms
        self.special_sxs = []

        # vocabulary
        self.min_sx_freq = min_sx_freq  # minimal symptom frequency
        self.max_voc_size = max_voc_size  # maximal symptom size

        # add special symptoms
        if add_special_sxs:  # special symptoms
            self.SX_PAD = '[PAD]'
            self.SX_START = '[START]'
            self.SX_END = '[END]'
            self.SX_UNK = '[UNKNOWN]'
            self.SX_CLS = '[CLS]'
            self.SX_MASK = '[MASK]'
            self.special_sxs.extend([self.SX_PAD, self.SX_START, self.SX_END, self.SX_UNK, self.SX_CLS, self.SX_MASK])
            self.sx2idx = {sx: idx for idx, sx in enumerate(self.special_sxs)}
            self.idx2sx = {idx: sx for idx, sx in enumerate(self.special_sxs)}
            self.num_special = len(self.special_sxs)
            self.num_sxs += self.num_special


        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('symptom vocabulary constructed using {} split and {} samples '
                  '({} symptoms with {} special symptoms)'.
                  format(len(samples), num_samples, self.num_sxs - self.num_special, self.num_special))

        # trim vocabulary
        self.trim_voc()

        assert self.num_sxs == len(self.sx2idx) == len(self.idx2sx)

    def add_symptom(self, sx: str) -> None:
        if sx not in self.sx2idx:
            self.sx2idx[sx] = self.num_sxs
            self.sx2count[sx] = 1
            self.idx2sx[self.num_sxs] = sx
            self.num_sxs += 1
        else:
            self.sx2count[sx] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            for sx in sample['exp_sxs']:
                self.add_symptom(sx)
            for sx in sample['imp_sxs']:
                self.add_symptom(sx)
        return len(samples)

    def trim_voc(self):
        sxs = [sx for sx in sorted(self.sx2count, key=self.sx2count.get, reverse=True)]
        if self.min_sx_freq is not None:
            sxs = [sx for sx in sxs if self.sx2count.get(sx) >= self.min_sx_freq]
        if self.max_voc_size is not None:
            sxs = sxs[: self.max_voc_size]
        sxs = self.special_sxs + sxs
        self.sx2idx = {sx: idx for idx, sx in enumerate(sxs)}
        self.idx2sx = {idx: sx for idx, sx in enumerate(sxs)}
        self.sx2count = {sx: self.sx2count.get(sx) for sx in sxs if sx in self.sx2count}
        self.num_sxs = len(self.sx2idx)
        print('trimmed to {} symptoms with {} special symptoms'.
              format(self.num_sxs - self.num_special, self.num_special))

    def encode(self, sxs: dict, keep_unk=True, add_start=False, add_end=False):
        sx_ids, attr_ids = [], []
        if add_start:
            sx_ids.append(self.start_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        for sx, attr in sxs.items():
            if sx in self.sx2idx:
                sx_ids.append(self.sx2idx.get(sx))
                attr_ids.append(self.SX_ATTR_MAP.get(attr))
            else:
                if keep_unk:
                    sx_ids.append(self.unk_idx)
                    attr_ids.append(self.SX_ATTR_MAP.get(attr))
        if add_end:
            sx_ids.append(self.end_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        return sx_ids, attr_ids

    def decoder(self, sx_ids, attr_ids):
        sx_attr = {}
        for sx_id, attr_id in zip(sx_ids, attr_ids):
            if attr_id not in [self.SX_ATTR_PAD_IDX, self.SX_ATTR_NM_IDX]:
                sx_attr.update({self.idx2sx.get(sx_id): self.SX_ATTR_MAP_INV.get(attr_id)})
        return sx_attr

    def __len__(self) -> int:
        return self.num_sxs

    @property
    def pad_idx(self) -> int:
        return self.sx2idx.get(self.SX_PAD)

    @property
    def start_idx(self) -> int:
        return self.sx2idx.get(self.SX_START)

    @property
    def end_idx(self) -> int:
        return self.sx2idx.get(self.SX_END)

    @property
    def unk_idx(self) -> int:
        return self.sx2idx.get(self.SX_UNK)

    @property
    def cls_idx(self) -> int:
        return self.sx2idx.get(self.SX_CLS)

    @property
    def mask_idx(self) -> int:
        return self.sx2idx.get(self.SX_MASK)

    @property
    def pad_sx(self) -> str:
        return self.SX_PAD

    @property
    def start_sx(self) -> str:
        return self.SX_START

    @property
    def end_sx(self) -> str:
        return self.SX_END

    @property
    def unk_sx(self) -> str:
        return self.SX_UNK

    @property
    def cls_sx(self) -> str:
        return self.SX_CLS

    @property
    def mask_sx(self) -> str:
        return self.SX_MASK


class DiseaseVocab:

    def __init__(self, samples: list = None):

        # dis is short for disease
        self.dis2idx = {}
        self.idx2dis = {}
        self.dis2count = {}
        self.num_dis = 0

        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('disease vocabulary constructed using {} split and {} samples\nnum of unique diseases: {}'.
                  format(len(samples), num_samples, self.num_dis))

    def add_disease(self, dis: str) -> None:
        if dis not in self.dis2idx:
            self.dis2idx[dis] = self.num_dis
            self.dis2count[dis] = 1
            self.idx2dis[self.num_dis] = dis
            self.num_dis += 1
        else:
            self.dis2count[dis] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            self.add_disease(sample['label'])
        return len(samples)

    def __len__(self) -> int:
        return self.num_dis

    def encode(self, dis):
        return self.dis2idx.get(dis)


class SymptomDataset(Dataset):

    def __init__(self, samples, sv: SymptomVocab, dv: DiseaseVocab, keep_unk: bool,
                 add_src_start: bool = False, add_tgt_start: bool = False, add_tgt_end: bool = False, train_mode: bool = False):
        self.samples = samples
        self.sv = sv
        self.dv = dv
        self.keep_unk = keep_unk
        self.size = len(self.sv)
        self.add_src_start = add_src_start
        self.add_tgt_start = add_tgt_start
        self.add_tgt_end = add_tgt_end
        self.train_mode = train_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        exp_sx_ids, exp_attr_ids = self.sv.encode(
            sample['exp_sxs'], keep_unk=self.keep_unk, add_start=self.add_src_start)
        imp_sx_ids, imp_attr_ids = self.sv.encode(
            sample['imp_sxs'], keep_unk=self.keep_unk, add_start=self.add_tgt_start, add_end=self.add_tgt_end)
        exp_exe_ids = [0 for i in range(len(exp_sx_ids))]
        imp_exe_ids = [0]
        imp_exe_ids.extend([0 for i in range(len(imp_sx_ids) - 2)])
        imp_exe_ids.append(0)
    
        exp_sx_ids, exp_attr_ids, exp_exe_ids, imp_sx_ids, imp_attr_ids, imp_exe_ids, label = to_tensor_vla(
            exp_sx_ids, exp_attr_ids, exp_exe_ids, imp_sx_ids, imp_attr_ids, imp_exe_ids, self.dv.encode(sample['label']),  dtype=torch.long)
        item = {
            'exp_sx_ids': exp_sx_ids,
            'exp_attr_ids': exp_attr_ids,
            'exp_exe_ids': exp_exe_ids,
            'imp_sx_ids': imp_sx_ids,
            'imp_attr_ids': imp_attr_ids,
            'imp_exe_ids': imp_exe_ids,
            'label': label,
            'id': index
        }
        return item


# language model
def lm_collater(samples):
    sx_ids = pad_sequence(
        [torch.cat([sample['exp_sx_ids'], sample['imp_sx_ids']]) for sample in samples], padding_value=0)
    attr_ids = pad_sequence(
        [torch.cat([sample['exp_attr_ids'], sample['imp_attr_ids']]) for sample in samples], padding_value=0)
    labels = torch.stack([sample['label'] for sample in samples])
    ids = [sample['id'] for sample in samples]
    items = {
        'sx_ids': sx_ids,
        'attr_ids': attr_ids,
        'labels': labels,
        'ids': ids
    }
    return items




def recursive_sum(item):
    if isinstance(item, list):
        try:
            return sum(item)
        except TypeError:
            return recursive_sum(sum(item, []))
    else:
        return item


def average(numerator, denominator):
    return 0 if recursive_sum(denominator) == 0 else recursive_sum(numerator) / recursive_sum(denominator)


def to_numpy(tensors):
    arrays = {}
    for key, tensor in tensors.items():
        arrays[key] = tensor.cpu().numpy()
    return arrays


def to_numpy_(tensor):
    return tensor.cpu().numpy()


def to_list(tensor):
    return to_numpy_(tensor).tolist()


def to_numpy_vla(*tensors):
    arrays = []
    for tensor in tensors:
        arrays.append(to_numpy_(tensor))
    return arrays


def to_tensor_(array, dtype=None):
    if dtype is None:
        return torch.tensor(array, device=device)
    else:
        return torch.tensor(array, dtype=dtype, device=device)


def to_tensor_vla(*arrays, dtype=None):
    tensors = []
    for array in arrays:
        tensors.append(to_tensor_(array, dtype))
    return tensors


def extract_features(sx_ids, attr_ids, sv: SymptomVocab):
    sx_feats, attr_feats = [], []
    exe_feats = []
    for idx in range(len(sx_ids)):
        sx_feat, attr_feat, exe_feat = [sv.start_idx], [sv.SX_ATTR_PAD_IDX], [sv.SX_ATTR_PAD_IDX]
        for sx_id, attr_id in zip(sx_ids[idx], attr_ids[idx]):
            if sx_id == sv.end_idx:
                break
            if attr_id not in [sv.SX_ATTR_PAD_IDX, sv.SX_ATTR_NM_IDX]:
                sx_feat.append(sx_id)
                attr_feat.append(attr_id)
                
        sx_feats.append(to_tensor_(sx_feat))
        attr_feats.append(to_tensor_(attr_feat))
        
    return sx_feats, attr_feats



def make_features_xfmr(sv: SymptomVocab, batch, si_sx_ids=None, si_attr_ids=None,  merge_act: bool = False,
                       merge_si: bool = False):
    # convert to numpy
    assert merge_act or merge_si
    sx_feats, attr_feats = [], []
    exe_feats = []
    
    si_sx_ids, si_attr_ids = to_numpy_vla(si_sx_ids, si_attr_ids)
    si_sx_feats, si_attr_feats = extract_features(si_sx_ids, si_attr_ids, sv)
    sx_feats += si_sx_feats
    attr_feats += si_attr_feats
        
    sx_feats = pad_sequence(sx_feats, padding_value=sv.pad_idx).long()
    attr_feats = pad_sequence(attr_feats, padding_value=sv.SX_ATTR_PAD_IDX).long()
    
    return sx_feats, attr_feats

class HumanAgent:

    def __init__(self, notrain_ds_loader, dev_ds_loader, test_ds_loader, human_acc):
        self.n_cls = num_dx
        diag_vector = generate_uniform_vector(num_dx, human_acc)
        other_p = generate_probability_matrix(num_dx)
        for i in notrain_ds_loader:
            tr_true = i['labels']
            tr_pred = simulate_human_pred(tr_true, num_dx_s, num_dx, diag_vector, other_p)
        for i in test_ds_loader:
            te_true = i['labels']
            te_pred = simulate_human_pred(te_true, num_dx_s, num_dx, diag_vector, other_p)

        self.tr_true = tr_true
        self.tr_pred = tr_pred
        self.te_pred = te_pred

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

    def make_alh_limit_plusone(self, category_indices, nums_limit, the = 0):
        limit_dict = {}
        n = len(self.limit_ex_pred)
        chunk_size = n // self.n_cls
        parts = [self.limit_ex_pred[i * chunk_size:(i + 1) * chunk_size] for i in range(self.n_cls)]

        self.limit_ex_pred = []
        for category in category_indices:
            self.limit_ex_pred.extend(parts[category])
            increase_id = category_indices[category][win_s: win_s+win_l]
            np.random.shuffle((increase_id))
            self.limit_ex_pred.extend(increase_id[:(nums_limit-chunk_size)])
        self.tr_pred_limit = self.tr_pred[self.limit_ex_pred]
        posterior_matr = 1. * confusion_matrix(self.tr_true[self.limit_ex_pred].cpu().numpy(), self.tr_pred[self.limit_ex_pred].cpu().numpy(), labels=np.arange(self.n_cls)) +0.01
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr) / (np.sum(posterior_matr, axis=0, keepdims=True))
        self.posterior_matr = torch.tensor(posterior_matr, device=device)
    
    def alh_choose_sample(self, num_al, new_len):
        update_sampele_dict = {}
        n = len(self.limit_ex_pred)
        chunk_size = n // self.n_cls
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
        h_preds_onehot = functional.one_hot(h_preds, num_classes=self.n_cls).float()
        hm_gamma_expanded = hm_gamma.unsqueeze(2)
        combined_matrix = h_preds_onehot * hm_gamma_expanded[:, 0, :] + norm_hp_outputs * hm_gamma_expanded[:, 1, :]
        y_comb = machine_probs * combined_matrix
        y_comb_norm = y_comb / torch.sum(y_comb, dim=1, keepdim=True)
        return y_comb_norm

    def get_h_gamma(self, norm_hp_outputs, h_preds, hm_gamma):
        h_preds_onehot = functional.one_hot(h_preds, num_classes=self.n_cls).float()
        hm_gamma_expanded = hm_gamma.unsqueeze(2)
        combined_matrix = h_preds_onehot * hm_gamma_expanded[:, 0, :] + norm_hp_outputs * hm_gamma_expanded[:, 1, :]
        
        return combined_matrix

    def combine_hm_eva(self, machine_probs, norm_hp_outputs):
        y_comb = machine_probs * norm_hp_outputs
        y_comb_norm = y_comb / torch.sum(y_comb, dim=1, keepdim=True)
        return y_comb_norm

    def print_acc_per_class(self, true, pred):
        conf_matrix = confusion_matrix(np.array(true), np.array(pred), labels=np.arange(self.n_cls))
        error_counts = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
        error_per_class = error_counts / (conf_matrix.sum(axis=1) + 1e-5 )
        return error_per_class, np.sum(conf_matrix, axis=1)

    def print_error_per_class(self, true, pred, h_pred):
        conf_matrix = confusion_matrix(to_numpy_(true), to_numpy_(pred), labels=np.arange(self.n_cls))
        conf_matrix_h = confusion_matrix(to_numpy_(true), to_numpy_(h_pred), labels=np.arange(self.n_cls))
        error_counts = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
        error_per_class = error_counts / (conf_matrix.sum(axis=1) + 1e-5 )
        error_counts_h = np.sum(conf_matrix_h, axis=1) - np.diag(conf_matrix_h)
        error_per_class_h = error_counts_h / (conf_matrix_h.sum(axis=1) + 1e-5 )
        return error_per_class_h, error_per_class
    
    def init_sx_ids(self, bsz):
        return 0

def simulate_human_pred(true_labels, id_s, id_e, diag_vector, other_p):
    predicted_labels = torch.randint(0, num_dx, (len(true_labels),), device='cuda:0')
    for i in range(id_e):
        indices = (true_labels == i).nonzero(as_tuple=True)[0]
        correct_predictions = round(len(indices) * diag_vector[i])
        incorrect_predictions = len(indices) - correct_predictions
        correct_indices = np.random.choice(indices.cpu().numpy(), correct_predictions, replace=False)
        incorrect_indices = list(set(indices.cpu().numpy()) - set(correct_indices))
        predicted_labels[correct_indices] = i
        other_nums = allocate_counts(other_p[i], incorrect_predictions)  
        for other_i in range(num_dx):
            other_i_indices = np.random.choice(incorrect_indices, other_nums[other_i], replace=False)
            predicted_labels[other_i_indices] = other_i
            incorrect_indices = list(set(incorrect_indices) - set(other_i_indices))
    return predicted_labels

def generate_probability_matrix(n):
    matrix = np.zeros((n, n))

    for i in range(n):
        random_values = np.random.rand(n-1)
        
        normalized_values = random_values / random_values.sum()
        
        matrix[i, :i] = normalized_values[:i]
        matrix[i, i+1:] = normalized_values[i:]
        
    return matrix

def allocate_counts(probabilities, total_count):
    expected_counts = np.array(probabilities) * total_count
    
    counts = np.round(expected_counts).astype(int)
    
    difference = total_count - np.sum(counts)
    
    while difference != 0:
        if difference > 0:
            min_index = np.argmin(counts)
            counts[min_index] += 1
            difference -= 1
        else:
            max_index = np.argmax(counts)
            counts[max_index] -= 1
            difference += 1
    
    return counts

def acc_from_conf(true_labels, normalized_confusion_matrix):
    predicted_labels = torch.randint(0, num_dx, (len(true_labels),), device='cuda:0')
    
    num_classes = normalized_confusion_matrix.shape[0]
    for i in range(num_classes):
        indices = np.where(true_labels == i)[0]
        pred_i_labels = np.random.choice(
            np.arange(num_classes), size=len(indices), p=normalized_confusion_matrix[i]
        )
        id_i = 0
        for id in indices:
            predicted_labels[id] = pred_i_labels[id_i]
            id_i += 1

    return predicted_labels


def get_dirichlet_params(acc, strength, n_cls):
    beta = 0.1
    alpha = beta * (n_cls - 1) * acc / (1. - acc)

    alpha *= strength
    beta *= strength

    alpha += 1
    beta += 1

    return alpha, beta


def generate_uniform_vector(size, mean, min_value=0.5, max_value=1.0):
    # min_value = mean * 2 - max_value
    vector = np.random.uniform(low=min_value, high=max_value, size=size)
    
    adjustment = mean - np.mean(vector)
    adjusted_vector = vector + adjustment
    
    adjusted_vector = np.clip(adjusted_vector, min_value, max_value)
    
    final_adjustment = mean - np.mean(adjusted_vector)
    final_vector = adjusted_vector + final_adjustment
    
    final_vector = np.clip(final_vector, min_value, max_value)
    
    return final_vector

def set_np_seed(seed):
    np.random.seed(seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def find_indices(a, b):
    indices = [(index, b.index(x))for index, x in enumerate(a) if x in b]
    if indices == []:
        return [], []
    else:
        i1, i2 = list(zip(*indices))
        return list(i1), list(i2)
        
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

def revise_conf(matrix):
    for i in range(matrix.shape[1]):
        val = matrix[i, i]
        if val <= 0.5:
            matrix[:, i] =  (matrix[:, i] + val) / 5
            matrix[i, i] = 0.5
            matrix[:, i] = matrix[:, i] / matrix[:, i].sum()
            print(matrix)
            assert 0
    return matrix

def revise_conf1(matrix):
    diag = torch.diag(matrix)
    mask = diag <= 0.5
    if mask.any():
        matrix[mask, mask] = 10
        matrix = matrix / matrix.sum(dim=0, keepdim=True)
    return matrix

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


class OracleCombiner:
    """ Implements the P+L combination method, fit using maximum likelihood
    """
    def __init__(self, calibration_method='temperature scaling', **kwargs):
        self.calibrator = None
        self.confusion_matrix = None  # conf[i, j] is assumed to be P(h = i | Y = j)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.eps = 1e-50

        self.use_cv = False
        self.calibration_method = calibration_method
        if self.calibration_method == 'temperature scaling':
            self.calibrator = TSCalibrator()
        elif self.calibration_method == 'dirichlet':
            # reg_norm : bool, true if regularization is used
            # reg_mu : None or float, if None regular L2 regularization is used
            # reg_lambda : 0 or float, l2 regularization term
            from dirichlet_python.dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
            self.calibrator = FullDirichletCalibrator(reg_norm=True, reg_lambda=0.0, reg_mu=None)
            self.use_cv = True
        elif self.calibration_method == 'ensemble temperature scaling':
            self.calibrator = EnsembleTSCalibrator()
        elif self.calibration_method == 'imax binning':
            mode = kwargs.pop('mode', 'sCW')
            num_bins = kwargs.pop('num_bins', 15)
            self.calibrator = IMaxCalibrator(mode=mode, num_bins=num_bins)
        elif self.calibration_method == 'none':
            self.calibrator = IdentityCalibrator()

    def calibrate(self, model_probs):
        return self.calibrator.calibrate(model_probs)

    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]

        # Estimate human confusion matrix
        # Entry [i, j]  is #(Y = i and h = j)
        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        # Swap so entry [i, j] is #(h = i and Y = j)
        conf_h = conf_h.T
        conf_h = np.clip(conf_h, self.eps, None)
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        # Normalize columns so entry [i, j] is P(h = i | Y = j)
        conf_h /= normalizer
        self.confusion_matrix = conf_h

        # Calibrate model probabilities
        if self.use_cv:
            self.fit_calibrator_cv(model_probs, y_true)
        else:
            self.fit_calibrator(model_probs, y_true)

    def fit_bayesian(self, model_probs, y_h, y_true, alpha=0.1, beta=0.1):
        """ This is the "plus one" parameterization, i.e. alpha,beta just need to be > 0
        Really corresponds to a Dirichlet(alpha+1, beta+1, beta+1, . . . ,beta+1) distribution
        """
        self.n_cls = model_probs.shape[1]

        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta

        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        conf_h += prior_matr
        # Swap so entry [i, j] is #(h = i and Y = j)
        conf_h = conf_h.T
        #conf_h = np.clip(conf_h, self.eps, None)
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        # Normalize columns so entry [i, j] is P(h = i | Y = j)
        conf_h = conf_h / normalizer
        self.confusion_matrix = conf_h

        # Calibrate model probabilities
        if self.use_cv:
            self.fit_calibrator_cv(model_probs, y_true)
        else:
            self.fit_calibrator(model_probs, y_true)

    def fit_calibrator(self, model_probs, y_true):
        clipped_model_probs = np.clip(model_probs, self.eps, 1)
        model_logits = np.log(clipped_model_probs)
        self.calibrator.fit(model_logits, y_true)

    def fit_calibrator_cv(self, model_probs, y_true):
        # Fits calibration maps that require hyperparameters, using cross-validation
        if self.calibration_method == 'dirichlet':
            reg_lambda_vals = [10., 1., 0., 5e-1, 1e-1, 1e-2, 1e-3]
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            gscv = GridSearchCV(self.calibrator, param_grid={'reg_lambda': reg_lambda_vals,
                                                             'reg_mu': [None]},
                                cv=skf, scoring='neg_log_loss', refit=True)
            gscv.fit(model_probs, y_true)
            self.calibrator = gscv.best_estimator_
        else:
            raise NotImplementedError

    def combine_proba(self, model_probs, y_h):
        """ Combines model probabilities with hard labels via the calibrate-confuse equation given the confusion matrix.

        Args:
            p_m: Array of model probabilities ; shape (n_samples, n_classes)
            y_h: List of hard labels ; shape (n_samples,)

        Returns:
            Normalized posterior probabilities P(Y | m, h). Entry [i, j] is P(Y = j | h_i, m_i)
        """
        assert model_probs.shape[0] == y_h.size, 'Size mismatch between model probs and human labels'
        assert model_probs.shape[1] == self.n_cls, 'Size mismatch between model probs and number of classes'

        n_samples = model_probs.shape[0]
        calibrated_model_probs = self.calibrate(model_probs)

        y_comb = np.empty((n_samples, self.n_cls))
        for i in range(n_samples):
            y_comb[i] = calibrated_model_probs[i] * self.confusion_matrix[y_h[i]]
            if np.allclose(y_comb[i], 0):  # Handle zero rows
                y_comb[i] = np.ones(self.n_cls) * (1./self.n_cls)

        # Don't forget to normalize :)
        assert np.all(np.isfinite(np.sum(y_comb, axis=1)))
        assert np.all(np.sum(y_comb, axis=1) > 0)
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb

    def combine(self, model_probs, y_h):
        """ Combines model probs and y_h to return hard labels
        """
        y_comb_soft = self.combine_proba(model_probs, y_h)
        return np.argmax(y_comb_soft, axis=1)

class MAPOracleCombiner(OracleCombiner):
    """ P+L combination method, fit using MAP estimates
    This is our preferred combination method.
    """
    def __init__(self, diag_acc=0.75, strength=1., mu_beta=0.5, sigma_beta=0.5, **kwargs):
        super().__init__()
        self.calibrator = None
        self.prior_params = {'mu_beta': mu_beta,
                             'sigma_beta': sigma_beta
        }
        #self.n_cls = None
        self.diag_acc = diag_acc
        self.strength = strength

    def fit(self, model_probs, y_h, y_true, model_logits=None):
        self.n_cls = model_probs.shape[1]

        # Get MAP estimate of confusion matrix
        alpha, beta = get_dirichlet_params(self.diag_acc, self.strength, self.n_cls)
        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta
        posterior_matr = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        posterior_matr += prior_matr
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr - np.ones(self.n_cls)) / (np.sum(posterior_matr, axis=0, keepdims=True) - self.n_cls)
        self.confusion_matrix = posterior_matr

        self.calibrator = TSCalibratorMAP()
        logits = np.log(np.clip(model_probs, 1e-50, 1))
        self.calibrator.fit(logits, y_true)

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

class Evaluator(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        emb_dim = 512
        self.num_dis = 10
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
    def __init__(self, channels=2, hidden_size=512, dim=10):
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
