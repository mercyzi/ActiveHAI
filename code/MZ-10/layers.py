# These are all neural networks.
import os
import torch
import torch.nn as nn
from torch.nn import functional
from torch.distributions import Categorical
import math
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import SymptomVocab, device, to_numpy_
from conf import *

weight_cus = 0

class SymptomEncoderXFMR(nn.Module):

    def __init__(self, num_sxs, num_dis, enc_dropout1):
        super().__init__()

        self.num_dis = num_dis
        self.sx_embedding = nn.Embedding(num_sxs, enc_emb_dim, padding_idx=0)
        self.attr_embedding = nn.Embedding(num_attrs, enc_emb_dim, padding_idx=0)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=enc_emb_dim,
                nhead=enc_num_heads,
                dim_feedforward=enc_num_layers,
                dropout=enc_dropout1,
                batch_first=False,
                activation='gelu'),
            num_layers=enc_num_layers)
        self.dis_fc1 = nn.Linear(enc_emb_dim, num_dis+2, bias=True)

    def forward(self, sx_ids, attr_ids, mask=None, src_key_padding_mask=None):
        inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids) 
        outputs = self.encoder(inputs, mask, src_key_padding_mask)
        return outputs

    def forward1(self, sx_ids, attr_ids, mask=None, src_key_padding_mask=None):
        inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids) 
        outputs1 = self.encoder.layers[0](inputs, mask, src_key_padding_mask)
        outputs2 = self.encoder.layers[1](outputs1, mask, src_key_padding_mask)
        return outputs1, outputs2
    
    def get_mp_features(self, sx_ids, attr_ids, pad_idx):
        src_key_padding_mask = sx_ids.eq(pad_idx).transpose(1, 0).contiguous()
        outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
        y = self.dis_fc1(outputs[0])
        return y[:,:-2], y[:,-2:], outputs[0] #machine_features, human_features
    
    def get_mp_features_mix(self, sx_ids, attr_ids, pad_idx):
        src_key_padding_mask = sx_ids.eq(pad_idx).transpose(1, 0).contiguous()
        outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
        return outputs[0] #machine_features, human_features

    def predict(self, sx_ids, attr_ids, pad_idx):
        outputs = self.get_mp_features(sx_ids, attr_ids, pad_idx)
        labels = outputs.argmax(dim=-1)
        return labels

    def inference(self, sx_ids, attr_ids, pad_idx):
        return self.simulate(sx_ids, attr_ids, pad_idx, inference=True)

    def compute_entropy(self, features):
        return torch.distributions.Categorical(functional.softmax(features, dim=-1)).entropy().item() / self.num_dis

    @staticmethod
    def compute_max_prob(features):
        return torch.max(functional.softmax(features, dim=-1))


class Agent(nn.Module):

    def __init__(self, num_sxs: int, num_dis: int, sv, enc_dropout1, graph=None):

        super().__init__()

        self.symptom_encoder = SymptomEncoderXFMR(
           num_sxs, num_dis, enc_dropout1
        )
        self.num_sxs = num_sxs
        self.sv = sv

    def forward(self):
        pass

        
    def load(self, path):
        if os.path.exists(path):
            state_dict = torch.load(path)
            shared_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict() and 'gnnmodel' not in k}
            self.load_state_dict(shared_state_dict, strict=False)
            # if verbose:
            #     print('loading pre-trained parameters from {} ...'.format(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        # if verbose:
        #     print('saving best model to {}'.format(path))






class SymptomEncoderXFMR1(nn.Module):

    def __init__(self, num_sxs, num_dis, enc_dropout1):
        super().__init__()

        self.num_dis = num_dis
        self.sx_embedding = nn.Embedding(num_sxs, enc_emb_dim, padding_idx=0)
        self.attr_embedding = nn.Embedding(num_attrs, enc_emb_dim, padding_idx=0)
        self.human_embedding = nn.Embedding(num_dis, enc_emb_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=enc_emb_dim,
                nhead=enc_num_heads,
                dim_feedforward=enc_num_layers,
                dropout=enc_dropout1,
                batch_first=False,
                activation='gelu'),
            num_layers=enc_num_layers)
        
        self.dis_fc1 = nn.Linear(enc_emb_dim, num_dis, bias=True)
        self.dis_fc2 = nn.Linear(enc_emb_dim, 1, bias=True)
        self.dropout = nn.Dropout(p=0.25)
        self.activation = nn.Sigmoid()

    def forward(self, sx_ids, attr_ids, mask=None, src_key_padding_mask=None):
        inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids) 
        outputs = self.encoder(inputs, mask, src_key_padding_mask)
        return outputs

    # mean pooling feature
    def get_mp_features(self, sx_ids, attr_ids, human_input, pad_idx):
        src_key_padding_mask = sx_ids.eq(pad_idx).transpose(1, 0).contiguous()
        outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
        human_emb = self.human_embedding(human_input)
        if feature_mode != True:
            hp_outputs = self.dis_fc1(self.dropout(human_emb))
            hp_outputs = torch.softmax(hp_outputs, dim=-1)
        else:
            h_outputs = self.dis_fc1(self.dropout(human_emb+outputs[0]))
            norm_h_outputs = torch.softmax(h_outputs, dim=-1)
            hm_gamma = self.activation(self.dis_fc2(outputs[0]))
            h_preds_onehot = functional.one_hot(human_input, num_classes=self.num_dis).float()
            hp_outputs = norm_h_outputs * hm_gamma + h_preds_onehot * (1-hm_gamma)
            
        return norm_h_outputs

    def get_mp_features_mix(self, sx_ids, attr_ids, pad_idx):
        src_key_padding_mask = sx_ids.eq(pad_idx).transpose(1, 0).contiguous()
        outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
        return outputs[0]

    def predict_mix(self, outputs, human_input):
        human_emb = self.human_embedding(human_input)
        hp_outputs = self.dis_fc1(self.dropout(outputs+human_emb))
        
        return hp_outputs

    def inference(self, sx_ids, attr_ids, pad_idx):
        return self.simulate(sx_ids, attr_ids, pad_idx, inference=True)

    def compute_entropy(self, features):
        return torch.distributions.Categorical(functional.softmax(features, dim=-1)).entropy().item() / self.num_dis

    @staticmethod
    def compute_max_prob(features):
        return torch.max(functional.softmax(features, dim=-1))

class Agent_ECH(nn.Module):

    def __init__(self, num_sxs: int, num_dis: int, sv, enc_dropout1, graph=None):

        super().__init__()

        self.symptom_encoder = SymptomEncoderXFMR1(
           num_sxs, num_dis, enc_dropout1
        )
        self.num_sxs = num_sxs
        self.sv = sv

    def forward(self):
        pass

        
    def load(self, path):
        if os.path.exists(path):
            state_dict = torch.load(path)
            shared_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict() and 'gnnmodel' not in k}
            self.load_state_dict(shared_state_dict, strict=False)
            if verbose:
                print('loading pre-trained parameters from {} ...'.format(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        if verbose:
            print('saving best model to {}'.format(path))

# [ECCV 2024] Learning to Complement or Defer to Multiple Users (LECODU) 
class CollaborationNet(nn.Module):
    def __init__(self, channels=2, hidden_size=512, dim=4):
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
        