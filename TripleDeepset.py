# Imports
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import itertools
import math
#from google.colab import drive
import sys
import pandas as pd
import numpy as np
import glob
import os
from sklearn import metrics
import pickle
import gzip
import urllib.request
import tempfile


# Define model
class MultiheadAttentionMY(nn.Module):
    def __init__(self, input_dim, embed_dim, orig_inp_dim, num_heads=1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim + orig_inp_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        # perform original Transformer initialization
        self._reset_parameters()

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, input, mask=None):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(torch.cat((x, input), dim=2))

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs - from transformers read RAJA's presentation
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o, attention

class DeepSetLayer(nn.Module):
    def __init__(self, original_in_features, in_features, out_features, kernel_size=1):
        super(DeepSetLayer, self).__init__()
        mid_features = int((original_in_features + in_features + out_features) / 2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(original_in_features + in_features, mid_features, kernel_size, bias=True),
            nn.ReLU(),
            nn.Conv1d(mid_features, out_features, kernel_size, bias=True),
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(original_in_features + in_features, mid_features, kernel_size, bias=True),
            nn.ReLU(),
            nn.Conv1d(mid_features, out_features, kernel_size, bias=True),
            nn.Tanh()
        )

    def forward(self, inp, avoid_maxpooling=False):
        x_T = inp.transpose(2, 1)  # B,N,C -> B,C,N
        # Summation and mean are invariant function for the deepset
        x = self.layer1(x_T) + self.layer2(x_T - x_T.mean(dim=2, keepdim=True))
        # normalization
        x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN
        x = x.transpose(1, 2)  # B,C,N -> B,N,C
        if not avoid_maxpooling:
            x = torch.max_pool1d(x, 4, 4)
        return x



class DeepSet(nn.Module):
    def __init__(self, in_features, conv_feats, attn_feats, classifier_layers=None):
        super(DeepSet, self).__init__()

        self.layers = nn.ModuleList([])

        self.layers.append(DeepSetLayer(in_features, 0, conv_feats[0]))  # For normalization (probably)
        for i in range(len(conv_feats)):
            self.layers.append(DeepSetLayer(in_features, conv_feats[i], conv_feats[i]))

        self.n_conv_layers = len(self.layers)
        self.activ = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attn_layers = nn.ModuleList([])
        for k in range(len(attn_feats)):
            self.attn_layers.append(MultiheadAttentionMY(attn_feats[k], attn_feats[k], in_features, 2))

    def forward(self, inp):
        # Feature extraction layers
        x = inp
        for layer_i in range(self.n_conv_layers):
            if layer_i == 0:
                x = self.layers[layer_i](x, avoid_maxpooling=True)
            else:
                if layer_i < self.n_conv_layers - 1:
                    x = self.layers[layer_i](x)
                else:
                    x = self.layers[layer_i](x, avoid_maxpooling=True)
            if layer_i < self.n_conv_layers - 1:
                x = self.activ(x)
                x = torch.cat((inp, x), dim=2)  # Skip connection.
        #x = torch.cat((inp, x), dim=2)
        for layer_k in range(len(self.attn_layers)):
            if layer_k != 0:
                x, attn = self.attn_layers[layer_k](x, inp, attn)
            else:
                x, attn = self.attn_layers[layer_k](x, inp)
        summy = x.sum(dim=1)
        return summy


class TripleDeepSet(nn.Module):
    def __init__(self, in_track_features, in_em_features, in_had_features, conv_feats,  attn_feats, cls_out):
        super(TripleDeepSet, self).__init__()
        self.tracks_deepset = DeepSet(in_track_features, conv_feats, attn_feats)
        self.em_deepset = DeepSet(in_em_features, conv_feats, attn_feats)
        self.had_deepset = DeepSet(in_had_features, conv_feats, attn_feats)
        self.do_layer = nn.Dropout(0.5)
        self.class_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])
        self.activ = nn.ReLU()
        features = cls_out
        for hidden_i in range(1, len(features)):
            self.class_layers.append(nn.Linear(features[hidden_i - 1], features[hidden_i]))
            self.bn_layers.append(nn.BatchNorm1d(features[hidden_i]))

    def forward(self, tracks, em, had):
        tracks_out = self.tracks_deepset(tracks)
        em_out = self.em_deepset(em)
        had_out = self.had_deepset(had)
        x = torch.cat((tracks_out, em_out, had_out), dim=1)
        inp = x
        for i in range(len(self.class_layers)):
            x = self.class_layers[i](x)
            if x.shape[0] > 1:
                x = self.bn_layers[i](x)

            if i < len(self.class_layers) - 1:
                x = self.activ(x)
                #x = self.do_layer(x)
        return x
