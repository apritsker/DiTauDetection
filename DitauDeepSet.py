from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F



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
        attn_logits = attn_logits / sqrt(d_k)
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
        values, attention = self.scaled_dot_product(q, k, v, mask=None)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o, attention


class DeepSet(nn.Module):
    def __init__(self, in_features, feats, classifier_layers, classify=True):
        super(DeepSet, self).__init__()
        self.classify = classify

        self.layers = nn.ModuleList([])

        self.layers.append(DeepSetLayer(in_features, 0, feats[0]))  # For normalization (probably)
        for i in range(1, len(feats)):
            self.layers.append(DeepSetLayer(in_features, feats[i - 1], feats[i]))

        self.n_layers = len(self.layers)
        self.activ = nn.ReLU()

        class_layers = []
        class_layers.append(nn.Linear(feats[-1], classifier_layers[0]))  # fully connected for the classification
        class_layers.append(nn.ReLU())  # activation function

        for hidden_i in range(1, len(classifier_layers)):
            class_layers.append(nn.Linear(classifier_layers[hidden_i - 1], classifier_layers[hidden_i]))
            class_layers.append(nn.ReLU())

        class_layers.append(nn.Linear(classifier_layers[-1], 2))

        self.classifier = nn.Sequential(*class_layers)
        self.softmax = nn.Softmax(dim=1)

        self.attn = MultiheadAttentionMY(feats[-1], feats[-1], in_features, 1)

    def forward(self, inp):
        # Feature extraction layers
        x = inp
        # if(x.shape[1] == 0): # if there are no tracks, skip.    
        #     return torch.ones((x.shape[0], 2), requires_grad=True).cuda() * 0.5

        for layer_i in range(self.n_layers):
            x = self.layers[layer_i](x)
            if layer_i < self.n_layers - 1:
                x = self.activ(x)
                x = torch.cat((inp, x), dim=2)  # Skip connection.
    
        x = self.attn(x, inp)[0]
        x = x.sum(dim=1)
        if(self.classify):
            x = self.softmax(self.classifier(x))
        return x

class DeepSetLayer(nn.Module):
    def __init__(self, original_in_features, in_features, out_features):
        super(DeepSetLayer, self).__init__()
        mid_features = int((original_in_features + in_features + out_features) / 2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(original_in_features + in_features, mid_features, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(mid_features, out_features, 1, bias=True),
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(original_in_features + in_features, mid_features, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(mid_features, out_features, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, inp):
        x_T = inp.transpose(2, 1)  # B,N,C -> B,C,N
        # Summation and mean are invariant function for the deepset
        x = self.layer1(x_T) + self.layer2(x_T - x_T.mean(dim=2, keepdim=True))
        # normalization
        x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN
        x = x.transpose(1, 2)  # B,C,N -> B,N,C
        return x

