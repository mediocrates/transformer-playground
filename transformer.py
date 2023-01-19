import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, p_dropout=0.1):
        assert d_model % 2 == 0
        super().__init__()
        self.d_model = d_model
        i = 2 * torch.arange(self.d_model // 2, dtype=torch.float)
        self.base_row = 10000 ** (-i.unsqueeze(0) / self.d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        L = x.shape[-2]
        cols = torch.arange(L, dtype=torch.float).unsqueeze(-1).to(x.device)
        half = cols * self.base_row.to(x.device)
        evens = torch.sin(half)
        odds = torch.cos(half)
        return self.dropout(x + torch.stack([evens, odds], dim=-1).view(L, self.d_model))


class DecoderSelfAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v=None):
        if d_v is None:
            d_v = d_k
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        d_k = x.shape[-1]
        A = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        A += torch.triu(torch.ones_like(A) * float("-inf"), diagonal=1)  # mask subsequent positions
        return torch.softmax(A, dim=-1) @ V


class MultiHeadDecoderAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.heads = nn.ModuleList(DecoderSelfAttention(d_model, d_k, d_v) for _ in range(h))
        self.W_O = nn.Linear(h * d_v, d_model, bias=False)

    def forward(self, x):
        return self.W_O(torch.cat([head(x) for head in self.heads], dim=-1))


class Sublayer(nn.Module):

    def __init__(self, layer, d_model, p_dropout=0.1):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.layer(x)))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=None):
        if d_ff is None:
            d_ff = 4 * d_model
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, d_ff=None, p_dropout=0.1):
        if d_v is None:
            d_v = d_k
        super().__init__()
        self.self_attention = Sublayer(
            MultiHeadDecoderAttention(d_model, d_k, d_v, h),
            d_model=d_model,
            p_dropout=p_dropout
        )
        self.ffn = Sublayer(
            FeedForward(d_model, d_ff),
            d_model=d_model,
            p_dropout=p_dropout
        )

    def forward(self, x):
        return self.ffn(self.self_attention(x))


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, d_vocab, d_model, d_k, d_v=None, h=8, d_ff=None, n=12, p_dropout=0.1, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=d_vocab, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model, p_dropout=p_dropout)
        self.decoders = nn.Sequential(*[Decoder(d_model, d_k, d_v, h, d_ff, p_dropout=p_dropout) for _ in range(n)])
        self.final = nn.Linear(d_model, d_vocab)

        self.tie_weights = tie_weights
        if tie_weights:
            # this works because nn.Embedding(x, y).weight is x by y, and nn.Linear(y, x).weight is ALSO x by y
            # and I think that by setting embedding.weight to something else, the old reference to nn.weight disappears.
            # but not 100% sure it goes away due to having 0 references
            self.embedding.weight = self.final.weight

    def forward(self, combined):
        a = self.embedding(combined)
        if self.tie_weights:
            # nn.Embedding weights are initialized to N(0, 1)
            # nn.Linear    weights are initialized to U(-sqrt(k), sqrt(k)) where k = 1/d_model
            # X = U(-x, x) has std(X) =  x/sqrt(3), so try to preserve the variance in the embedding layer by multiplying by sqrt(3)/x = sqrt(3 * d_model)
            a *= np.sqrt(3 * self.d_model)
        a += self.pe.forward(a)
        return self.final(self.decoders(a))
