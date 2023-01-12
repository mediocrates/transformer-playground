import numpy as np
import torch
from torch import nn

from util import FeedForward, Sublayer


class EncoderSelfAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        A = Q @ K.transpose(-2, -1) / np.sqrt(self.d_model)
        return torch.softmax(A, dim=-1) @ V


class EncoderDecoderAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x, enc_out):
        d_model = x.shape[-1]
        Q = self.W_Q(enc_out)
        K = self.W_K(enc_out)
        V = self.W_V(x)
        A = Q @ K.transpose(-2, -1) / np.sqrt(d_model)
        return torch.softmax(A, dim=-1) @ V


class MultiHeadEncoderAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.heads = nn.ModuleList(EncoderSelfAttention(d_model, d_k, d_v) for _ in range(h))
        self.W_O = nn.Linear(h * d_v, d_model, bias=False)

    def forward(self, x):
        return self.W_O(torch.cat([head(x) for head in self.heads], dim=-1))


class MultiHeadEncoderDecoderAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.heads = nn.ModuleList(EncoderDecoderAttention(d_model, d_k, d_v) for _ in range(h))
        self.W_O = nn.Linear(h * d_v, d_model, bias=False)

    def forward(self, x, enc_out):
        return self.W_O(torch.cat([head(x, enc_out) for head in self.heads], dim=-1))


class Encoder(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, d_ff):
        super().__init__()
        self.attention = Sublayer(
            MultiHeadEncoderAttention(d_model, d_k, d_v, h),
            d_model=d_model
        )
        self.ffn = Sublayer(
            FeedForward(d_model, d_ff),
            d_model=d_model
        )

    def forward(self, x):
        return self.ffn(self.attention(x))


class EncoderStack(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, d_ff, n):
        super().__init__()
        self.encoders = nn.Sequential(*[Encoder(d_model, d_k, d_v, h, d_ff) for _ in range(n)])

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x
