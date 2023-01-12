import numpy as np
import torch
from torch import nn

from util import FeedForward, MultiHeadDecoderAttention, PositionalEncoding, Sublayer


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

    def __init__(self, d_vocab, d_model, d_k, d_v=None, h=8, d_ff=None, n=12, p_dropout=0.1, tie_weights=False):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=d_vocab, embedding_dim=d_model)
        # torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.pe = PositionalEncoding(d_model, p_dropout=p_dropout)
        self.decoders = nn.Sequential(*[Decoder(d_model, d_k, d_v, h, d_ff, p_dropout=p_dropout) for _ in range(n)])
        self.final = nn.Linear(d_model, d_vocab)

        # TODO (justin): this doesn't exactly work yet
        self.tie_weights = tie_weights
        if tie_weights:
            self.embedding.weight = self.final.weight  # weight-tying

    def forward(self, combined):
        a = self.embedding(combined)
        if self.tie_weights:
            a *= np.sqrt(3 * self.d_model)
        a += self.pe.forward(a)
        # self.decoders(a) @ self.embedding.weight.transpose(-2, -1) / np.sqrt(3 * d_model),
        return self.final(self.decoders(a))

    def predict(self, prompt=None, start_token=0, end_token=2, max_len=50):
        device = next(self.parameters()).device
        if prompt is not None:
            combined = prompt
        else:
            combined = torch.zeros(1).fill_(start_token).to(torch.int64).to(device)
        pick = None
        while pick != end_token and combined.shape[-1] < max_len:
            self.eval()
            out = self(combined.unsqueeze(0))
            self.train()
            pick = out[0, -1].argmax(dim=-1).unsqueeze(0)
            combined = torch.cat([combined, pick], dim=-1)
        return combined
