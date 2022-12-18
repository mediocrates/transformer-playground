import numpy as np
import torch
from torch import nn

from util import FeedForward, MultiHeadDecoderAttention, PositionalEncoding, Sublayer


class Decoder(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, d_ff=None, p_dropout=0.1):
        if d_v is None:
            d_v = d_k
        super().__init__()
        self.self_attention = Sublayer(MultiHeadDecoderAttention(d_model, d_k, d_v, h), p_dropout=p_dropout)
        self.ffn = Sublayer(FeedForward(d_model, d_ff), p_dropout=p_dropout)

    def forward(self, x):
        return self.ffn(self.self_attention(x))


class DecoderStack(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, d_ff, n, p_dropout=0.1):
        super().__init__()
        self.decoders = nn.Sequential(*[Decoder(d_model, d_k, d_v, h, d_ff, p_dropout=p_dropout) for _ in range(n)])

    def forward(self, x):
        for decoder in self.decoders:
            x = decoder(x)
        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, d_vocab, d_model, d_k, d_v=None, h=8, d_ff=None, n=6, p_dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=d_vocab, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model)
        self.decoders = DecoderStack(d_model, d_k, d_v, h, d_ff, n, p_dropout=p_dropout)

    def forward(self, tgt, src, delim_token=1):
        """
        Args:
            tgt: desired output
            src: input representation, if it exists
            delim_token: character delimiting between source and target
        """
        bs = tgt.shape[0]
        d_model = tgt.shape[-1]
        combined = torch.cat([tgt, torch.tensor([delim_token]).to(tgt.device).expand(bs, -1), src], dim=-1)
        a = self.embedding(combined)
        a += self.pe.forward(a)
        return torch.softmax(
            self.decoders(a).mean(dim=-2) @ self.embedding.weight.transpose(-2, -1) / np.sqrt(3 * d_model),
            dim=-1
        )

    def predict(self, src=None, tgt=None, start_token=0, delim_token=1, end_token=2, max_len=50):
        device = next(self.parameters()).device
        if tgt is None:
            tgt = torch.zeros(1, 1).fill_(start_token).to(torch.int64).to(device)
        if src is None:
            src = tgt
        pick = None
        while pick != end_token and tgt.shape[-1] < max_len:
            out = self.forward(tgt, src, delim_token=delim_token)
            pick = out.argmax(dim=-1)
            tgt = torch.cat([tgt, pick.unsqueeze(-1)], dim=-1)
        return tgt
