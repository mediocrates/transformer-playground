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

    def predict(self, prompt=None, start_token=0, end_token=2, context_len=128, max_len=50):
        """

        Args:
            prompt: Tokenized tensor of indices corresponding to words.
            start_token: Start token.
            end_token: End token.
            context_len: Max length of context when feeding into model.
            max_len: Max length of sequence before terminating generation.
        """
        device = next(self.parameters()).device
        if prompt is not None:
            combined = prompt
        else:
            combined = torch.zeros(1).fill_(start_token).to(torch.int64).to(device)
        pick = None
        while pick != end_token and combined.shape[-1] < max_len:
            self.eval()
            out = self(combined[-context_len:].unsqueeze(0))
            self.train()
            pick = out[0, -1].argmax(dim=-1).unsqueeze(0)
            combined = torch.cat([combined, pick], dim=-1)
        return combined
