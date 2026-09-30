"""Readout de Broca: vocalizador de superficie (nao-Transformer).

Separa responsabilidades:
- S_i + G + M: raciocinio deliberativo lento (ideia amodal).
- Broca: maquina sequencial rapida de emissao (micro-sintaxe).

h_t = GRUCell([e_t | G_m | pooled(S)], h_{t-1})
logits = W_vocab LN(h_t) + W_skip e_t
         ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
         conceito global       sintaxe local (residual imediato)

O skip de e_t resolve bigramas direto e libera G para semantica.
O concat de pooled(S_A,S_B,S_C) e o cross-readout sobre os modulos.
"""
import torch
import torch.nn as nn


class BrocaReadout(nn.Module):
    def __init__(self, emb_dim, global_dim, pooled_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        inp = int(emb_dim) + int(global_dim) + int(pooled_dim)
        self.gru = nn.GRUCell(inp, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, vocab_size)
        # skip residual: sintaxe local direta e_t -> logits
        self.skip = nn.Linear(emb_dim, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.skip.weight)

    def init_state(self, B, device=None, dtype=None):
        ref = self.head.weight
        device = device or ref.device
        dtype = dtype or ref.dtype
        return torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)

    def forward(self, e_t, Gm, pooled_concat, h_prev):
        x = torch.cat([e_t, Gm, pooled_concat], dim=-1)
        h = self.gru(x, h_prev)
        return self.head(self.norm(h)) + self.skip(e_t), h
