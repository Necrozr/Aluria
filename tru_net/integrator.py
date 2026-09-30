"""Espaco comum: recebe MENSAGENS, nao estados privados."""
import torch
import torch.nn as nn


class CommonSpace(nn.Module):
    def __init__(self, msg_dim, n_modules, global_dim, alpha_global=0.3):
        super().__init__()
        self.global_dim = int(global_dim)
        self.gru = nn.GRUCell(msg_dim * n_modules, self.global_dim)
        # viscosidade global: quanto do passado sobrevive
        self.alpha_global = float(alpha_global)

    def init_state(self, B, device=None):
        # GRUCell nao tem estado fixo; usamos vetor zero como G0
        ref = next(self.parameters())
        device = device or ref.device
        return torch.zeros(B, self.global_dim, device=device, dtype=ref.dtype)

    def forward(self, msgs, G):
        # msgs: lista de [B, Mc]
        x = torch.cat(msgs, dim=-1)
        cand = self.gru(x, G)
        a = self.alpha_global
        return a * G + (1.0 - a) * cand
