"""Canal limitado com InfoMax: anti-colapso de codebook.

- continuo: m = tanh(LN(W flat(S)))
- discreto: simbolos via Gumbel-Softmax + entropy bonus + ruido de canal + annealing.

L_symbol = -H(p_bar) + beta * mean(H(p_b))
  maximiza uso uniforme do vocab (marginal), minimiza indecisao por mensagem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def symbol_loss(logits, beta=0.3):
    # logits: [B,K]
    p = torch.softmax(logits, dim=-1)
    p_bar = p.mean(dim=0)                                   # [K]
    H_bar = -(p_bar * torch.log(p_bar + 1e-8)).sum()
    H_cond = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
    return -H_bar + beta * H_cond, H_bar.detach(), H_cond.detach()


class LimitedChannel(nn.Module):
    def __init__(self, R, D, msg_dim, n_symbols=0, tau=1.0, noise_p=0.0):
        super().__init__()
        self.R, self.D = int(R), int(D)
        self.msg_dim = int(msg_dim)
        self.n_symbols = int(n_symbols)
        self.tau = float(tau)
        self.noise_p = float(noise_p)
        flat = self.R * self.D
        if self.n_symbols <= 0:
            self.proj = nn.Linear(flat, self.msg_dim)
            self.norm = nn.LayerNorm(self.msg_dim)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            self.to_logits = nn.Linear(flat, self.n_symbols)
            self.sym_emb = nn.Embedding(self.n_symbols, self.msg_dim)
            nn.init.xavier_uniform_(self.to_logits.weight)
            nn.init.zeros_(self.to_logits.bias)

    def set_tau(self, tau):
        self.tau = float(tau)

    def set_noise(self, p):
        self.noise_p = float(p)

    def forward(self, S, hard=False, noise_p=None):
        B = S.shape[0]
        flat = S.reshape(B, -1)
        if self.n_symbols <= 0:
            m = torch.tanh(self.norm(self.proj(flat)))
            return m, {"symbols": None, "logits": None}
        logits = self.to_logits(flat)
        if self.training and not hard:
            y = F.gumbel_softmax(logits, tau=self.tau, hard=False)
            m = y @ self.sym_emb.weight
            sym = y.argmax(dim=-1)
        else:
            sym = logits.argmax(dim=-1)
            m = self.sym_emb(sym)
        # ruido de canal: troca simbolo por aleatorio (forca redundancia)
        p = self.noise_p if noise_p is None else float(noise_p)
        if self.training and p > 0.0:
            mask = (torch.rand(B, device=S.device) < p)
            if mask.any():
                rnd = torch.randint(0, self.n_symbols, (int(mask.sum()),), device=S.device)
                sym = sym.clone()
                sym[mask] = rnd
                m = m.clone()
                m[mask] = self.sym_emb(rnd)
        return torch.tanh(m), {"symbols": sym, "logits": logits}
