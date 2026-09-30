"""Modulos de estado com viscosidade temporal + quebra de simetria.

Atualizacao:
    S_{t+1} = alpha * S_t + (1 - alpha) * cand
    cand = tanh(in_banda(x) + rec(h_attn) + bias) * (1 + gate(G))

- in_banda: K projecoes por faixa temporal (rapida/media/lenta/ultralenta).
- h_attn: attention readout sobre as R linhas (substitui mean que diluia tudo).
- gate(G): espaco comum como neuromodulacao (habilita/inibe), nao injecao bruta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_alpha_bands(R, bands=(0.1, 0.5, 0.9, 0.98)):
    R = int(R)
    n = len(bands)
    alpha = torch.zeros(R)
    band_idx = torch.zeros(R, dtype=torch.long)
    for r in range(R):
        k = (r * n) // R
        alpha[r] = bands[k]
        band_idx[r] = k
    return alpha, band_idx


class ViscousRNNModule(nn.Module):
    def __init__(self, name, R, D, input_dim, global_dim, alphas="bands", alpha_bands=(0.1, 0.5, 0.9, 0.98)):
        super().__init__()
        self.name = name
        self.R = int(R)
        self.D = int(D)
        self.input_dim = int(input_dim)
        self.global_dim = int(global_dim)
        if isinstance(alphas, str) and alphas == "bands":
            alpha, band_idx = build_alpha_bands(self.R, alpha_bands)
        else:
            alpha = torch.as_tensor(alphas, dtype=torch.float32)
            assert alpha.numel() == self.R
            _, band_idx = build_alpha_bands(self.R, alpha_bands)
        self.K = len(alpha_bands) if isinstance(alphas, str) else int(band_idx.max().item() + 1)
        self.register_buffer("alpha", alpha.view(self.R, 1))
        self.register_buffer("band_idx", band_idx)
        # uma projecao por faixa temporal
        self.in_bands = nn.ModuleList([nn.Linear(self.input_dim, self.D) for _ in range(self.K)])
        # gate neuromodulatorio a partir do espaco comum
        self.gate_proj = nn.Linear(self.global_dim, self.D)
        self.rec = nn.Linear(self.D, self.D, bias=False)
        # attention readout sobre linhas
        self.read_attn = nn.Linear(self.D, 1)
        self.row_bias = nn.Parameter(torch.zeros(self.R, self.D))
        for p in self.in_bands:
            nn.init.xavier_uniform_(p.weight)
            nn.init.zeros_(p.bias)
        nn.init.xavier_uniform_(self.rec.weight)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def init_state(self, B, device=None, dtype=None):
        device = device or self.row_bias.device
        dtype = dtype or self.row_bias.dtype
        return torch.zeros(B, self.R, self.D, device=device, dtype=dtype)

    def pooled(self, S):
        # attention pooling: [B,R,D] -> [B,D]
        w = torch.softmax(self.read_attn(S), dim=1)  # [B,R,1]
        return (w * S).sum(dim=1)

    def band_means(self, S):
        # [B,K,D] medias por faixa (para analise / pooling estruturado)
        B = S.shape[0]
        out = torch.zeros(B, self.K, self.D, device=S.device, dtype=S.dtype)
        for k in range(self.K):
            mask = (self.band_idx == k)
            if mask.any():
                out[:, k, :] = S[:, mask, :].mean(dim=1)
        return out

    def forward(self, x, g, S):
        # x:[B,input_dim] visao privada | g:[B,global_dim] | S:[B,R,D]
        band_outs = torch.stack([p(x) for p in self.in_bands], dim=1)  # [B,K,D]
        in_r = band_outs[:, self.band_idx, :]                          # [B,R,D]
        h_pool = self.pooled(S.detach() if False else S)               # [B,D]
        rec_term = self.rec(h_pool).unsqueeze(1)                       # [B,1,D]
        gate = torch.sigmoid(self.gate_proj(g)).unsqueeze(1)           # [B,1,D] em [0,1]
        base = in_r + rec_term + self.row_bias.unsqueeze(0)
        cand = torch.tanh(base) * (1.0 + gate)                         # modulacao, nao injecao
        a = self.alpha.unsqueeze(0)                                    # [1,R,1]
        return a * S + (1.0 - a) * cand

    def mutate_alphas(self, noise=0.05):
        with torch.no_grad():
            self.alpha.add_(torch.randn_like(self.alpha) * noise).clamp_(0.0, 0.995)


def diversity_loss(summaries):
    """Penalidade de ortogonalidade: sum_{i<j} cos^2(h_i,h_j)."""
    n = len(summaries)
    if n < 2:
        return summaries[0].new_zeros(())
    loss = summaries[0].new_zeros(())
    for i in range(n):
        for j in range(i + 1, n):
            hi = F.normalize(summaries[i], dim=-1)
            hj = F.normalize(summaries[j], dim=-1)
            cos = (hi * hj).sum(dim=-1)  # [B]
            loss = loss + (cos ** 2).mean()
    return loss
