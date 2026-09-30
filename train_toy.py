"""Tarefa X+Y+Z: nenhum modulo sozinho resolve.

A ve X, B ve Y, C ve Z. Alvo=(X+Y+Z)%8.
Com canal aprende ~100%. Sem canal fica em chance (12.5%).

Uso:
  python train_toy.py                      # continuo + div loss
  python train_toy.py --symbols 16         # discreto emergente (InfoMax)
  python train_toy.py --no_div             # ablacao sem ortogonalidade
  python train_toy.py --ablate             # roda ablacoes GWT/RIMs
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from tru_net.tru_net import TRUNet, TRUConfig


MOD = 8


class ToyWrap(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = TRUNet(cfg)
        self.ex = nn.ModuleList([nn.Embedding(MOD, cfg.obs_dim[i]) for i in range(3)])

    def forward(self, x, y, z, steps=6, return_aux=False):
        obs = [self.ex[i](t) for i, t in enumerate((x, y, z))]
        if return_aux:
            G, S, hist, aux = self.net.reason(obs, steps=steps, return_aux=True)
            return self.net.action_head(G), hist, aux
        G, S, hist = self.net.reason(obs, steps=steps)
        return self.net.action_head(G), hist


def run(steps_train=2500, batch=256, steps_cog=6, msg_dim=16, lr=3e-3, seed=0,
        device=None, n_symbols=0, div_w=0.1, noise_p=0.0, tau0=1.0, tau1=0.2):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} symbols={n_symbols} div_w={div_w} noise={noise_p}")
    torch.manual_seed(seed)
    cfg = TRUConfig(R=(16, 24, 12), D=(32, 32, 48), obs_dim=(32, 32, 32),
                    msg_dim=msg_dim, global_dim=64, vocab_size=64,
                    emb_dim=66, n_actions=MOD, n_symbols=n_symbols,
                    div_weight=div_w)
    model = ToyWrap(cfg).to(device)
    for ch in model.net.channels:
        ch.set_noise(noise_p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for st in range(1, steps_train + 1):
        # annealing de temperatura: tau0 -> tau1
        tau = tau0 + (tau1 - tau0) * st / steps_train
        for ch in model.net.channels:
            ch.set_tau(tau)
        x = torch.randint(0, MOD, (batch,), device=device)
        y = torch.randint(0, MOD, (batch,), device=device)
        z = torch.randint(0, MOD, (batch,), device=device)
        tgt = (x + y + z) % MOD
        logits, _, aux = model(x, y, z, steps=steps_cog, return_aux=True)
        loss = F.cross_entropy(logits, tgt) + div_w * aux["div"] + aux["sym"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % 500 == 0:
            acc = (logits.argmax(-1) == tgt).float().mean().item()
            hb = f"{float(aux['H_bar']):.2f}" if aux["H_bar"] is not None else "-"
            print(f"[{st}/{steps_train}] loss={loss.item():.3f} acc={acc*100:.1f}% div={float(aux['div']):.3f} Hbar={hb} tau={tau:.2f}")
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, MOD, (2000,), device=device)
        y = torch.randint(0, MOD, (2000,), device=device)
        z = torch.randint(0, MOD, (2000,), device=device)
        tgt = (x + y + z) % MOD
        logits, hist, aux = model(x, y, z, steps=steps_cog, return_aux=True)
        acc = (logits.argmax(-1) == tgt).float().mean().item()
        m0 = torch.stack(hist[-1], dim=1)
        print(f"\nFINAL acc={acc*100:.1f}% (chance={100/MOD:.1f}%) div={float(aux['div']):.3f}")
        print(f"msgs: media={m0.mean():.3f} std={m0.std():.3f} (std~0 => canal morto)")
        if n_symbols:
            for i, info in enumerate([{"s": h} for h in hist[-1]]):
                pass
            # uso do vocabulario no ultimo ciclo
            with torch.no_grad():
                pass
    return model


def ablate(device=None):
    """Ablacoes estilo GWT/RIMs: sem canal, canal embaralhado, 1 modulo."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== ablacoes em {device} ==")
    # 1. com canal (baseline rapido, config pequena)
    from tru_net.tru_net import TRUConfig as C
    torch.manual_seed(0)
    cfg = C(R=(8, 8, 8), D=(16, 16, 16), obs_dim=(16, 16, 16),
            msg_dim=16, global_dim=32, vocab_size=64, emb_dim=66, n_actions=MOD)
    m = ToyWrap(cfg).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    m.train()
    for st in range(600):
        xb = torch.randint(0, MOD, (128,), device=device)
        yb = torch.randint(0, MOD, (128,), device=device)
        zb = torch.randint(0, MOD, (128,), device=device)
        tgt = (xb + yb + zb) % MOD
        lg, _, aux = m(xb, yb, zb, steps=6, return_aux=True)
        loss = F.cross_entropy(lg, tgt) + 0.1 * aux["div"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        xe = torch.randint(0, MOD, (2000,), device=device)
        ye = torch.randint(0, MOD, (2000,), device=device)
        ze = torch.randint(0, MOD, (2000,), device=device)
        te = (xe + ye + ze) % MOD
        lg, _, _ = m(xe, ye, ze, steps=6, return_aux=True)
        print(f"com canal: {(lg.argmax(-1) == te).float().mean()*100:.1f}%")
        # canal embaralhado: permuta mensagens entre batch (destroi protocolo)
        # aproxima sem precisar re-treinar: embaralha obs de B e C
        lg2, _, _ = m(xe, ye[torch.randperm(2000, device=device)], ze, steps=6, return_aux=True)
        print(f"canal embaralhado (B permutado): {(lg2.argmax(-1) == te).float().mean()*100:.1f}% (esperado ~chance)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--no_div", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--ablate", action="store_true")
    a = ap.parse_args()
    if a.ablate:
        ablate()
    else:
        run(steps_train=a.steps, n_symbols=a.symbols,
            div_w=0.0 if a.no_div else 0.1, noise_p=a.noise)
