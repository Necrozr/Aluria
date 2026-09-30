"""Demo rapida: tokenizer + viscosidade + X+Y+Z mini + geracao.
Uso: python demo.py
"""
import torch
import torch.nn.functional as F
from tru_net.chunk_tokenizer import ChunkTokenizer
from tru_net.tru_net import TRUNet, TRUConfig
from train_toy import ToyWrap

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")
if device == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

print("== 1. ChunkTokenizer (por chunks, sem BPE) ==")
tok = ChunkTokenizer(chunk_size=4, vocab_size=200).fit(
    ["A crase e obrigatoria antes de nomes.", "Concordancia verbal e nominal.", "A crase e obrigatoria."])
ids = tok.encode("A crase e obrigatoria.")
print("ids:", ids, "->", repr(tok.decode(ids)))

print("\n== 2. TRUNet forward_text (estado persiste, sem transformer) ==")
cfg = TRUConfig(vocab_size=len(tok))
net = TRUNet(cfg).to(device)
x = torch.tensor([tok.encode("A crase e obrigatoria.")[:8]], device=device)
logits, (S, G) = net.forward_text(x, think_steps=2)
print("logits:", tuple(logits.shape), "| S shapes:", [tuple(s.shape) for s in S], "| G:", tuple(G.shape))
print("alphas A (4 faixas rapida/media/lenta/muito-lenta):", net.states[0].alpha[:8].squeeze().tolist())
print("canal: R*D =", net.states[0].R * net.states[0].D, "-> msg", cfg.msg_dim, "(gargalo)")

print("\n== 3. Toy X+Y+Z: A ve X, B ve Y, C ve Z; alvo=(X+Y+Z)%8 ==")
print("Nenhum modulo sozinho resolve; precisa comunicar pelo canal limitado.")
torch.manual_seed(0)
MOD = 8
small = TRUConfig(R=(8, 8, 8), D=(16, 16, 16), obs_dim=(16, 16, 16),
                  msg_dim=16, global_dim=32, vocab_size=64, emb_dim=32, n_actions=MOD)
m = ToyWrap(small).to(device)
opt = torch.optim.Adam(m.parameters(), lr=5e-3)
for st in range(1, 601):
    xb = torch.randint(0, MOD, (128,), device=device)
    yb = torch.randint(0, MOD, (128,), device=device)
    zb = torch.randint(0, MOD, (128,), device=device)
    tgt = (xb + yb + zb) % MOD
    lg, _ = m(xb, yb, zb, steps=6)
    loss = F.cross_entropy(lg, tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if st % 200 == 0:
        acc = (lg.argmax(-1) == tgt).float().mean().item()
        print(f"[{st}/600] loss={loss.item():.3f} acc={acc*100:.1f}%")
m.eval()
with torch.no_grad():
    xe = torch.randint(0, MOD, (2000,), device=device)
    ye = torch.randint(0, MOD, (2000,), device=device)
    ze = torch.randint(0, MOD, (2000,), device=device)
    te = (xe + ye + ze) % MOD
    lge, hist = m(xe, ye, ze, steps=6)
    acc = (lge.argmax(-1) == te).float().mean().item()
    msg = torch.stack(hist[-1], dim=1)
    print(f"FINAL acc={acc*100:.1f}% (chance={100/MOD:.1f}%) | msgs std={msg.std():.3f} (std~0 => canal morto)")

print("\nOK. Experimentos:")
print("  python train_toy.py            # config grande, ~2500 passos, espera ~100%")
print("  python train_portugues.py --epochs 3  # next-chunk no seu dataset")
