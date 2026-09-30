"""
Benchmark comparativo blindado: Broca vs Linear no Dataset de Português.

Blindagens metodológicas (revisão):
  1. Train CE puro logado separado do total (CE+div+sym) — o sym=-H_bar
     torna o total negativo e incomparável entre modelos.
  2. Warmup CUDA + ordem aleatória por seed — o 1º modelo pagava warmup.
  3. Tokenizer com fit SOMENTE no train — antes vazava val para o vocab.
  4. Multi-seed (default 42,43,44) com média±desvio.

Uso:
  python compare_readouts.py --epochs 3
  python compare_readouts.py --epochs 1 --limit 120 --seeds 42   # smoke test
"""
import argparse
import json
import time
import math
import random
import torch
import torch.nn.functional as F

from tru_net.chunk_tokenizer import ChunkTokenizer
from tru_net.tru_net import TRUNet, TRUConfig
from train_portugues import load_texts


def evaluate(model, val_data, seq_len, batch_size, device, tok_len):
    model.eval()
    tot_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            b = val_data[i:i + batch_size]
            if not b:
                continue
            x = torch.tensor([s[:seq_len] for s in b], device=device)
            y = torch.tensor([s[1:seq_len + 1] for s in b], device=device)
            logits, _ = model.forward_text(x, think_steps=2, update_memory=False)
            loss = F.cross_entropy(logits.reshape(-1, tok_len), y.reshape(-1))
            tot_loss += loss.item()
            n_batches += 1
    avg_loss = tot_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def warmup(model, device, tok_len, seq_len=32):
    """5 passos dummy para aquecer kernels CUDA/allocator antes do cronômetro."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, tok_len, (4, seq_len), device=device)
    y = torch.randint(0, tok_len, (4, seq_len), device=device)
    for _ in range(5):
        logits, _, aux = model.forward_text(x, think_steps=1, return_aux=True, update_memory=False)
        loss = F.cross_entropy(logits.reshape(-1, tok_len), y.reshape(-1)) + aux["div"] * 0.0 + aux["sym"] * 0.0
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_experiment(readout_type, train_data, val_data, tok, epochs, batch_size, seq_len, think_steps, symbols, div_w, device, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    cfg = TRUConfig(R=(16, 24, 12), D=(32, 32, 48), obs_dim=(32, 32, 32),
                    msg_dim=16, global_dim=64, vocab_size=len(tok),
                    emb_dim=66, n_actions=8, n_symbols=symbols,
                    div_weight=div_w, readout=readout_type)
    model = TRUNet(cfg).to(device)
    warmup(model, device, len(tok), seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    history = {"train_ce": [], "train_tot": [], "val_loss": [], "val_ppl": []}
    start_time = time.time()

    for ep in range(epochs):
        model.train()
        random.shuffle(train_data)
        tot_ce, tot_all, n_batches = 0.0, 0.0, 0
        for i in range(0, len(train_data), batch_size):
            b = train_data[i:i + batch_size]
            if not b:
                continue
            x = torch.tensor([s[:seq_len] for s in b], device=device)
            y = torch.tensor([s[1:seq_len + 1] for s in b], device=device)
            logits, _, aux = model.forward_text(x, think_steps=think_steps, return_aux=True, update_memory=True)
            ce = F.cross_entropy(logits.reshape(-1, len(tok)), y.reshape(-1))
            loss = ce
            if div_w > 0:
                loss = loss + div_w * aux["div"]
            if aux["sym"] is not None:
                loss = loss + aux["sym"]

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_ce += ce.item()
            tot_all += loss.item()
            n_batches += 1

        train_ce = tot_ce / max(n_batches, 1)
        train_tot = tot_all / max(n_batches, 1)
        val_loss, val_ppl = evaluate(model, val_data, seq_len, batch_size, device, len(tok))
        history["train_ce"].append(train_ce)
        history["train_tot"].append(train_tot)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        print(f"[{readout_type.upper()}] Epoch {ep+1}/{epochs} | TrainCE: {train_ce:.3f} | TrainTot: {train_tot:.3f} | Val: {val_loss:.3f} | PPL: {val_ppl:.2f}")

    elapsed = time.time() - start_time
    return model, history, elapsed


def mean_std(xs):
    import statistics
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_portugues_concurso.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--chunk_size", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--think", type=int, default=2)
    ap.add_argument("--limit", type=int, default=877)
    ap.add_argument("--symbols", type=int, default=16)
    ap.add_argument("--div_w", type=float, default=0.1)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== BENCHMARK BLINDADO: BROCA vs LINEAR ({len(seeds)} seeds) ===")
    print(f"Dispositivo: {device} (GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'})")

    all_texts = load_texts(args.data, args.limit)
    per_seed, gen_models = {}, {}
    for seed in seeds:
        # split ANTES do fit: tokenizer vê só o train
        rng = random.Random(seed)
        idx = list(range(len(all_texts)))
        rng.shuffle(idx)
        cut = int(0.9 * len(idx))
        train_texts = [all_texts[i] for i in idx[:cut]]
        val_texts = [all_texts[i] for i in idx[cut:]]
        tok = ChunkTokenizer(chunk_size=args.chunk_size, vocab_size=args.vocab).fit(train_texts)
        data_tr = [tok.encode(t) for t in train_texts]
        data_va = [tok.encode(t) for t in val_texts]
        data_tr = [s for s in data_tr if len(s) > args.seq + 1]
        data_va = [s for s in data_va if len(s) > args.seq + 1]
        print(f"\n--- seed {seed}: train={len(data_tr)} val={len(data_va)} V={len(tok)} (fit só no train) ---")

        # ordem aleatória por seed: ninguém paga warmup sistematicamente
        order = ["linear", "broca"]
        random.Random(seed).shuffle(order)
        print(f"ordem: {order}")
        for ro in order:
            model, hist, elapsed = run_experiment(
                ro, data_tr, data_va, tok, epochs=args.epochs,
                batch_size=args.batch, seq_len=args.seq, think_steps=args.think,
                symbols=args.symbols, div_w=args.div_w, device=device, seed=seed)
            per_seed.setdefault(ro, []).append((hist, elapsed))
            if seed == seeds[0]:
                gen_models[ro] = (model, tok)

    # agregação
    import statistics
    rows = {}
    for ro in ("linear", "broca"):
        runs = per_seed[ro]
        for key in ("train_ce", "train_tot", "val_loss", "val_ppl"):
            vals = [h[key][-1] for h, _ in runs]
            rows[(ro, key)] = (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        ts = [t for _, t in runs]
        rows[(ro, "time")] = (statistics.mean(ts), statistics.pstdev(ts) if len(ts) > 1 else 0.0)

    def cell(ro, key, fmt="{:.3f}"):
        m, s = rows[(ro, key)]
        return fmt.format(m) + f" ± {s:.3f}" if len(seeds) > 1 else fmt.format(m)

    # qualitativo com os modelos da 1ª seed (todos os 3 prompts, sem cherry-pick)
    results_gen = []
    prompts = ["A crase e obrigatoria", "O sujeito da oracao", "Concordancia verbal"]
    model_lin, tok_g = gen_models["linear"]
    model_bro, _ = gen_models["broca"]
    for p in prompts:
        pref = torch.tensor(tok_g.encode(p), device=device)
        out_lin = tok_g.decode(model_lin.generate(pref, steps=16, think_steps=args.think))
        out_broca = tok_g.decode(model_bro.generate(pref, steps=16, think_steps=args.think))
        results_gen.append({"prompt": p, "linear": out_lin, "broca": out_broca})
        print(f"\n[PROMPT]: {p}\n  Linear: {repr(out_lin)}\n  Broca:  {repr(out_broca)}")

    vb = rows[("broca", "val_ppl")][0]
    vl = rows[("linear", "val_ppl")][0]
    report = f"""# Comparativo Blindado: Broca vs Linear (TRU-Net)

**Ambiente:** `{device}` ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})
**Config:** épocas={args.epochs} seq={args.seq} batch={args.batch} símbolos={args.symbols} seeds={args.seeds}
**Protocolo:** tokenizer com fit só no train; warmup CUDA + ordem aleatória; val em CE puro.

## Métricas (média ± desvio entre seeds, última época)

| Readout | Train CE | Train Tot (c/ aux) | Val Loss (CE) | Val PPL | Tempo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear** | {cell('linear','train_ce')} | {cell('linear','train_tot')} | {cell('linear','val_loss')} | **{cell('linear','val_ppl','{:.2f}')}** | {cell('linear','time','{:.1f}')}s |
| **Broca** | {cell('broca','train_ce')} | {cell('broca','train_tot')} | {cell('broca','val_loss')} | **{cell('broca','val_ppl','{:.2f}')}** | {cell('broca','time','{:.1f}')}s |

Redução de PPL: {vl/vb:.1f}x (linear {vl:.2f} → broca {vb:.2f}).
Nota: Train Tot inclui o bônus InfoMax (-H̄) e é negativo por construção — comparar Train CE.

## Geração (seed {seeds[0]}, 3 prompts sem cherry-pick)
"""
    for item in results_gen:
        report += f"\n### Prompt: `{item['prompt']}`\n- **Linear:** `{item['linear']}`\n- **Broca:** `{item['broca']}`\n"

    with open("comparativo_readout.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nRelatório salvo em 'comparativo_readout.md'!")


if __name__ == "__main__":
    main()
