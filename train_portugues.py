"""Treino next-chunk no dataset de portugues (RNN, sem Transformer).

Uso: python train_portugues.py --epochs 3 --chunk_size 4 --vocab 2000
"""
import argparse, json
import torch
import torch.nn.functional as F
from tru_net.chunk_tokenizer import ChunkTokenizer
from tru_net.tru_net import TRUNet, TRUConfig


def load_texts(path, limit=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            o = json.loads(line)
            t = (o.get("instruction", "") + "\n" + o.get("input", "") + "\n" + o.get("output", ""))[:2000]
            texts.append(t)
    return texts


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
    ap.add_argument("--symbols", type=int, default=0, help=">0 ativa canal discreto emergente")
    ap.add_argument("--div_w", type=float, default=0.1)
    ap.add_argument("--readout", default="broca", choices=["linear", "broca"])
    ap.add_argument("--device", default=None, help="Dispositivo: cpu ou cuda (se None, detecta automaticamente)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"dispositivo de treino: {device}")

    texts = load_texts(args.data, args.limit)
    print(f"textos: {len(texts)}")
    tok = ChunkTokenizer(chunk_size=args.chunk_size, vocab_size=args.vocab).fit(texts)
    tok.save("chunk_tokenizer.json")
    print(f"tokenizer chunks: k={args.chunk_size} V={len(tok)}")

    data = [tok.encode(t) for t in texts]
    data = [s for s in data if len(s) > args.seq + 1]
    print(f"seqs uteis: {len(data)}")

    cfg = TRUConfig(R=(16, 24, 12), D=(32, 32, 48), obs_dim=(32, 32, 32),
                    msg_dim=16, global_dim=64, vocab_size=len(tok),
                    emb_dim=66, n_actions=8, n_symbols=args.symbols,
                    div_weight=args.div_w, readout=args.readout)
    model = TRUNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    import random
    for ep in range(args.epochs):
        random.shuffle(data)
        tot, n = 0.0, 0
        for i in range(0, len(data) - args.batch, args.batch):
            b = data[i:i + args.batch]
            x = torch.tensor([s[:args.seq] for s in b], device=device)
            y = torch.tensor([s[1:args.seq + 1] for s in b], device=device)
            logits, _, aux = model.forward_text(x, think_steps=args.think, return_aux=True)
            loss = F.cross_entropy(logits.reshape(-1, len(tok)), y.reshape(-1))
            loss = loss + args.div_w * aux["div"] + aux["sym"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(n,1):.3f}")
    torch.save(model.state_dict(), "tru_net_texto.pt")
    print("salvo: tru_net_texto.pt + chunk_tokenizer.json")


if __name__ == "__main__":
    main()
