"""TRU-Net v2+v3: quebra de simetria + memoria episodica + readout de Broca.

- Disjoint slicing: e = [e_A|e_B|e_C], cada modulo ve so sua fatia.
- Papeis: A=hipotese (atual, dropout leve), B=memoria (media movel lenta),
  C=exploracao (dropout agressivo + ruido).
- G como neuromodulacao (dentro de states.py), nao injecao.
- M: memoria episodica entre batches (nivel M da teoria).
- Readout: Broca (GRU de superficie + skip de e_t + cross pooled(S));
  cortex associa / workspace modula, nao emite palavra direta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .states import ViscousRNNModule, diversity_loss
from .channel import LimitedChannel, symbol_loss
from .integrator import CommonSpace
from .readout import BrocaReadout


class EpisodicMemory(nn.Module):
    """Nivel M: fila circular de vetores globais medios (sobrevive ao episodio)."""
    def __init__(self, slots=32, dim=64, momentum=0.9):
        super().__init__()
        self.slots = int(slots)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.register_buffer("M", torch.zeros(self.slots, self.dim))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros(()))
        self.read_proj = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.read_proj.weight)
        nn.init.zeros_(self.read_proj.bias)

    @torch.no_grad()
    def update(self, g_mean):
        # g_mean: [dim] media do batch atual
        i = int(self.ptr.item()) % self.slots
        self.M.data[i] = self.momentum * self.M.data[i] + (1.0 - self.momentum) * g_mean.data
        self.ptr.fill_((i + 1) % self.slots)
        self.filled.fill_(1.0)

    def read(self, G):
        if not bool(self.filled.item()):
            return G, None
        M = self.M.detach()
        scores = (G @ M.t()) / (self.dim ** 0.5)   # [B,slots]
        w = torch.softmax(scores, dim=-1)
        r = w @ M                                   # [B,dim]
        return G + self.read_proj(r), w


from dataclasses import dataclass


@dataclass
class TRUConfig:
    R: tuple = (16, 24, 12)
    D: tuple = (32, 32, 48)
    obs_dim: tuple = (32, 32, 32)
    msg_dim: int = 16
    global_dim: int = 64
    vocab_size: int = 2000
    emb_dim: int = 66          # divisivel por 3 -> fatias [22,22,22]
    n_actions: int = 8
    n_symbols: int = 0
    alpha_bands: tuple = (0.1, 0.5, 0.9, 0.98)
    names: tuple = ("hipotese", "memoria", "exploracao")
    mem_slots: int = 32
    div_weight: float = 0.1
    sym_beta: float = 0.3
    dropout_a: float = 0.1     # papel A: leve
    dropout_c: float = 0.5     # papel C: agressivo
    noise_c: float = 0.1
    readout: str = "broca"     # "linear" (legado) | "broca" (GRU + skip + cross)
    readout_hidden: int = 128


class TRUNet(nn.Module):
    def __init__(self, cfg: TRUConfig):
        super().__init__()
        self.cfg = cfg
        n = len(cfg.R)
        assert len(cfg.D) == n and len(cfg.obs_dim) == n and len(cfg.names) == n
        self.n_mod = n
        # fatias disjuntas da embedding
        base, rem = divmod(cfg.emb_dim, n)
        self.slice_dims = [base + (1 if i < rem else 0) for i in range(n)]
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.view = nn.ModuleList([nn.Linear(self.slice_dims[i], cfg.obs_dim[i]) for i in range(n)])
        self.states = nn.ModuleList([
            ViscousRNNModule(cfg.names[i], cfg.R[i], cfg.D[i], cfg.obs_dim[i],
                             cfg.global_dim, alphas="bands", alpha_bands=cfg.alpha_bands)
            for i in range(n)
        ])
        self.channels = nn.ModuleList([
            LimitedChannel(cfg.R[i], cfg.D[i], cfg.msg_dim, n_symbols=cfg.n_symbols)
            for i in range(n)
        ])
        self.common = CommonSpace(cfg.msg_dim, n, cfg.global_dim)
        self.memory = EpisodicMemory(cfg.mem_slots, cfg.global_dim)
        self.action_head = nn.Linear(cfg.global_dim, cfg.n_actions)
        self.text_head = nn.Linear(cfg.global_dim, cfg.vocab_size)  # modo linear legado
        self.pooled_dim = int(sum(cfg.D))
        self.readout = None
        if cfg.readout == "broca":
            self.readout = BrocaReadout(cfg.emb_dim, cfg.global_dim,
                                        self.pooled_dim, cfg.readout_hidden,
                                        cfg.vocab_size)

    def split_views(self, e):
        # e:[B,E] -> lista de fatias disjuntas
        return list(torch.split(e, self.slice_dims, dim=-1))

    def role_obs(self, slices, ema_b=None, train=True):
        """Aplica papeis cognitivos distintos. Retorna (obs, novo_ema_b)."""
        sa, sb, sc = slices
        # A: hipotese rapida, dropout leve
        oa = torch.tanh(self.view[0](F.dropout(sa, p=self.cfg.dropout_a, training=train)))
        # B: memoria lenta, media movel
        if ema_b is None:
            ema_b = sb
        else:
            ema_b = 0.7 * ema_b + 0.3 * sb
        ob = torch.tanh(self.view[1](ema_b))
        # C: exploracao, dropout agressivo + ruido
        sc_d = F.dropout(sc, p=self.cfg.dropout_c, training=train)
        if train and self.cfg.noise_c > 0:
            sc_d = sc_d + torch.randn_like(sc_d) * self.cfg.noise_c
        oc = torch.tanh(self.view[2](sc_d))
        return [oa, ob, oc], ema_b

    def init_cog(self, B, device):
        S = [m.init_state(B, device=device) for m in self.states]
        G = self.common.init_state(B, device=device)
        return S, G

    def cog_step(self, obs, S, G, noise_p=None):
        msgs, infos = [], []
        for i in range(self.n_mod):
            m, info = self.channels[i](S[i], noise_p=noise_p)
            msgs.append(m)
            infos.append(info)
        G2 = self.common(msgs, G)
        S2 = [self.states[i](obs[i], G2, S[i]) for i in range(self.n_mod)]
        return S2, G2, msgs, infos

    def aux_losses(self, S, infos, msgs=None):
        # diversidade sobre mensagens (mesmo dim) — estados tem D distintos
        ref = msgs[0] if msgs is not None else self.states[0].pooled(S[0])
        ldiv = diversity_loss(msgs) if msgs is not None else ref.new_zeros(())
        lsym, Hb, Hc = ref.new_zeros(()), None, None
        nlg = 0
        for info in infos:
            if info.get("logits") is not None:
                l, hb, hc = symbol_loss(info["logits"], beta=self.cfg.sym_beta)
                lsym = lsym + l
                Hb, Hc = hb, hc
                nlg += 1
        if nlg:
            lsym = lsym / nlg
        pooled = [self.states[i].pooled(S[i]) for i in range(self.n_mod)]
        return {"div": ldiv, "sym": lsym, "H_bar": Hb, "H_cond": Hc,
                "pooled": pooled}

    def reason(self, obs, steps=4, return_aux=False):
        B = obs[0].shape[0]
        S, G = self.init_cog(B, obs[0].device)
        hist, last_infos, last_msgs = [], None, None
        for _ in range(steps):
            S, G, msgs, infos = self.cog_step(obs, S, G)
            hist.append([m.detach() for m in msgs])
            last_infos, last_msgs = infos, msgs
        if return_aux:
            return G, S, hist, self.aux_losses(S, last_infos, last_msgs)
        return G, S, hist

    def emit(self, e_t, Gm, S, h_prev):
        """Um passo do vocalizador. Retorna (logits, h_new)."""
        if self.readout is None:
            return self.text_head(Gm), h_prev
        pooled = torch.cat([self.states[i].pooled(S[i]) for i in range(self.n_mod)], dim=-1)
        return self.readout(e_t, Gm, pooled, h_prev)

    def init_readout(self, B, device):
        if self.readout is None:
            return None
        return self.readout.init_state(B, device=device)

    def forward_text(self, ids, think_steps=2, return_aux=False, update_memory=True):
        B, T = ids.shape
        train = self.training
        S, G = self.init_cog(B, ids.device)
        h = self.init_readout(B, ids.device)
        logits, ema_b, last_infos, last_msgs = [], None, None, None
        for t in range(T):
            e = self.embedding(ids[:, t])
            slices = self.split_views(e)
            obs, ema_b = self.role_obs(slices, ema_b, train=train)
            for _ in range(think_steps):
                S, G, msgs, infos = self.cog_step(obs, S, G)
                last_infos, last_msgs = infos, msgs
            Gm, _ = self.memory.read(G)
            lg, h = self.emit(e, Gm, S, h)
            logits.append(lg)
        out = torch.stack(logits, dim=1)
        if update_memory and not train:
            pass
        if train and update_memory:
            self.memory.update(G.detach().mean(dim=0))
        elif not train and update_memory:
            self.memory.update(G.detach().mean(dim=0))
        if return_aux:
            return out, (S, G), self.aux_losses(S, last_infos, last_msgs)
        return out, (S, G)

    @torch.no_grad()
    def generate(self, prefix_ids, steps=16, think_steps=2):
        self.eval()
        ids = prefix_ids.tolist()[0] if prefix_ids.dim() == 2 else list(prefix_ids)
        cur = torch.tensor([ids], device=prefix_ids.device)
        S, G = self.init_cog(1, prefix_ids.device)
        h = self.init_readout(1, prefix_ids.device)
        ema_b = None
        for t in range(cur.shape[1]):
            e = self.embedding(cur[:, t])
            obs, ema_b = self.role_obs(self.split_views(e), ema_b, train=False)
            for _ in range(think_steps):
                S, G, _, _ = self.cog_step(obs, S, G)
            Gm, _ = self.memory.read(G)
            _, h = self.emit(e, Gm, S, h)  # aquece Broca com o prefixo
        out = list(ids)
        last = cur[:, -1:]
        for _ in range(steps):
            e = self.embedding(last.squeeze(1))
            obs, ema_b = self.role_obs(self.split_views(e), ema_b, train=False)
            for _ in range(think_steps):
                S, G, _, _ = self.cog_step(obs, S, G)
            Gm, _ = self.memory.read(G)
            lg, h = self.emit(e, Gm, S, h)
            nxt = lg.argmax(dim=-1, keepdim=True)
            out.append(int(nxt.item()))
            last = nxt
        return out
