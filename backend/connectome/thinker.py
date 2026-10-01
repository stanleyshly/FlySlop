"""Thinker model (P6a): token LM whose recurrent core is the sparse connectome, plus matched controls.

Pipeline per token:  Embedding(V, d) -> Linear(d, n_in) (low-rank input currents) -> K micro-steps of
``SparseRecurrent`` (state carried across tokens) -> DN rates -> LayerNorm -> Linear(n_DN, r) ->
Linear(r, V).  Conditioning is by sequence: ``prompt SEP target`` (edit stages: serialise
target/buffer/cursor tokens in the same stream); ``PAD`` tokens leave the state untouched so
left-padded batches work.

Kinds (``build_model(kind, cfg)``):
  real | shuffled | random_sparse : same trainable count (embedding + input + readout + one scalar per edge)
  gru                             : dense GRU, hidden size chosen so the trainable total matches ``real``
  frozen                          : real topology, ``log_mag`` randomised (deterministic by seed) then frozen;
                                    trainable = embedding + input projection + readout only (reported separately)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .circuit_io import load_circuit
from .runtime import SparseRecurrent, _spectral_radius

ROOT = Path(__file__).resolve().parents[2]
PAD, BOS, EOS = 0, 1, 2
SEP = 16                       # first reserved control id (C3 leaves 16-31 free); used only as prompt/target separator
KINDS = ("real", "shuffled", "random_sparse", "frozen", "gru")
MATCHED = ("real", "shuffled", "random_sparse", "gru")


@dataclass
class ThinkerConfig:
    circuit: str = "data/connectome/thinker_circuit_small.npz"
    vocab_size: int = 1024
    embed_dim: int = 32
    readout_rank: int = 64
    k_steps: int = 4
    backend: str = "csr"
    spectral_radius: float = 4.0     # from measure_propagation(): DN var across tokens 0.016 (4e-5 at 0.5)
    init_scale: float = 1.0
    input_scale: float = 1.0         # std of input currents at init
    tau: float = 1.0
    tbptt: int = 32                  # truncate gradient / detach state every this many tokens (0 = full BPTT)
    seed: int = 0
    sep_id: int = SEP
    gru_hidden: int | None = None    # None -> solved to match the real model's trainable count
    device: str = "auto"


def _load(cfg: ThinkerConfig):
    p = Path(cfg.circuit)
    return load_circuit(p if p.is_absolute() else ROOT / p)[0]


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Base(nn.Module):
    """Shared embedding / readout / sequence logic.  Subclasses define ``_core`` and ``init_state``."""
    kind = "base"

    def __init__(self, cfg: ThinkerConfig, n_read: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.read_norm = nn.LayerNorm(n_read)
        self.read_down = nn.Linear(n_read, cfg.readout_rank)
        self.read_up = nn.Linear(cfg.readout_rank, cfg.vocab_size)
        g = torch.Generator().manual_seed(cfg.seed + 101)      # identical embedding init across kinds
        with torch.no_grad():
            self.embed.weight.normal_(generator=g)
            for lin in (self.read_down, self.read_up):
                bound = 1 / math.sqrt(lin.in_features)
                lin.weight.uniform_(-bound, bound, generator=g)
                lin.bias.zero_()

    def readout(self, h):                         # h [..., n_read] -> logits [..., V]
        return self.read_up(self.read_down(self.read_norm(h)))

    def _core(self, emb, state):                  # -> (new_state, features)
        raise NotImplementedError

    def init_state(self, batch, device=None):
        raise NotImplementedError

    def _features(self, state):                   # state -> readout features
        return state

    def _weights(self):                           # context: cache per-sequence derived weights
        return _NullCtx()

    @staticmethod
    def _where(keep, old, new):
        return torch.where(keep[:, None], old, new)

    def _advance(self, ids, state):
        """One token: returns new state (PAD tokens keep the old state)."""
        new = self._core(self.embed(ids), state)
        return self._where(ids == PAD, state, new)

    def step(self, token_ids, state=None):
        """token_ids [B] -> (logits [B,V], state)."""
        if state is None:
            state = self.init_state(token_ids.shape[0], token_ids.device)
        state = self._advance(token_ids, state)
        return self.readout(self._features(state)), state

    def forward(self, seq, state=None, return_state=False):
        """seq [B,T] -> logits [B,T,V].  Detaches state every ``cfg.tbptt`` tokens (truncated BPTT)."""
        B, T = seq.shape
        if state is None:
            state = self.init_state(B, seq.device)
        tb, feats = self.cfg.tbptt, []
        with self._weights():
            for t in range(T):
                if tb and t and t % tb == 0:
                    state = state.detach()
                state = self._advance(seq[:, t], state)
                feats.append(self._features(state))
        logits = self.readout(torch.stack(feats, 1))
        return (logits, state) if return_state else logits

    def iter_chunks(self, seq, state=None, chunk=None):
        """Yield (t0, logits[B,c,V]) per chunk with state detached between chunks.  A trainer calling
        ``loss.backward()`` per chunk gets real TBPTT memory savings (``forward`` only truncates gradients)."""
        chunk = chunk or self.cfg.tbptt or seq.shape[1]
        B, T = seq.shape
        if state is None:
            state = self.init_state(B, seq.device)
        for t0 in range(0, T, chunk):
            logits, state = self.forward_chunk(seq[:, t0:t0 + chunk], state.detach())
            yield t0, logits

    def forward_chunk(self, seq, state):
        feats = []
        with self._weights():
            for t in range(seq.shape[1]):
                state = self._advance(seq[:, t], state)
                feats.append(self._features(state))
        return self.readout(torch.stack(feats, 1)), state

    @torch.no_grad()
    def prime(self, prompt_ids, state=None):
        """Feed prompt [B,Tp] (left-padded with PAD ok); returns (last logits [B,V], state)."""
        B = prompt_ids.shape[0]
        state = self.init_state(B, prompt_ids.device) if state is None else state
        logits = None
        with self._weights():
            for t in range(prompt_ids.shape[1]):
                logits, state = self.step(prompt_ids[:, t], state)
        return logits, state

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=64, eos=EOS, temperature=0.0, top_k=0, add_sep=True,
                 generator=None):
        """Greedy (temperature 0) or sampled continuation.  prompt_ids [B,Tp] tensor (left-pad with PAD)
        or list of lists.  Feeds ``prompt SEP`` then generates.  Returns [B, <=max_new] (PAD after eos)."""
        if not torch.is_tensor(prompt_ids):
            L = max(len(p) for p in prompt_ids)
            prompt_ids = torch.tensor([[PAD] * (L - len(p)) + list(p) for p in prompt_ids])
        dev = next(self.parameters()).device
        prompt_ids = prompt_ids.to(dev)
        if add_sep:
            prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.shape[0], 1), self.cfg.sep_id)], 1)
        logits, state = self.prime(prompt_ids)
        B = prompt_ids.shape[0]
        out = prompt_ids.new_full((B, max_new), PAD)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        with self._weights():
            return self._gen_loop(logits, state, out, done, max_new, eos, temperature, top_k, generator)

    def _gen_loop(self, logits, state, out, done, max_new, eos, temperature, top_k, generator):
        for i in range(max_new):
            if temperature and temperature > 0:
                lg = logits / temperature
                if top_k:
                    kth = lg.topk(top_k, -1).values[:, -1:]
                    lg = lg.masked_fill(lg < kth, -float("inf"))
                nxt = torch.multinomial(lg.softmax(-1), 1, generator=generator)[:, 0]
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            out[:, i] = nxt
            done = done | (nxt == eos)
            if bool(done.all()):
                break
            logits, state = self.step(nxt, state)
        return out

    # ------------------------------------------------------------------ bookkeeping
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_total(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ThinkerConnectome(_Base):
    def __init__(self, cfg: ThinkerConfig, variant: str = "real", circuit=None):
        circuit = _load(cfg) if circuit is None else circuit
        core = SparseRecurrent(circuit, backend=cfg.backend, k_steps=cfg.k_steps, variant=variant,
                               seed=cfg.seed, init_scale=cfg.init_scale, tau=cfg.tau,
                               input_idx=circuit["idx_VP"], spectral_radius=cfg.spectral_radius,
                               device=cfg.device)
        super().__init__(cfg, int(core.idx_DN.numel()))
        self.kind = variant
        self._w = None
        self.core = core
        self.in_proj = nn.Linear(cfg.embed_dim, core.n_in, bias=False)
        nn.init.normal_(self.in_proj.weight, std=cfg.input_scale / math.sqrt(cfg.embed_dim))
        if variant == "frozen":
            self._randomise_frozen()
        self.to(cfg.device)

    def _randomise_frozen(self):
        """Random magnitudes (same mean/std of log_mag as the real init, deterministic in seed), then
        re-normalise to the configured spectral radius and freeze."""
        c, core = self.cfg, self.core
        with torch.no_grad():
            g = torch.Generator().manual_seed(c.seed + 7919)
            lm = core.log_mag
            core.log_mag.copy_(lm.mean() + lm.std() * torch.randn(lm.shape, generator=g))
            if c.spectral_radius:
                n, w = core.n, core.edge_weights().double()
                mv = lambda v: torch.zeros(n, 1, dtype=torch.float64).index_add_(0, core.post, w[:, None] * v[core.pre])
                rho = _spectral_radius(mv, n, seed=c.seed)
                if rho > 0:
                    core.log_mag += math.log(c.spectral_radius / rho)
        core.log_mag.requires_grad_(False)

    def init_state(self, batch, device=None):
        return torch.zeros(batch, self.core.n, device=device or self.core.log_mag.device)

    def _weights(self):
        outer = self

        class _Ctx:
            def __enter__(c):
                c.prev = outer._w
                if c.prev is None:
                    w = outer.core.edge_weights()
                    outer._w = (w, outer.core.dense_weight() if outer.core.backend == "dense" else None)
                return c

            def __exit__(c, *a):
                outer._w = c.prev
                return False
        return _Ctx()

    def _core(self, emb, state):
        core, cur = self.core, self.in_proj(emb)
        if self._w is not None:
            w, dense = self._w
        else:
            w = core.edge_weights()
            dense = core.dense_weight() if core.backend == "dense" else None
        x = cur.new_zeros(cur.shape[0], core.n)
        x[:, core.input_idx] = cur
        r, a = state, core.alpha
        for _ in range(core.k_steps):        # same dynamics as SparseRecurrent.forward, weights hoisted
            act = core.nonlinearity(core._matvec(w, r, dense) + x)
            r = act if a == 1.0 else (1.0 - a) * r + a * act
        return r

    def _features(self, state):
        return state[..., self.core.idx_DN]


class ThinkerGRU(_Base):
    kind = "gru"

    def __init__(self, cfg: ThinkerConfig, hidden: int):
        super().__init__(cfg, hidden)
        self.hidden = hidden
        self.cell = nn.GRUCell(cfg.embed_dim, hidden)
        self.to(cfg.device)

    def init_state(self, batch, device=None):
        return torch.zeros(batch, self.hidden, device=device or self.cell.weight_ih.device)

    def _core(self, emb, state):
        return self.cell(emb, state)


def _gru_params(cfg: ThinkerConfig, h: int) -> int:
    d, V, r = cfg.embed_dim, cfg.vocab_size, cfg.readout_rank
    return V * d + 3 * (h * d + h * h + 2 * h) + 2 * h + h * r + r + r * V + V


def solve_gru_hidden(cfg: ThinkerConfig, target: int) -> int:
    lo, hi = 1, 4096
    while lo < hi:                       # smallest h with params >= target, then pick nearer neighbour
        mid = (lo + hi) // 2
        if _gru_params(cfg, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if abs(_gru_params(cfg, lo) - target) <= abs(_gru_params(cfg, lo - 1) - target) else lo - 1


def build_model(kind: str, cfg: ThinkerConfig | None = None) -> _Base:
    cfg = cfg or ThinkerConfig()
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if cfg.device == "auto":
        from .device import resolve_device
        cfg = replace(cfg, device=resolve_device("auto"))
    torch.manual_seed(cfg.seed)          # same seed => identical embedding / input / readout init across kinds
    if kind == "gru":
        h = cfg.gru_hidden
        if h is None:
            ref = build_model("real", replace(cfg, device="cpu"))
            h = solve_gru_hidden(cfg, ref.n_trainable())
        return ThinkerGRU(cfg, h)
    return ThinkerConnectome(cfg, variant=kind)


def param_report(model: _Base) -> dict:
    groups = {}
    for name, p in model.named_parameters():
        g = name.split(".")[0]
        d = groups.setdefault(g, {"trainable": 0, "frozen": 0})
        d["trainable" if p.requires_grad else "frozen"] += p.numel()
    tr = sum(d["trainable"] for d in groups.values())
    fr = sum(d["frozen"] for d in groups.values())
    return {"kind": model.kind, "trainable": tr, "frozen": fr, "total": tr + fr, "groups": groups,
            **({"hidden": model.hidden} if isinstance(model, ThinkerGRU) else {})}


def measure_propagation(cfg: ThinkerConfig, n_tokens: int = 64, seed: int = 0) -> dict:
    """Feed distinct random tokens (single step from zero state) and report DN activity statistics.
    ``dn_var_across_tokens``: variance over tokens of each DN's rate, averaged over DNs (signal);
    ``dn_rms``: mean rate magnitude; ``frac_dn_active``: DNs with |rate| > 1e-3 for some token."""
    m = build_model("real", cfg)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(16, cfg.vocab_size, (n_tokens,), generator=g)
    with torch.no_grad():
        st = m._advance(ids, m.init_state(n_tokens))
        dn = m._features(st)
        st2 = st
        for _ in range(3):                                   # short continuation with a different token stream
            st2 = m._advance(torch.randint(16, cfg.vocab_size, (n_tokens,), generator=g), st2)
        dn3 = m._features(st2)
    return {"spectral_radius": cfg.spectral_radius, "input_scale": cfg.input_scale,
            "dn_var_across_tokens": float(dn.var(0).mean()), "dn_rms": float(dn.pow(2).mean().sqrt()),
            "frac_dn_active": float((dn.abs().max(0).values > 1e-3).float().mean()),
            "all_rms": float(st.pow(2).mean().sqrt()), "dn_var_after_4_tokens": float(dn3.var(0).mean()),
            "all_rms_after_4_tokens": float(st2.pow(2).mean().sqrt())}


def quick_overfit(model: _Base, seq: torch.Tensor, steps: int = 60, lr: float = 1e-2, conn_lr: float = 3e-3,
                  max_seconds: float | None = None, target_acc: float = 0.95, log_every: int = 5):
    """Teacher-forced memorisation of ``seq`` [B,T] (loss on positions 1..T-1).  Returns curve
    [(step, loss, acc, seconds)].  Stops early once ``target_acc`` is reached or ``max_seconds`` passes."""
    import time
    core = [p for n, p in model.named_parameters() if n.startswith(("core.", "cell.")) and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("core.", "cell.")) and p.requires_grad]
    opt = torch.optim.Adam([{"params": rest, "lr": lr}, {"params": core, "lr": conn_lr}])
    x, y = seq[:, :-1], seq[:, 1:]
    curve, t0 = [], time.time()
    for s in range(1, steps + 1):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        acc = float((logits.argmax(-1) == y).float().mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if s % log_every == 0 or s == 1:
            curve.append((s, float(loss.detach()), acc, time.time() - t0))
        if acc >= target_acc or (max_seconds and time.time() - t0 > max_seconds):
            curve.append((s, float(loss.detach()), acc, time.time() - t0))
            break
    return curve
