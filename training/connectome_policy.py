"""Connectome-constrained actor for SB3 PPO/imitation (PLAN §5.1 step 5).

The actor is a rate network over the neurons of the MaleCNS right-foreleg
circuit (``backend.connectome``). Connectivity sparsity and sign are taken
from the chosen wiring variant; only synapse magnitudes, biases, the input
encoders, and the linear action readout are learned::

    r_{t+1} = tanh(W r_t + E_dn(task obs) + E_sn(proprio obs) + b),  t < steps
    latent  = r_T[DN ∪ MN]      ->  linear readout to the 5 actions

Descending neurons (DN) receive the task/visual observations (next key
offset, progress, ...); right-foreleg sensory neurons (SN) receive the
proprioceptive/tactile ones (claw height/velocity, key depression, holding,
body velocity). That observation-to-cell assignment is a modelling assumption,
not a measured mapping. ``silence`` zeroes named observation groups for the
silenced-sensory ablation. The critic is an ordinary MLP in every variant.

This module holds two policies:

* ``ConnectomePolicy`` / ``ConnectomeActor`` (legacy, 360 neurons, dense masked matrix, used by
  ``training.connectome_experiment`` via ``common.make_model``; unchanged).
* ``ConnectomeActorCriticPolicy`` (P2b, new): the actor is ``backend.connectome.runtime.SparseRecurrent``
  over the 3173-neuron typing circuit (``data/connectome/typing_circuit.npz``) or, through
  ``legacy_circuit_arrays``, the legacy 360-neuron json on the dense backend. See its docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

from backend.connectome import group_indices, load_circuit, variant_matrix
from backend.connectome.circuit_io import content_hash, load_circuit as load_circuit_npz
from backend.connectome.device import resolve_device
from backend.connectome.runtime import POPULATIONS, SparseRecurrent
from backend.fly_env import FlyTypingEnv

PROPRIO = ("tip_z", "tip_vz", "key_depression", "rf_holding", "lf_tip_z", "lf_tip_vz", "lf_holding", "body_vx", "body_vy")


def labels_for_dim(obs_dim: int) -> tuple[str, ...]:
    """Observation labels of ``FlyTypingEnv`` for an obs size (33 legacy, + key one-hot, + editor fields)."""
    from backend.fly_env import obs_labels
    for key_obs in (False, True):
        for editor_obs in (False, True):
            if len(obs_labels(key_obs, editor_obs)) == obs_dim:
                return obs_labels(key_obs, editor_obs)
    raise ValueError(f"no FlyTypingEnv observation layout has {obs_dim} entries")


def obs_groups(labels=FlyTypingEnv.OBS_LABELS) -> tuple[list[int], list[int]]:
    proprio = [i for i, name in enumerate(labels) if name in PROPRIO]
    task = [i for i in range(len(labels)) if i not in proprio]
    return task, proprio


class ConnectomeActor(nn.Module):
    def __init__(self, obs_dim: int, mode: str = "connectome", seed: int = 0, steps: int = 3,
                 silence: tuple[str, ...] = (), circuit_path: str | None = None):
        super().__init__()
        circuit = load_circuit(circuit_path) if circuit_path else load_circuit()
        idx = group_indices(circuit)
        w = torch.as_tensor(variant_matrix(circuit, mode, seed))
        self.mode, self.steps, self.n = mode, int(steps), w.shape[0]
        self.task_idx, self.proprio_idx = obs_groups()
        if obs_dim != len(FlyTypingEnv.OBS_LABELS):
            raise ValueError("ConnectomeActor expects the FlyTypingEnv observation")
        mask = (w != 0).float()
        self.register_buffer("mask", mask)
        self.register_buffer("sign", torch.sign(w))
        row = w.abs().sum(dim=1, keepdim=True) + 1.0
        init = (w.abs() / row * 1.5).clamp_min(1e-4)
        if mode == "dense":   # no sign/wiring prior: free signed weights on allowed blocks
            self.weight = nn.Parameter(w / row * 1.5)
        else:
            self.log_mag = nn.Parameter(torch.log(init) * mask)
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.register_buffer("dn", torch.as_tensor(idx["DN"]))
        self.register_buffer("sn", torch.as_tensor(idx["SN"]))
        self.register_buffer("out_idx", torch.as_tensor(np.concatenate([idx["DN"], idx["MN"]])))
        self.enc_dn = nn.Linear(len(self.task_idx), len(idx["DN"]))
        self.enc_sn = nn.Linear(len(self.proprio_idx), len(idx["SN"]))
        silenced = set()
        for name in silence:
            if name == "proprio":
                silenced |= set(self.proprio_idx)
            elif name in FlyTypingEnv.OBS_LABELS:
                silenced.add(FlyTypingEnv.OBS_LABELS.index(name))
            else:
                raise ValueError(f"unknown observation group to silence: {name}")
        keep = torch.ones(obs_dim)
        keep[list(silenced)] = 0.0
        self.register_buffer("keep", keep)
        self.latent_dim = len(self.out_idx)
        self.last_activity: torch.Tensor | None = None

    def effective_weight(self) -> torch.Tensor:
        if self.mode == "dense":
            return self.weight * self.mask
        return self.sign * torch.exp(self.log_mag) * self.mask

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs * self.keep
        drive = torch.zeros(obs.shape[0], self.n, device=obs.device, dtype=obs.dtype)
        drive[:, self.dn] = self.enc_dn(obs[:, self.task_idx])
        drive[:, self.sn] = self.enc_sn(obs[:, self.proprio_idx])
        w = self.effective_weight()
        r = torch.tanh(drive + self.bias)
        for _ in range(self.steps):
            r = torch.tanh(r @ w.T + drive + self.bias)
        self.last_activity = r.detach()
        return r[:, self.out_idx]


class ConnectomeExtractor(nn.Module):
    def __init__(self, obs_dim: int, actor_kwargs: dict, critic_arch=(256, 256)):
        super().__init__()
        self.actor = ConnectomeActor(obs_dim, **actor_kwargs)
        layers, last = [], obs_dim
        for width in critic_arch:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.critic = nn.Sequential(*layers)
        self.latent_dim_pi, self.latent_dim_vf = self.actor.latent_dim, last

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.actor(features)

    def forward_critic(self, features):
        return self.critic(features)


class ConnectomePolicy(ActorCriticPolicy):
    """``policy_kwargs={"actor_kwargs": {"mode": ..., "seed": ..., "silence": [...]}}``."""

    def __init__(self, *args, actor_kwargs: dict | None = None, critic_arch=(256, 256), **kwargs):
        self.actor_kwargs = dict(actor_kwargs or {})
        self.critic_arch = tuple(critic_arch)
        kwargs.pop("net_arch", None)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ConnectomeExtractor(self.features_dim, self.actor_kwargs, self.critic_arch)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(actor_kwargs=self.actor_kwargs, critic_arch=self.critic_arch)
        return data


# ============================================================================ P2b: runtime-based policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIRCUIT = "data/connectome/typing_circuit.npz"
DEFAULT_CONNECTOME_KWARGS = {
    "circuit": DEFAULT_CIRCUIT, "variant": "real", "backend": "auto", "device": "auto", "k_steps": 4,
    "init_scale": 2.0, "spectral_radius": None, "tau": 1.0, "encoder": "split", "input_gain": 4.0,
    "readout": "mn_dn", "seed": 0, "silence": [],
}


def legacy_circuit_arrays(circuit: dict) -> dict[str, np.ndarray]:
    """Adapt the legacy ``rf_leg_circuit.json`` dict (list edges ``[pre, post, count, sign]``) to the
    array format ``SparseRecurrent`` reads. Use with ``backend="dense"``."""
    n = len(circuit["neurons"])
    edges = sorted(circuit["edges"], key=lambda e: (e[1], e[0]))
    pre = np.array([e[0] for e in edges], np.int32)
    post = np.array([e[1] for e in edges], np.int64)
    out = {"pre": pre, "weight": np.array([e[2] for e in edges], np.int32),
           "edge_sign": np.array([e[3] for e in edges], np.int8),
           "indptr": np.concatenate([[0], np.cumsum(np.bincount(post, minlength=n))]).astype(np.int64)}
    groups = [x["group"] for x in circuit["neurons"]]
    for g in POPULATIONS:
        out[f"idx_{g}"] = np.array([i for i, x in enumerate(groups) if x == g], np.int32)
    return out


def resolve_circuit(circuit: str | Path) -> tuple[dict[str, np.ndarray], str]:
    """(arrays, content hash) for an ``.npz`` typing circuit or a legacy ``.json`` circuit."""
    path = Path(circuit)
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix == ".json":
        arrays = legacy_circuit_arrays(load_circuit(path))
        return arrays, content_hash(arrays)
    arrays, _meta, stored = load_circuit_npz(path)
    return arrays, stored or content_hash(arrays)


class Readout(nn.Module):
    """Maps runtime rates ``[B, n]`` to the features that SB3's ``action_net`` (linear, ``out_dim`` -> 8) reads.

    P3 registers a MN -> joint antagonist readout with ``register_readout`` and selects it with
    ``connectome_kwargs={"readout": "<name>"}``. ``factory(runtime) -> Readout``.
    """
    out_dim: int

    def forward(self, runtime: SparseRecurrent, rates: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PopulationReadout(Readout):
    """Concatenated rates of named populations; ``action_net`` on top is the labelled linear readout."""

    def __init__(self, runtime: SparseRecurrent, populations=("MN", "DN")):
        super().__init__()
        self.populations = tuple(populations)
        self.out_dim = sum(int(getattr(runtime, f"idx_{p}").numel()) for p in self.populations)
        if self.out_dim == 0:
            raise ValueError(f"circuit has no neurons in {self.populations}")

    def forward(self, runtime, rates):
        return torch.cat([runtime.rates(p, rates) for p in self.populations], dim=1)


READOUTS = {"mn_dn": lambda rt: PopulationReadout(rt, ("MN", "DN")),
            "mn": lambda rt: PopulationReadout(rt, ("MN",))}


def register_readout(name: str, factory) -> None:
    READOUTS[name] = factory


class RuntimeActor(nn.Module):
    """obs -> learned encoder -> input currents on DN+SN -> ``SparseRecurrent`` (K micro-steps) -> readout features.

    Stateless per env step: the recurrent state restarts at zero on every call (no carry-over), so the
    action is a pure function of the observation and PPO minibatches are exact.
    ``encoder="split"``: task obs -> DN currents, proprio obs -> SN currents (as the legacy actor);
    ``"full"``: every obs feeds every input neuron. ``silence`` zeroes obs groups (ablation).
    """

    def __init__(self, obs_dim: int, circuit=DEFAULT_CIRCUIT, variant="real", backend="auto", device="auto",
                 k_steps=4, init_scale=2.0, spectral_radius=None, tau=1.0, encoder="split", input_gain=4.0,
                 readout="mn_dn", seed=0, silence=()):
        super().__init__()
        arrays, self.circuit_hash = resolve_circuit(circuit)
        n = int(arrays["indptr"].shape[0] - 1)
        self.device_name = resolve_device(device, n)
        legacy = str(circuit).endswith(".json")
        if backend == "auto" and legacy:
            backend = "dense"
        input_idx = np.concatenate([arrays["idx_DN"], arrays["idx_SN"]]).astype(np.int64)
        self.runtime = SparseRecurrent(arrays, backend=backend, k_steps=k_steps, variant=variant, seed=seed,
                                       init_scale=init_scale, tau=tau, input_idx=input_idx,
                                       spectral_radius=spectral_radius, device=self.device_name)
        n_dn, n_in = int(arrays["idx_DN"].size), len(input_idx)
        self.input_gain = float(input_gain)
        self.enc = nn.Linear(obs_dim, n_in)
        mask = torch.ones(n_in, obs_dim)
        if encoder == "split":
            task, proprio = obs_groups(labels_for_dim(obs_dim))
            mask[:n_dn, proprio] = 0.0
            mask[n_dn:, task] = 0.0
        elif encoder != "full":
            raise ValueError("encoder must be 'split' or 'full'")
        self.register_buffer("enc_mask", mask)
        silenced = set()
        for name in silence:
            if name == "proprio":
                silenced |= set(obs_groups()[1])
            elif name in FlyTypingEnv.OBS_LABELS:
                silenced.add(FlyTypingEnv.OBS_LABELS.index(name))
            else:
                raise ValueError(f"unknown observation group to silence: {name}")
        keep = torch.ones(obs_dim)
        keep[list(silenced)] = 0.0
        self.register_buffer("keep", keep)
        self.runtime.circuit_arrays = arrays     # read by readouts that need MN types (mn_antagonist)
        self.readout = READOUTS[readout](self.runtime)
        self.latent_dim = self.readout.out_dim
        self.last_activity: torch.Tensor | None = None
        self.reset_encoder()

    def reset_encoder(self) -> None:
        """Init so that unit obs gives O(input_gain) currents (row-normalised runtime attenuates otherwise)."""
        with torch.no_grad():
            fan_in = self.enc_mask.sum(1, keepdim=True).clamp_min(1.0)
            self.enc.weight.copy_(torch.randn_like(self.enc.weight) / fan_in.sqrt() * self.input_gain)
            self.enc.bias.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs * self.keep
        currents = nn.functional.linear(obs, self.enc.weight * self.enc_mask, self.enc.bias)
        _state, rates = self.runtime(currents.to(self.runtime.log_mag.device))
        self.last_activity = rates.detach()
        return self.readout(self.runtime, rates).to(obs.device)


class RuntimeExtractor(nn.Module):
    def __init__(self, obs_dim: int, actor_kwargs: dict, critic_arch=(256, 256)):
        super().__init__()
        self.actor = RuntimeActor(obs_dim, **actor_kwargs)
        layers, last = [], obs_dim
        for width in critic_arch:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.critic = nn.Sequential(*layers)
        self.latent_dim_pi, self.latent_dim_vf = self.actor.latent_dim, last

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.actor(features)

    def forward_critic(self, features):
        return self.critic(features)


class ConnectomeActorCriticPolicy(ActorCriticPolicy):
    """SB3 policy with a ``SparseRecurrent`` connectome actor and an MLP critic.

    ``policy_kwargs={"connectome_kwargs": {...}, "critic_arch": (256, 256)}``; keys of
    ``connectome_kwargs`` are ``DEFAULT_CONNECTOME_KWARGS`` (circuit path, variant real|shuffled|
    random_sparse|frozen, backend, device auto|cpu|mps, k_steps, init_scale, spectral_radius, encoder,
    readout, silence). The action is ``action_net(readout(runtime(encoder(obs))))``: with the default
    readout, a linear map from MN+DN rates to the 8-D action mean. All kwargs are stored in
    ``_get_constructor_parameters`` so ``PPO.load`` rebuilds the same architecture; the wiring is
    re-derived from the circuit file + seed and the state_dict then restores learned weights.
    A checkpoint trained on mps keeps its recorded ``device``/backend unless overridden with
    ``load_ppo(..., device=...)`` in ``training.train_ppo``.
    """

    def __init__(self, *args, connectome_kwargs: dict | None = None, critic_arch=(256, 256), **kwargs):
        unknown = set(connectome_kwargs or {}) - set(DEFAULT_CONNECTOME_KWARGS)
        if unknown:
            raise ValueError(f"unknown connectome_kwargs: {sorted(unknown)}")
        self.connectome_kwargs = {**DEFAULT_CONNECTOME_KWARGS, **(connectome_kwargs or {})}
        self.connectome_kwargs["silence"] = list(self.connectome_kwargs["silence"])
        self.critic_arch = tuple(critic_arch)
        kwargs.pop("net_arch", None)
        super().__init__(*args, **kwargs)
        self.mlp_extractor.actor.reset_encoder()   # undo SB3's orthogonal init of the encoder

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        readout = self.mlp_extractor.actor.readout
        if getattr(readout, "identity_action", False):
            # The readout outputs the env action itself (P3 mn_antagonist): no free linear action_net on top.
            if readout.out_dim != self.action_space.shape[0]:
                raise ValueError(f"readout has {readout.out_dim} outputs, action space is {self.action_space.shape}")
            self.action_net = nn.Identity()
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RuntimeExtractor(self.features_dim, self.connectome_kwargs, self.critic_arch)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(connectome_kwargs=self.connectome_kwargs, critic_arch=self.critic_arch)
        return data

    @property
    def actor(self) -> RuntimeActor:
        return self.mlp_extractor.actor


from training import mn_readout as _mn_readout  # noqa: E402,F401  (registers the "mn_antagonist" readout)
