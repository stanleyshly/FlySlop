"""LLM teacher for stage-5 data generation (runs alone; see docs/architecture.md, RAM cap).

    uv run --extra teacher python -m training.teacher --prompt "Write a Python function double(x)."

Backends: ``auto`` selects MLX on macOS and Torch elsewhere (CUDA when available, CPU otherwise);
``mlx``, ``torch``, ``llamacpp`` (llama-cli subprocess) and ``ollama`` (HTTP,
localhost:11434); the latter two are thin and untested here. Models load lazily on first
``generate``. Importing this module never imports mlx or loads weights.

Checkpoints:
  TEACHER_MODEL   mlx-community/gemma-4-e2b-it-4bit   (Gemma 4 E2B-it, 4-bit MLX, ~3.6 GB on disk)
  STANDIN_MODEL   mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit  (STAND-IN for smoke tests only,
                  it is NOT the teacher and its outputs must not be used as teacher data)
Memory: refuses to load if the estimated footprint exceeds the RAM cap (backend.memory_budget).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import memory_budget as mb  # noqa: E402

TEACHER_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
STANDIN_MODEL = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
TORCH_TEACHER_MODEL = "google/gemma-4-E2B-it"
TORCH_STANDIN_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
BACKENDS = ("auto", "mlx", "torch", "llamacpp", "ollama")
LOAD_OVERHEAD = 1.15          # weights x this + activations/KV
LOAD_FIXED_BYTES = int(0.4 * mb.GB)


# Measured on the M1 (mx.get_peak_memory, 256-token generation) + 0.2 GB margin. Disk size (3.6 GB)
# overstates the footprint because the vision/audio towers are never touched for text prompts.
MEASURED_BYTES = {
    TEACHER_MODEL: int(2.7 * mb.GB), STANDIN_MODEL: int(0.6 * mb.GB),
    TORCH_TEACHER_MODEL: int(11.8 * mb.GB), TORCH_STANDIN_MODEL: int(1.4 * mb.GB),
}


def estimate_model_bytes(model_id: str) -> int:
    """Estimated resident footprint: measured table, else local dir / Hub safetensors size x overhead."""
    if model_id in MEASURED_BYTES:
        return MEASURED_BYTES[model_id]
    p = Path(model_id)
    if p.is_dir():
        weights = sum(f.stat().st_size for f in p.glob("*.safetensors")) or sum(f.stat().st_size for f in p.glob("*.gguf"))
    else:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, files_metadata=True)
        weights = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith((".safetensors", ".gguf")))
    return int(weights * LOAD_OVERHEAD + LOAD_FIXED_BYTES)


class Teacher:
    def __init__(self, model_id: str = TEACHER_MODEL, backend: str = "auto", max_tokens: int = 256,
                 temperature: float = 0.0, seed: int = 0, max_ram_gb: float | None = None,
                 system: str | None = None, skip_estimate: bool = False, thinking: bool = False):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = self._auto_backend() if backend == "auto" else backend
        self.model_id = model_id
        self._runtime_model_id = model_id
        if self.backend == "torch" and model_id == TEACHER_MODEL:
            self._runtime_model_id = TORCH_TEACHER_MODEL
        elif self.backend == "torch" and model_id == STANDIN_MODEL:
            self._runtime_model_id = TORCH_STANDIN_MODEL
        self.max_tokens, self.temperature, self.seed = int(max_tokens), float(temperature), int(seed)
        self.max_ram_gb, self.system, self.skip_estimate, self.thinking = max_ram_gb, system, skip_estimate, thinking
        self._model = self._tokenizer = None
        self.device = "mlx" if self.backend == "mlx" else None
        self.last_stats: dict = {}

    @staticmethod
    def _auto_backend() -> str:
        if sys.platform == "darwin" and importlib.util.find_spec("mlx_lm") is not None:
            return "mlx"
        return "torch"

    # -- loading -----------------------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None or self.backend not in ("mlx", "torch"):
            return
        if self.backend == "mlx":
            need = estimate_model_bytes(self._runtime_model_id)
            if not self.skip_estimate and not mb.check_fits(need, f"load {self._runtime_model_id}", raise_error=True, config_gb=self.max_ram_gb):
                return
            import mlx.core as mx
            from mlx_lm import load
            cap = mb.max_ram_bytes(self.max_ram_gb)
            mx.set_memory_limit(int(cap * 0.95))
            mx.set_cache_limit(int(0.25 * mb.GB))
            self._model, self._tokenizer = load(self.model_id)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer
            from backend.connectome.device import resolve_device
            device = resolve_device("auto")
            self.device = device
            need = estimate_model_bytes(self._runtime_model_id)
            if not self.skip_estimate:
                if device.startswith("cuda"):
                    free, _total = torch.cuda.mem_get_info(device)
                    if need > free:
                        raise MemoryError(f"load {self._runtime_model_id} needs about {need / mb.GB:.1f} GB GPU memory; "
                                           f"only {free / mb.GB:.1f} GB is free")
                elif not mb.check_fits(need, f"load {self._runtime_model_id}", raise_error=True,
                                       config_gb=self.max_ram_gb):
                    return
            is_standin = self._runtime_model_id == TORCH_STANDIN_MODEL
            self._tokenizer = (AutoTokenizer if is_standin else AutoProcessor).from_pretrained(self._runtime_model_id)
            model_cls = AutoModelForCausalLM if is_standin else AutoModelForMultimodalLM
            self._model = model_cls.from_pretrained(
                self._runtime_model_id, torch_dtype="auto", device_map={"": device}
            )

    def _prompt_text(self, prompt: str) -> str:
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            msgs = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": prompt}]
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=self.thinking)
        return prompt

    # -- generation --------------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        self.load()
        t0 = time.perf_counter()
        if self.backend == "mlx":
            text, n = self._gen_mlx(prompt)
        elif self.backend == "torch":
            text, n = self._gen_torch(prompt)
        elif self.backend == "llamacpp":
            text, n = self._gen_llamacpp(prompt)
        else:
            text, n = self._gen_ollama(prompt)
        dt = time.perf_counter() - t0
        self.last_stats = {"new_tokens": n, "seconds": dt, "tokens_per_s": n / dt if dt > 0 else 0.0}
        return text

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Sequential (one 8 GB machine, one model); stats aggregate over the batch."""
        outs, tokens, t0 = [], 0, time.perf_counter()
        for p in prompts:
            outs.append(self.generate(p))
            tokens += self.last_stats["new_tokens"]
        dt = time.perf_counter() - t0
        self.last_stats = {"new_tokens": tokens, "seconds": dt, "tokens_per_s": tokens / dt if dt > 0 else 0.0}
        return outs

    def _gen_mlx(self, prompt: str) -> tuple[str, int]:
        self.load()
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(self.seed)
        sampler = make_sampler(temp=self.temperature)
        text, n = "", 0
        for resp in stream_generate(self._model, self._tokenizer, self._prompt_text(prompt),
                                    max_tokens=self.max_tokens, sampler=sampler):
            text += resp.text
            n = resp.generation_tokens
        return text, n

    def _gen_torch(self, prompt: str) -> tuple[str, int]:
        self.load()
        import torch
        inputs = self._tokenizer(text=self._prompt_text(prompt), return_tensors="pt")
        inputs = inputs.to(next(self._model.parameters()).device)
        torch.manual_seed(self.seed)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0,
                **({"temperature": self.temperature} if self.temperature > 0 else {}),
            )
        prompt_len = inputs["input_ids"].shape[-1]
        new_ids = output[0, prompt_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True), int(new_ids.numel())

    def _gen_llamacpp(self, prompt: str) -> tuple[str, int]:
        exe = shutil.which("llama-cli")
        if not exe:
            raise RuntimeError("llama-cli not found (brew install llama.cpp); model_id must be a GGUF path or -hf repo")
        args = [exe, "-n", str(self.max_tokens), "--temp", str(self.temperature), "--seed", str(self.seed),
                "-p", prompt, "--no-display-prompt", "-no-cnv"]
        args += ["-m", self.model_id] if os.path.exists(self.model_id) else ["-hf", self.model_id]
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        return out, max(1, len(out.split()))   # approximate token count

    def _gen_ollama(self, prompt: str) -> tuple[str, int]:
        body = json.dumps({"model": self.model_id, "prompt": prompt, "stream": False,
                           "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": self.max_tokens}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        return d["response"], int(d.get("eval_count", 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--standin", action="store_true", help=f"use {STANDIN_MODEL} (NOT the teacher)")
    ap.add_argument("--backend", default="auto", choices=BACKENDS)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-estimate", action="store_true",
                    help="bypass the pre-load size estimate; the RSS watchdog still aborts at the cap")
    ap.add_argument("--max-ram-gb", type=float, default=None)
    a = ap.parse_args(argv)
    mb.start_watchdog(config_gb=a.max_ram_gb)
    model = STANDIN_MODEL if a.standin else a.model
    t = Teacher(model, a.backend, a.max_tokens, a.temperature, a.seed, a.max_ram_gb, skip_estimate=a.skip_estimate)
    out = t.generate(a.prompt)
    print(out)
    s = t.last_stats
    mlx_peak = ""
    if t.backend == "mlx":
        import mlx.core as mx
        mlx_peak = f" mlx_peak_mem={mx.get_peak_memory() / mb.GB:.2f} GB"
    print(f"\n[teacher] model={t._runtime_model_id} backend={t.backend} device={t.device or 'external'} "
          f"tokens={s['new_tokens']} "
          f"{s['tokens_per_s']:.1f} tok/s (gen {s['seconds']:.1f}s) peak_rss={mb.peak_rss() / mb.GB:.2f} GB "
          f"cap={mb.max_ram_bytes(a.max_ram_gb) / mb.GB:.1f} GB{mlx_peak}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
