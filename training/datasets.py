"""Dataset loaders, near-duplicate-aware splits and transcription pools (P5a).

Unified record: ``{id, source, lang: py|sv, prompt, code, tests?, licence}``.

Sources (see THIRD_PARTY.md for the verified licences):
  mbpp, humaneval, verilogeval, rtllm  downloaded into ``data/cache/datasets`` (git-ignored)
  authored                             ``data/corpus`` manifest entries written for FlySlop
  lab                                  local student-lab files, written ONLY to ``data/private``

Only the standard library and numpy are used.  Nothing here loads a model.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "datasets"
PRIVATE = ROOT / "data" / "private"
LAB_RECORDS = PRIVATE / "lab_records.jsonl"
LAB_CHECKOUT = ROOT / "lab-group40-fa25"

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
VERILOGEVAL_URL = "https://codeload.github.com/NVlabs/verilog-eval/tar.gz/main"
RTLLM_URL = "https://codeload.github.com/hkust-zhiyao/RTLLM/tar.gz/main"

LICENCES = {
    "mbpp": "CC-BY-4.0",
    "humaneval": "MIT",
    "verilogeval": "MIT",
    "rtllm": "MIT",
    "authored": "project-authored",
    "lab": "student-lab-unlicensed-private",
}
# Sources whose text must never leave data/private or be redistributed.
PRIVATE_SOURCES = frozenset({"lab"})


def record(id: str, source: str, lang: str, prompt: str, code: str, tests: str | None = None) -> dict[str, Any]:
    assert lang in ("py", "sv")
    out: dict[str, Any] = {"id": f"{source}/{id}", "source": source, "lang": lang, "prompt": prompt,
                           "code": code, "licence": LICENCES[source]}
    if tests:
        out["tests"] = tests
    return out


# ----------------------------------------------------------------------------- downloads

def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def _fetch_tar(url: str, dest: Path) -> Path:
    if not dest.exists() or not any(dest.iterdir()):
        dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for m in tar.getmembers():
                parts = Path(m.name).parts[1:]
                if not parts or ".." in parts or not (m.isfile() or m.isdir()):
                    continue
                target = dest.joinpath(*parts)
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(m).read())  # type: ignore[union-attr]
    return dest


def download(cache: Path = CACHE) -> None:
    """Fetch every permitted public dataset into the cache (idempotent)."""
    _fetch(MBPP_URL, cache / "mbpp" / "mbpp.jsonl")
    _fetch(HUMANEVAL_URL, cache / "humaneval" / "HumanEval.jsonl.gz")
    _fetch_tar(VERILOGEVAL_URL, cache / "verilog-eval")
    _fetch_tar(RTLLM_URL, cache / "rtllm")


# ----------------------------------------------------------------------------- loaders

def load_mbpp(cache: Path = CACHE) -> list[dict[str, Any]]:
    path = cache / "mbpp" / "mbpp.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        tests = "\n".join(d["test_list"])
        if d.get("test_setup_code"):
            tests = d["test_setup_code"] + "\n" + tests
        out.append(record(str(d["task_id"]), "mbpp", "py", d["text"], d["code"].replace("\r\n", "\n"), tests))
    return out


def load_humaneval(cache: Path = CACHE) -> list[dict[str, Any]]:
    path = cache / "humaneval" / "HumanEval.jsonl.gz"
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tests = d["test"] + f"\n\ncheck({d['entry_point']})\n"
            out.append(record(d["task_id"].split("/")[-1], "humaneval", "py", d["prompt"],
                              d["prompt"] + d["canonical_solution"], tests))
    return out


def load_verilogeval(cache: Path = CACHE) -> list[dict[str, Any]]:
    """spec-to-rtl problems. Code = reference renamed RefModule -> TopModule.

    ``tests`` = testbench + reference (RefModule) so ``code + tests`` is a complete simulation.
    """
    base = cache / "verilog-eval" / "dataset_spec-to-rtl"
    out = []
    for prompt_file in sorted(base.glob("Prob*_prompt.txt")):
        stem = prompt_file.name[: -len("_prompt.txt")]
        ref = base / f"{stem}_ref.sv"
        test = base / f"{stem}_test.sv"
        if not (ref.exists() and test.exists()):
            continue
        ref_text = ref.read_text(encoding="utf-8")
        out.append(record(stem, "verilogeval", "sv", prompt_file.read_text(encoding="utf-8"),
                          re.sub(r"\bRefModule\b", "TopModule", ref_text),
                          test.read_text(encoding="utf-8") + "\n" + ref_text))
    return out


def load_rtllm(cache: Path = CACHE) -> list[dict[str, Any]]:
    base = cache / "rtllm"
    out = []
    for desc in sorted(base.rglob("design_description.txt")):
        d = desc.parent
        designs = sorted(d.glob("verified_*.v"))
        tb = d / "testbench.v"
        if not designs or not tb.exists():
            continue
        out.append(record(str(d.relative_to(base)), "rtllm", "sv", desc.read_text(encoding="utf-8", errors="replace"),
                          designs[0].read_text(encoding="utf-8", errors="replace"),
                          tb.read_text(encoding="utf-8", errors="replace")))
    return out


def load_authored() -> list[dict[str, Any]]:
    """FlySlop-authored SV from the corpus manifest (lab-derived entries are excluded here)."""
    manifest = ROOT / "data" / "corpus" / "manifest.jsonl"
    out = []
    if not manifest.exists():
        return out
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("source_repo") != "FlySlop authored corpus":
            continue
        out.append(record(m["id"], "authored", "sv", m.get("description", ""),
                          (ROOT / m["snippet_path"]).read_text(encoding="utf-8")))
    return out


def _printable(text: str) -> bool:
    return all(c == "\n" or c == "\t" or 32 <= ord(c) < 127 for c in text)


def extract_lab(checkout: Path = LAB_CHECKOUT, out_path: Path = LAB_RECORDS,
                min_bytes: int = 200, max_bytes: int = 20000) -> int:
    """Extract lab .v/.sv/.py files (same idea as scripts/extract_lab_snippets.py).

    Output goes ONLY to the git-ignored data/private/. Returns the record count.
    """
    if not (checkout / ".git").is_dir():
        raise FileNotFoundError(f"no lab checkout at {checkout}")
    seen: set[str] = set()
    recs = []
    for path in sorted(checkout.rglob("*")):
        if path.suffix not in (".v", ".sv", ".py") or ".git" in path.parts or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            continue
        if not (min_bytes <= len(text.encode()) <= max_bytes) or not _printable(text):
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        lang = "py" if path.suffix == ".py" else "sv"
        if lang == "py":
            try:
                ast.parse(text)
            except SyntaxError:
                continue
        r = record(str(path.relative_to(checkout)), "lab", lang, "", text)
        r["standalone"] = "`include" not in text
        recs.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs), encoding="utf-8")
    return len(recs)


def load_lab(path: Path = LAB_RECORDS) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_all(cache: Path = CACHE, include_private: bool = False) -> list[dict[str, Any]]:
    """All available records; missing caches are skipped (call ``download()`` first)."""
    recs: list[dict[str, Any]] = []
    for name, fn, rel in (("mbpp", load_mbpp, "mbpp/mbpp.jsonl"), ("humaneval", load_humaneval, "humaneval/HumanEval.jsonl.gz"),
                          ("verilogeval", load_verilogeval, "verilog-eval"), ("rtllm", load_rtllm, "rtllm")):
        if (cache / rel).exists():
            recs += fn(cache)
    recs += load_authored()
    if include_private:
        recs += load_lab()
    return recs


def counts(recs: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(f"{r['source']}/{r['lang']}" for r in recs).items()))


# ----------------------------------------------------------------------------- similarity / splits

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
_P = (1 << 61) - 1


def normalized_tokens(code: str, lang: str) -> list[str]:
    """Strip comments, collapse identifiers/numbers so renamed copies still collide."""
    if lang == "py":
        code = re.sub(r"#[^\n]*", " ", code)
    else:
        code = re.sub(r"//[^\n]*", " ", code)
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    keep = _KEYWORDS_PY if lang == "py" else _KEYWORDS_SV
    out = []
    for t in _TOKEN.findall(code):
        if t[0].isdigit():
            out.append("0")
        elif t[0].isalpha() or t[0] == "_":
            out.append(t if t in keep else "id")
        else:
            out.append(t)
    return out


_KEYWORDS_PY = frozenset("""def return if else elif for while in not and or is import from class try except
    with as lambda yield pass break continue None True False assert raise""".split())
_KEYWORDS_SV = frozenset("""module endmodule input output inout wire reg logic assign always always_comb always_ff
    always_latch begin end if else case endcase default for posedge negedge parameter localparam
    function endfunction integer initial generate endgenerate genvar""".split())


def shingles(tokens: list[str], n: int = 3) -> set[int]:
    if len(tokens) < n:
        tokens = tokens + [""] * (n - len(tokens))
    return {int.from_bytes(hashlib.blake2b(" ".join(tokens[i:i + n]).encode(), digest_size=8).digest(), "big") % _P
            for i in range(len(tokens) - n + 1)}


def minhash(sh: set[int], num_perm: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _P, size=num_perm, dtype=np.uint64)
    x = np.fromiter(sh, dtype=np.uint64)
    # (a*x + b) mod 2^64 hash family; uint64 wraparound is intended.
    with np.errstate(over="ignore"):
        h = x[:, None] * a[None, :] + b[None, :]
    return h.min(axis=0)


def near_duplicate_clusters(recs: list[dict[str, Any]], threshold: float = 0.7, num_perm: int = 64,
                            seed: int = 0) -> list[int]:
    """Cluster label per record (union-find over estimated Jaccard >= threshold)."""
    sigs = np.stack([minhash(shingles(normalized_tokens(r["code"], r["lang"])), num_perm, seed) for r in recs]) \
        if recs else np.zeros((0, num_perm), dtype=np.uint64)
    parent = list(range(len(recs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(recs)):
        if i + 1 >= len(recs):
            break
        sim = (sigs[i + 1:] == sigs[i]).mean(axis=1)
        for j in np.nonzero(sim >= threshold)[0]:
            ri, rj = find(i), find(i + 1 + int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(len(recs))]


def split_records(recs: list[dict[str, Any]], seed: int = 0, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
                  holdout_sources: Iterable[str] = (), threshold: float = 0.7) -> dict[str, list[dict[str, Any]]]:
    """Deterministic train/val/test split.

    * Records in a near-duplicate cluster always share a split (no leakage).
    * ``holdout_sources`` go entirely to test (split by source).
    * Clusters spanning a held-out source and another source are pulled into test too.
    * Remaining clusters are assigned by a seeded hash of the cluster's smallest record id.
    """
    holdout = set(holdout_sources)
    labels = near_duplicate_clusters(recs, threshold=threshold, seed=seed)
    members: dict[int, list[dict[str, Any]]] = {}
    for r, c in zip(recs, labels):
        members.setdefault(c, []).append(r)
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    cut1, cut2 = fractions[0], fractions[0] + fractions[1]
    for group in members.values():
        key = min(r["id"] for r in group)
        if any(r["source"] in holdout for r in group):
            name = "test"
        else:
            u = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
            name = "train" if u < cut1 else "val" if u < cut2 else "test"
        out[name] += group
    for v in out.values():
        v.sort(key=lambda r: r["id"])
    return out


# ----------------------------------------------------------------------------- stage-2 pools

def _ok_chars(s: str) -> bool:
    return bool(s) and all(32 <= ord(c) < 127 for c in s)


def transcription_pool(train: list[dict[str, Any]], seed: int = 0,
                       word_len: tuple[int, int] = (3, 8),
                       line_lens: tuple[int, ...] = (16, 32, 64, 128),
                       per_level: int = 500) -> dict[str, list[dict[str, Any]]]:
    """Curriculum pool for stage 2: ``words`` first, then code strings of growing length.

    Levels: ``words`` (identifier/operator tokens), then ``code_<L>`` single-line strings of
    length <= L (each level holds strictly longer strings than the previous one), then
    ``code_multi`` (2-4 consecutive lines). Only printable-ASCII text from the given (train)
    records is used. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    words: dict[str, str] = {}
    lines: dict[str, str] = {}
    multi: dict[str, str] = {}
    for r in train:
        src = r["source"]
        code_lines = [ln.rstrip() for ln in r["code"].replace("\t", "    ").split("\n")]
        for ln in code_lines:
            s = ln.strip()
            if _ok_chars(s) and len(s) >= 2:
                lines.setdefault(s, src)
            for w in s.split():
                if word_len[0] <= len(w) <= word_len[1] and _ok_chars(w):
                    words.setdefault(w, src)
        nz = [ln for ln in code_lines if ln.strip()]
        for i in range(0, max(0, len(nz) - 1), 2):
            chunk = "\n".join(l.strip() for l in nz[i:i + 3])
            if _ok_chars(chunk.replace("\n", "")) and 20 <= len(chunk) <= 200:
                multi.setdefault(chunk, src)

    def pick(d: dict[str, str], level: str, pred=lambda s: True) -> list[dict[str, Any]]:
        keys = sorted(k for k in d if pred(k))
        idx = rng.permutation(len(keys))[:per_level]
        return [{"text": keys[i], "level": level, "source": d[keys[i]]} for i in sorted(idx)]

    pool = {"words": pick(words, "words")}
    lo = 0
    for L in line_lens:
        pool[f"code_{L}"] = pick(lines, f"code_{L}", lambda s, lo=lo, L=L: lo < len(s) <= L)
        lo = L
    pool["code_multi"] = pick(multi, "code_multi")
    return pool


def split_hash(splits: dict[str, list[dict[str, Any]]]) -> str:
    return hashlib.sha256(json.dumps({k: [r["id"] for r in v] for k, v in splits.items()}, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="fetch public datasets into data/cache/datasets")
    ap.add_argument("--extract-lab", action="store_true", help="write lab records to data/private (git-ignored)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.download:
        download()
    if args.extract_lab:
        print("lab records:", extract_lab())
    allr = load_all(include_private=True)
    print("counts:", json.dumps(counts(allr), indent=1))
    sp = split_records(allr, seed=args.seed)
    print({k: len(v) for k, v in sp.items()}, split_hash(sp)[:16])
