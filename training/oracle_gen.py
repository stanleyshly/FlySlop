"""Stage-5 oracle data generation (P5b): teacher-written specs -> teacher code -> filters -> pairs.

Real Gemma gate run (200 pairs; run it ALONE, it needs ~2.7 GB):

    FLYSLOP_MAX_RAM_GB=4 uv run --extra teacher --extra training python -m training.oracle_gen \
        --run gemma-gate200 --limit 200 --time-budget-s 3600

Stand-in smoke (0.6 GB, NOT teacher data):

    FLYSLOP_MAX_RAM_GB=1 uv run --extra teacher --extra training python -m training.oracle_gen \
        --standin --run smoke --limit 20 --time-budget-s 100

Other commands: ``--split-only`` (write train/val/test.jsonl from pairs.jsonl) and ``--estimate``.

Two task kinds
  seed   record already has a prompt and tests (mbpp, humaneval, verilogeval, rtllm) or a
         description (authored): the prompt goes straight to the teacher.
  synth  record has no NL prompt (lab, or authored without one): units are extracted (python
         functions via ast; SV modules and always-blocks via pyslang), the teacher writes a
         spec from the unit, and the spec goes back to the teacher for fresh code.

Every attempt is one line in ``data/private/oracle/<run>/pairs.jsonl``.  Everything stays under
``data/private`` because lab-derived text must not leave it.  Resume skips ids already present.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import memory_budget as mb  # noqa: E402
from training import datasets as ds  # noqa: E402
from training.tokenizer import normalize  # noqa: E402

OUT_ROOT = ds.PRIVATE / "oracle"
SEED_SOURCES = ("mbpp", "humaneval", "verilogeval", "rtllm", "authored")
SYNTH_SOURCES = ("lab", "authored")
ALL_SOURCES = ("mbpp", "humaneval", "verilogeval", "rtllm", "authored", "lab")

# ----------------------------------------------------------------------------- fixed templates
# Changing any of these changes TEMPLATE_HASH, which is recorded in every manifest/pair.
TEMPLATES = {
    "spec_py": (
        "Read the following Python function and write a concise natural-language specification for it, "
        "as a user would ask another programmer to write it. Mention the function name, its arguments "
        "and what it returns. 1 to 4 sentences. Do not include any code and do not use code blocks.\n\n"
        "```python\n{code}\n```\n\nSpecification:"),
    "spec_sv": (
        "Read the following SystemVerilog and write a concise natural-language specification for it, "
        "as a user would ask a hardware designer to write it. Mention the module name, every port with "
        "its direction and width, and the behaviour (clocking, reset, state). 1 to 6 sentences. "
        "Do not include any code and do not use code blocks.\n\n{context}"
        "```systemverilog\n{code}\n```\n\nSpecification:"),
    "gen_py": (
        "Write a Python function that satisfies this specification. Reply with a single ```python code "
        "block containing only the code, no explanation.\n\nSpecification:\n{spec}\n"),
    "gen_sv": (
        "Write a synthesizable SystemVerilog module that satisfies this specification. Reply with a "
        "single ```systemverilog code block containing only the module(s), no testbench and no "
        "explanation.\n\nSpecification:\n{spec}\n"),
}
TEMPLATE_HASH = hashlib.sha256(json.dumps(TEMPLATES, sort_keys=True).encode()).hexdigest()[:16]

# filter bounds
MIN_CODE_CHARS, MAX_CODE_CHARS = 30, 3000
MIN_SPEC_CHARS, MAX_SPEC_CHARS = 20, 900
UNIT_MIN_CHARS, UNIT_MAX_CHARS = 60, 2500
NEARDUP_TEST = 0.7          # vs any test-split record
NEARDUP_ACCEPTED = 0.9      # vs pairs already accepted (within run)
STAGES = ("extract", "ascii", "length", "syntax", "test", "dup", "neardup")


# ----------------------------------------------------------------------------- extraction of code blocks
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+#-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_LANG_TAGS = {"py": {"python", "py", "python3"}, "sv": {"systemverilog", "verilog", "sv", "v"}}


def extract_code(text: str, lang: str) -> tuple[str | None, dict[str, Any]]:
    """Pull the code out of a chat reply.

    Preference: longest fenced block tagged for ``lang``; else longest fenced block; else the raw
    text if it looks like code.  An unterminated fence (truncated generation) is accepted but
    flagged ``truncated``.  Thinking blocks are removed first.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("\r\n", "\n")
    blocks = []
    for m in _FENCE.finditer(text):
        closed = m.group(0).rstrip().endswith("```") and m.group(0).count("```") >= 2
        blocks.append((m.group(1).lower(), m.group(2).strip("\n"), not closed))
    blocks = [b for b in blocks if b[1].strip()]
    tagged = [b for b in blocks if b[0] in _LANG_TAGS[lang]]
    pick = max(tagged or blocks, key=lambda b: len(b[1]), default=None)
    if pick:
        return pick[1], {"fenced": True, "truncated": pick[2]}
    body = text.strip()
    looks = re.search(r"^\s*def \w+\(", body, re.M) if lang == "py" else re.search(r"^\s*module\s+\w+", body, re.M)
    if looks and "```" not in body:
        return body, {"fenced": False, "truncated": False}
    return None, {"fenced": False, "truncated": False}


def clean_spec(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "```" in text:
        return None                                     # spec leaked code
    text = re.sub(r"^(specification|spec)\s*:\s*", "", text, flags=re.I).strip()
    text = normalize(text)
    return text if MIN_SPEC_CHARS <= len(text) <= MAX_SPEC_CHARS else None


# ----------------------------------------------------------------------------- unit extraction
def py_units(code: str) -> list[dict[str, str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("__"):
            seg = ast.get_source_segment(code, node)
            if seg and UNIT_MIN_CHARS <= len(seg) <= UNIT_MAX_CHARS:
                out.append({"name": node.name, "kind": "function", "code": seg, "context": ""})
    return out


def sv_units(code: str) -> list[dict[str, str]]:
    """Modules, plus always-blocks of larger modules (with the module header as context)."""
    import pyslang
    tree = pyslang.syntax.SyntaxTree.fromText(code)
    root, K = tree.root, pyslang.syntax.SyntaxKind
    mods = [root] if root.kind == K.ModuleDeclaration else \
        [m for m in getattr(root, "members", []) if m.kind == K.ModuleDeclaration]
    out = []
    for m in mods:
        name = m.header.name.valueText
        text = str(m).strip()
        if UNIT_MIN_CHARS <= len(text) <= UNIT_MAX_CHARS:
            out.append({"name": name, "kind": "module", "code": text, "context": ""})
        if len(text) > UNIT_MAX_CHARS:
            header = str(m.header).strip()
            for i, x in enumerate(m.members):
                if x.kind == K.AlwaysBlock or x.kind.name.startswith("Always"):
                    seg = str(x).strip()
                    if UNIT_MIN_CHARS <= len(seg) <= UNIT_MAX_CHARS:
                        out.append({"name": f"{name}.always{i}", "kind": "always", "code": seg,
                                    "context": f"It is one always-block of a module with this header (write a complete "
                                               f"module with the same ports containing this behaviour):\n{header}\n\n"})
    return out


def build_tasks(recs: list[dict[str, Any]], sources: Iterable[str], seed: int) -> list[dict[str, Any]]:
    """Deterministic, source-interleaved task list.  Task = seed-prompt or synth-from-unit."""
    want = set(sources)
    per: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        if r["source"] not in want:
            continue
        if r["source"] in ("mbpp", "humaneval", "verilogeval", "rtllm") and not r.get("tests"):
            continue
        has_prompt = len(r.get("prompt", "").strip()) >= 20
        if has_prompt:
            per.setdefault(r["source"], []).append({
                "id": r["id"], "kind": "seed", "rec_id": r["id"], "source": r["source"], "lang": r["lang"],
                "prompt": r["prompt"].strip(), "tests": r.get("tests") or "", "licence": r["licence"],
                "ref_code": r["code"]})
        elif r["source"] in SYNTH_SOURCES:
            units = py_units(r["code"]) if r["lang"] == "py" else sv_units(r["code"])
            for u in units:
                per.setdefault(r["source"], []).append({
                    "id": f"{r['id']}#{u['name']}", "kind": "synth", "rec_id": r["id"], "source": r["source"],
                    "lang": r["lang"], "unit": u, "tests": "", "licence": r["licence"], "ref_code": u["code"]})
    rng = random.Random(seed)
    for v in per.values():
        v.sort(key=lambda t: t["id"])
        rng.shuffle(v)
    order, out = sorted(per), []
    while any(per.values()):
        for s in order:
            if per[s]:
                out.append(per[s].pop())
    return out


# ----------------------------------------------------------------------------- filters
def _sig(code: str, lang: str) -> np.ndarray:
    return ds.minhash(ds.shingles(ds.normalized_tokens(code, lang)))


def norm_hash(code: str, lang: str) -> str:
    return hashlib.sha256(" ".join(ds.normalized_tokens(code, lang)).encode()).hexdigest()[:16]


class Filters:
    """Filter chain; holds dedup state (accepted hashes/signatures, test-split signatures)."""

    def __init__(self, test_recs: Iterable[dict[str, Any]] = (), judge: Callable[..., dict] | None = None,
                 timeout: float = 60.0, neardup_test: float = NEARDUP_TEST,
                 neardup_accepted: float = NEARDUP_ACCEPTED):
        self.neardup_test, self.neardup_accepted = neardup_test, neardup_accepted
        if judge is None:
            from backend.exec_judge import judge as judge_fn
            judge = judge_fn
        self.judge, self.timeout = judge, timeout
        self.seen: dict[str, set[str]] = {"py": set(), "sv": set()}
        self.acc_sigs: dict[str, list[np.ndarray]] = {"py": [], "sv": []}
        self.test_sigs: dict[str, np.ndarray] = {}
        by: dict[str, list[np.ndarray]] = {"py": [], "sv": []}
        for r in test_recs:
            by[r["lang"]].append(_sig(r["code"], r["lang"]))
        for k, v in by.items():
            if v:
                self.test_sigs[k] = np.stack(v)

    def register(self, code: str, lang: str) -> None:
        self.seen[lang].add(norm_hash(code, lang))
        self.acc_sigs[lang].append(_sig(code, lang))

    def run(self, raw: str, task: dict[str, Any], check_test_neardup: bool = True) -> dict[str, Any]:
        """Returns {passed, fail_stage, filters:{stage:bool}, code, judge:{...}}."""
        lang, tests = task["lang"], task.get("tests", "")
        res: dict[str, Any] = {"passed": False, "fail_stage": None, "filters": {}, "code": None, "judge": None}

        def fail(stage: str) -> dict[str, Any]:
            res["filters"][stage] = False
            res["fail_stage"] = stage
            return res

        code, info = extract_code(raw, lang)
        res["extract"] = info
        if code is None:
            return fail("extract")
        res["filters"]["extract"] = True
        code = normalize(code).strip("\n") + "\n"
        res["code"] = code
        res["filters"]["ascii"] = True                  # normalize() maps everything to printable ASCII
        if not (MIN_CODE_CHARS <= len(code) <= MAX_CODE_CHARS):
            return fail("length")
        res["filters"]["length"] = True
        if lang == "sv" and not re.search(r"\bmodule\b", code):
            return fail("syntax")
        if lang == "py" and not re.search(r"\bdef\b|\bclass\b", code):
            return fail("syntax")
        j = self.judge(lang, code, tests, timeout=self.timeout)
        res["judge"] = {k: j.get(k) for k in ("ok", "stage", "error", "simulator") if k in j}
        res["judge"]["error"] = (res["judge"].get("error") or "")[:300]
        if not j["ok"]:
            # split the judge's stage into syntax (parse/elab) vs test failures
            if j.get("stage") == "test":
                res["filters"]["syntax"] = True         # parse/elab passed, the tests did not
                return fail("test")
            return fail("syntax")
        res["filters"]["syntax"] = True
        res["filters"]["test"] = True
        res["tested"] = bool(tests.strip()) and j.get("simulator", "x") is not None
        h = norm_hash(code, lang)
        if h in self.seen[lang]:
            return fail("dup")
        sig = _sig(code, lang)
        if self.acc_sigs[lang] and (np.stack(self.acc_sigs[lang]) == sig).mean(axis=1).max() >= self.neardup_accepted:
            return fail("dup")
        res["filters"]["dup"] = True
        if check_test_neardup and lang in self.test_sigs and (self.test_sigs[lang] == sig).mean(axis=1).max() >= self.neardup_test:
            return fail("neardup")
        res["filters"]["neardup"] = True
        res["passed"] = True
        return res


# ----------------------------------------------------------------------------- pipeline
def source_split(recs: list[dict[str, Any]], seed: int, holdout: Iterable[str] = ()) -> dict[str, str]:
    """rec_id -> split, from P5a's cluster-consistent split_records over all pool records."""
    sp = ds.split_records(recs, seed=seed, holdout_sources=holdout)
    return {r["id"]: name for name, rs in sp.items() for r in rs}


def _write_jsonl_line(f, obj: dict[str, Any]) -> None:
    f.write(json.dumps(obj, sort_keys=True) + "\n")
    f.flush()


def load_done(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                    # torn last line from a killed run
    return {r["id"] for r in rows}, rows


def attempt(task: dict[str, Any], teacher, filt: Filters, splits: dict[str, str], lim: dict[str, int]) -> dict[str, Any]:
    """One task: (synth spec ->) generation -> filters.  Returns the pairs.jsonl row."""
    lang = task["lang"]
    t0, toks = time.perf_counter(), 0

    def gen(prompt: str, max_tokens: int) -> str:
        nonlocal toks
        teacher.max_tokens = max_tokens
        out = teacher.generate(prompt)
        toks += int(teacher.last_stats.get("new_tokens", 0))
        return out

    row: dict[str, Any] = {
        "id": task["id"], "rec_id": task["rec_id"], "kind": task["kind"], "source": task["source"], "lang": lang,
        "licence": task["licence"], "private": task["source"] in ds.PRIVATE_SOURCES,
        "teacher": teacher.model_id, "template_hash": TEMPLATE_HASH,
        "split": splits.get(task["rec_id"], "train"), "prompt": None, "code": None, "raw": None,
    }
    if task["kind"] == "synth":
        u = task["unit"]
        p = TEMPLATES["spec_py" if lang == "py" else "spec_sv"].format(code=u["code"], context=u["context"])
        spec = clean_spec(gen(p, lim["spec"]))
        row["provenance"] = {"unit": u["name"], "unit_kind": u["kind"], "from": task["rec_id"]}
        if spec is None:
            row.update(passed=False, fail_stage="spec", filters={"spec": False})
            return _finish(row, t0, toks)
        row["filters"] = {"spec": True}
        spec_text = spec
    else:
        spec_text = task["prompt"]
        if task["source"] == "mbpp" and task["tests"]:
            first = next((l for l in task["tests"].splitlines() if l.strip().startswith("assert")), "")
            spec_text += f"\nYour code should pass this test:\n{first}"
        row["provenance"] = {"from": task["rec_id"]}
        row["filters"] = {}
    row["prompt"] = spec_text
    raw = gen(TEMPLATES["gen_py" if lang == "py" else "gen_sv"].format(spec=spec_text), lim["code"])
    row["raw"] = raw[:4000]
    res = filt.run(raw, task, check_test_neardup=row["split"] == "train")
    row["filters"].update(res["filters"])
    row.update(passed=res["passed"], fail_stage=res["fail_stage"], code=res["code"], judge=res["judge"],
               truncated=res["extract"]["truncated"], tested=res.get("tested", False))
    if res["passed"]:
        filt.register(res["code"], lang)
    return _finish(row, t0, toks)


def _finish(row: dict[str, Any], t0: float, toks: int) -> dict[str, Any]:
    row["new_tokens"], row["seconds"] = toks, round(time.perf_counter() - t0, 3)
    return row


def stage_table(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per lang: attempts, pass rate, and per-stage pass rate (conditional on reaching the stage)."""
    out: dict[str, Any] = {}
    order = ("spec",) + STAGES
    for lang in ("py", "sv"):
        rs = [r for r in rows if r["lang"] == lang]
        if not rs:
            continue
        d: dict[str, Any] = {"attempts": len(rs), "passed": sum(bool(r["passed"]) for r in rs)}
        d["pass_rate"] = round(d["passed"] / len(rs), 3)
        for s in order:
            reached = [r for r in rs if s in r.get("filters", {})]
            if reached:
                d[s] = f"{sum(r['filters'][s] for r in reached)}/{len(reached)}"
        out[lang] = d
    return out


def manifest_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c: dict[str, Any] = {"attempts": len(rows), "passed": sum(bool(r["passed"]) for r in rows)}
    c["by_source"] = {}
    for r in rows:
        d = c["by_source"].setdefault(f"{r['source']}/{r['lang']}", {"attempts": 0, "passed": 0})
        d["attempts"] += 1
        d["passed"] += bool(r["passed"])
    c["fail_stage"] = {}
    for r in rows:
        if not r["passed"]:
            c["fail_stage"][r["fail_stage"]] = c["fail_stage"].get(r["fail_stage"], 0) + 1
    c["new_tokens"] = sum(r.get("new_tokens", 0) for r in rows)
    c["gen_seconds"] = round(sum(r.get("seconds", 0.0) for r in rows), 2)
    return c


def write_split(run_dir: Path) -> dict[str, int]:
    """Passed pairs -> train/val/test.jsonl (split inherited from the source record's cluster split,
    so no near-duplicate cluster straddles splits; exact/near dups were already dropped)."""
    _, rows = load_done(run_dir / "pairs.jsonl")
    keep = [r for r in rows if r["passed"]]
    counts = {}
    for name in ("train", "val", "test"):
        sel = [r for r in keep if r["split"] == name]
        (run_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sel), encoding="utf-8")
        counts[name] = len(sel)
    return counts


def estimate(tok_s: float = 26.0, avg_tokens: float = 300.0, prefill_s: float = 1.0,
             pass_rate: float = 0.4, targets=(200, 10000)) -> dict[str, Any]:
    """ETA for a teacher at ``tok_s``: per attempt = avg_tokens / tok_s + prefill overhead."""
    per = avg_tokens / tok_s + prefill_s
    out = {"tok_s": tok_s, "avg_tokens_per_attempt": avg_tokens, "sec_per_attempt": round(per, 2),
           "assumed_pass_rate": pass_rate}
    for n in targets:
        att = n / pass_rate
        out[f"{n}_pairs"] = {"attempts": int(att), "hours": round(att * per / 3600, 2)}
    return out


def run(a: argparse.Namespace, teacher=None, judge=None) -> dict[str, Any]:
    run_dir = OUT_ROOT / a.run if not a.out_dir else Path(a.out_dir)
    if ds.PRIVATE.resolve() not in run_dir.resolve().parents and run_dir.resolve() != ds.PRIVATE.resolve():
        raise SystemExit(f"outputs must live under {ds.PRIVATE}")
    run_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = run_dir / "pairs.jsonl"
    sources = [s for s in a.sources.split(",") if s]

    pool = ds.load_all(include_private=True)
    splits = source_split(pool, a.seed, a.holdout_sources.split(",") if a.holdout_sources else ())
    test_recs = [r for r in pool if splits.get(r["id"]) == "test"]
    tasks = build_tasks(pool, sources, a.seed)
    done, prior = load_done(pairs_path)
    filt = Filters(test_recs, judge=judge, timeout=a.judge_timeout,
                   neardup_test=getattr(a, "neardup_test", NEARDUP_TEST),
                   neardup_accepted=getattr(a, "neardup_accepted", NEARDUP_ACCEPTED))
    for r in prior:
        if r["passed"]:
            filt.register(r["code"], r["lang"])
    todo = [t for t in tasks if t["id"] not in done]
    if a.limit is not None:
        todo = todo[: max(0, a.limit - len(done))]

    if teacher is None and todo:
        from training.teacher import STANDIN_MODEL, Teacher
        mb.start_watchdog(config_gb=a.max_ram_gb)
        teacher = Teacher(STANDIN_MODEL if a.standin else a.model, "mlx", 384, a.temperature, a.seed, a.max_ram_gb)
    lim = {"spec": a.spec_tokens, "code": a.code_tokens}
    start, new_rows = time.monotonic(), []
    stopped = "complete"
    with open(pairs_path, "a", encoding="utf-8") as f:
        for i, t in enumerate(todo):
            if a.time_budget_s is not None and time.monotonic() - start >= a.time_budget_s:
                stopped = "time_budget"
                break
            row = attempt(t, teacher, filt, splits, lim)
            _write_jsonl_line(f, row)
            new_rows.append(row)
            if not a.quiet:
                print(f"[{len(done) + i + 1}] {row['id']} {'PASS' if row['passed'] else 'fail:' + str(row['fail_stage'])} "
                      f"{row['new_tokens']}tok {row['seconds']}s", file=sys.stderr, flush=True)
    if todo and stopped == "complete" and a.limit is not None and len(new_rows) < len(todo):
        stopped = "limit"
    _, rows = load_done(pairs_path)
    secs = sum(r.get("seconds", 0) for r in new_rows)
    toks = sum(r.get("new_tokens", 0) for r in new_rows)
    measured = (toks / secs) if secs else 0.0
    n_pass = sum(bool(r["passed"]) for r in rows)
    manifest = {
        "run": a.run, "teacher_model": getattr(teacher, "model_id", None) if teacher else None,
        "standin": bool(a.standin), "template_hash": TEMPLATE_HASH, "seed": a.seed,
        "sources": sources, "limit": a.limit, "time_budget_s": a.time_budget_s, "stopped": stopped,
        "split_hash": hashlib.sha256(json.dumps(sorted(splits.items())).encode()).hexdigest()[:16],
        "counts": manifest_counts(rows), "stage_table": stage_table(rows),
        "this_session": {"attempts": len(new_rows), "wall_s": round(time.monotonic() - start, 1),
                         "tok_per_s": round(measured, 1), "peak_rss_gb": round(mb.peak_rss() / mb.GB, 2)},
        "estimate_at_26tok_s": estimate(26.0, (toks / len(new_rows)) if new_rows else 300.0,
                                        pass_rate=max(0.05, n_pass / len(rows)) if rows else 0.4),
        "outputs_private": True,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    manifest["split_counts"] = write_split(run_dir)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="default")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sources", default=",".join(ALL_SOURCES))
    ap.add_argument("--limit", type=int, default=None, help="total attempts in the run file (resume counts prior ones)")
    ap.add_argument("--time-budget-s", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-sources", default="", help="comma list of sources wholly in the test split")
    ap.add_argument("--spec-tokens", type=int, default=160)
    ap.add_argument("--code-tokens", type=int, default=384)
    ap.add_argument("--judge-timeout", type=float, default=60.0)
    ap.add_argument("--neardup-test", type=float, default=NEARDUP_TEST, help="reject >= this similarity to a test record")
    ap.add_argument("--neardup-accepted", type=float, default=NEARDUP_ACCEPTED,
                    help="reject >= this similarity to an already accepted pair")
    ap.add_argument("--max-ram-gb", type=float, default=None)
    ap.add_argument("--split-only", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.estimate:
        print(json.dumps(estimate(), indent=1))
        return 0
    if a.model is None:
        from training.teacher import TEACHER_MODEL
        a.model = TEACHER_MODEL
    if a.split_only:
        print(json.dumps(write_split(OUT_ROOT / a.run if not a.out_dir else Path(a.out_dir))))
        return 0
    m = run(a)
    print(json.dumps({k: m[k] for k in ("run", "teacher_model", "counts", "stage_table", "this_session",
                                        "estimate_at_26tok_s", "split_counts", "stopped")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
