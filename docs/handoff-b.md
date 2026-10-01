# Package B handoff: corpus and SV validation

## Deliverables

- `backend/corpus.py` loads `data/corpus/manifest.jsonl`, provides `list_targets(split=None)`, and returns metadata plus UTF-8 `text` from `get_target(target_id)`. It checks recorded SHA-256 digests and rejects paths outside the project root.
- `backend/judge.py` provides `validate_sv(text)` and `judge_text(text, target_id)`. Judge output keeps exact transcription (`exact` and `text`), syntax (`syntax`), and elaboration (`elaboration`) separate, with concise machine-readable pyslang diagnostics.
- The manifest retains provenance and rights notes for all three pre-existing student snippets. It adds split and similarity-group metadata. Twenty-one newly authored processor examples are under `data/corpus/authored/`, including the root integration target `fly_demo`.
- `scripts/validate_corpus.py` checks digests, group split isolation, syntax, and standalone elaboration. `tests/test_corpus_judge.py` checks metadata/text loading, exact scoring, and a malformed syntax control.

## Verification

Commands run from repository root:

```sh
uv run python -m unittest tests/test_corpus_judge.py
uv run python scripts/validate_corpus.py
```

Both completed successfully. Results: 4 unit tests passed; all 24 corpus targets passed syntax and elaboration; 21 are original authored examples. A malformed control fails syntax, and an unresolved module passes syntax but fails elaboration. `byte_enable_decoder` keeps its known `NewlineEOF` warning because its original bytes are preserved.

## Split and rights notes

Similarity groups are kept within one split. Authored arithmetic, comparison, and shift families form the current test split; other authored families are train. This is a small transcription corpus, not a statistically sufficient generalization benchmark. The ECE 4750 lab snippets remain marked as student work with no license found; this package does not grant downstream redistribution rights for those three files. The new examples are original and intended for redistribution with FlySlop.

## Integration contract

The server can use:

```python
from backend.corpus import get_target, list_targets
from backend.judge import judge_text

target = get_target("fly_demo")  # dict containing manifest fields and `text`
result = judge_text(typed_text, "fly_demo")
```

`judge_text` returns JSON-serializable nested results. Validation uses slang/pyslang parse diagnostics and elaboration diagnostics independently. Passing elaboration is not a functional correctness result.
