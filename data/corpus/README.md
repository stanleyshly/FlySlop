# SystemVerilog transcription corpus

`manifest.jsonl` is the source of truth for target IDs, text paths, SHA-256 digests, rights notes, declared validation level, and train/test split metadata. The corpus currently has 24 standalone modules: three copied from the user-provided ECE 4750 lab checkout and 21 original examples authored for this project.

The three lab files are byte-for-byte copies of student lab code, not lecture excerpts. Their source checkout was `cornell-ece4750/lab-group40-fa25` at commit `fea4ad8c8b928c2e84911af6b508ce7adbc8adc3`. No license file was found there. The user provided the checkout and authorized use in this project; the recorded rights note does not imply third-party redistribution rights. Do not publish those three files outside the project without checking permission. The original `authored/` examples are intended for redistribution with this project.

Targets are grouped by `similarity_group`; a group is assigned wholly to one split to keep close structural siblings from crossing train/test. The existing lab excerpts are assigned two to train and one to test. Authored examples currently belong to train to avoid claiming held-out generalization from an unreviewed corpus. These labels are corpus organization only, not benchmark results.

Run validation with:

```sh
uv run python scripts/validate_corpus.py
uv run python -m unittest tests/test_corpus_judge.py
```

The command checks digests, split isolation, syntax, and elaboration with `pyslang`. Syntax and elaboration are reported independently by the backend judge. A harmless `NewlineEOF` formatting warning is retained for the byte-for-byte `byte_enable_decoder` copy. Passing parse/elaboration does not establish functional correctness.
