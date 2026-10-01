# Corpus package verification results

Recorded during package B implementation.

- `uv run python -m unittest tests/test_corpus_judge.py`: 4 tests passed.
- `uv run python scripts/validate_corpus.py`: 24 targets passed syntax and standalone elaboration; 21 are authored examples; similarity groups do not cross splits.
- The malformed parse control fails syntax as expected.
- An unresolved module passes syntax but fails elaboration as expected.
- The byte-preserved student `byte_enable_decoder` snippet reports only the known `NewlineEOF` warning.
