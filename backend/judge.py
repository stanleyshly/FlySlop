"""Exact text, syntax, and elaboration evaluation using pyslang."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pyslang

from .corpus import get_target


def _diagnostic_text(diagnostic: Any, source_manager: Any) -> str:
    engine = pyslang.DiagnosticEngine(source_manager)
    return engine.formatMessage(diagnostic)


def _severity(diagnostic: Any, source_manager: Any) -> str:
    return str(pyslang.DiagnosticEngine(source_manager).getSeverity(diagnostic.code, diagnostic.location))


def validate_sv(text: str) -> dict[str, Any]:
    """Parse and elaborate source separately, returning JSON-safe results."""
    with tempfile.TemporaryDirectory(prefix="flyslop-judge-") as temp:
        path = Path(temp) / "target.sv"
        path.write_text(text, encoding="utf-8")
        tree = pyslang.syntax.SyntaxTree.fromFile(str(path))
        parse_diags = list(tree.diagnostics)
        syntax_ok = not any(d.isError() for d in parse_diags)

        compilation = pyslang.ast.Compilation()
        compilation.addSyntaxTree(tree)
        all_diags = list(compilation.getAllDiagnostics())
        # Syntax diagnostics are included in the compilation diagnostics. NewlineEOF is
        # purely formatting and does not invalidate source.
        errors = [d for d in all_diags if d.isError()]
        elaboration_ok = syntax_ok and not errors
        sm = tree.sourceManager
        diagnostics = [
            {"code": str(d.code), "severity": _severity(d, sm), "message": _diagnostic_text(d, sm)}
            for d in all_diags
        ]
        return {
            "syntax": {"passed": syntax_ok, "diagnostics": [
                {"code": str(d.code), "severity": _severity(d, sm), "message": _diagnostic_text(d, sm)}
                for d in parse_diags
            ]},
            "elaboration": {"passed": elaboration_ok, "diagnostics": diagnostics},
        }


def judge_text(text: str, target_id: str) -> dict[str, Any]:
    """Compare raw transcription bytes-as-text and validate its SV independently."""
    target = get_target(target_id)
    result = validate_sv(text)
    expected = target["text"]
    result.update({
        "target_id": target_id,
        "exact": {"passed": text == expected, "expected_length": len(expected), "actual_length": len(text)},
        # Character error rate is kept raw and case-sensitive to match exact scoring.
        "text": {"expected": expected, "actual": text},
    })
    return result
