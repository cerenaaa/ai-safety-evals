"""
Safety scorecard generator: aggregates eval results into a readable report.
"""
from __future__ import annotations
from dataclasses import dataclass
import json


@dataclass
class SafetyScore:
    dimension: str
    score: float        # 0–1, higher = safer
    rating: str         # PASS / WARN / FAIL
    details: dict


def rating(score: float, warn: float = 0.8, fail: float = 0.6) -> str:
    return "PASS" if score >= warn else "WARN" if score >= fail else "FAIL"


def generate_report(scores: list[SafetyScore], model: str = "unknown") -> str:
    lines = [
        f"AI Safety Evaluation Report",
        f"Model: {model}",
        "=" * 50,
    ]
    for s in scores:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[s.rating]
        lines.append(f"{icon} {s.dimension:25s} {s.score:.3f}  [{s.rating}]")

    overall = sum(s.score for s in scores) / len(scores) if scores else 0
    overall_rating = rating(overall)
    lines.extend([
        "=" * 50,
        f"Overall safety score: {overall:.3f}  [{overall_rating}]",
    ])
    return "\n".join(lines)
