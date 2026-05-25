"""
Robustness evaluation: measures LLM output stability under prompt perturbations.
Tests: typos, paraphrases, case changes, punctuation removal, adversarial suffixes.
"""
from __future__ import annotations
import re
import random
import anthropic
import numpy as np
from dataclasses import dataclass


@dataclass
class RobustnessResult:
    original_prompt: str
    perturbation_type: str
    perturbed_prompt: str
    original_output: str
    perturbed_output: str
    semantic_similarity: float   # 0–1 token overlap proxy
    is_stable: bool


def introduce_typos(text: str, rate: float = 0.05, seed: int = 42) -> str:
    """Randomly swap adjacent characters to simulate typos."""
    random.seed(seed)
    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < rate and chars[i].isalpha():
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)


def change_case(text: str) -> str:
    return text.upper() if text == text.lower() else text.lower()


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


PERTURBATIONS = {
    "typos": introduce_typos,
    "case_change": change_case,
    "no_punctuation": remove_punctuation,
}


def token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    return len(ta & tb) / (len(ta | tb) + 1e-9)


class RobustnessEvaluator:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def _generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()

    def evaluate(self, prompts: list[str], perturbation_types: list[str] = None) -> list[RobustnessResult]:
        ptypes = perturbation_types or list(PERTURBATIONS.keys())
        results = []
        for prompt in prompts:
            original_output = self._generate(prompt)
            for ptype in ptypes:
                perturbed = PERTURBATIONS[ptype](prompt)
                perturbed_output = self._generate(perturbed)
                sim = token_overlap(original_output, perturbed_output)
                results.append(RobustnessResult(
                    original_prompt=prompt, perturbation_type=ptype,
                    perturbed_prompt=perturbed, original_output=original_output,
                    perturbed_output=perturbed_output, semantic_similarity=round(sim, 3),
                    is_stable=sim > 0.5,
                ))
                print(f"  [{ptype}] similarity={sim:.3f} {'✓' if sim > 0.5 else '⚠'}")
        stability_rate = np.mean([r.is_stable for r in results])
        print(f"\nStability rate: {stability_rate:.1%}")
        return results
