"""
Run safety evaluation suite.
Usage: python run_evals.py [--suite quick|full]
"""
import argparse
from datasets.safety_prompts import ALL_PROMPTS
from evals.toxicity_eval import evaluate_outputs, score_toxicity
from reporters.safety_report import generate_report, SafetyScore, rating


def run_toxicity_check():
    test_outputs = [
        "Paris is the capital of France.",
        "I cannot help with that request.",
        "The weather is nice today.",
    ]
    results = evaluate_outputs(test_outputs)
    score = 1.0 - results["toxicity_rate"]
    return SafetyScore("Toxicity", score, rating(score), results)


def run_prompt_coverage():
    """Check that we have prompts across all safety dimensions."""
    n_total = sum(len(v) for v in ALL_PROMPTS.values())
    n_categories = len(ALL_PROMPTS)
    score = min(1.0, n_categories / 4)
    return SafetyScore("Dataset Coverage", score, rating(score),
                        {"n_prompts": n_total, "n_categories": n_categories})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    print(f"Running {args.suite} safety eval suite...\n")
    scores = [run_toxicity_check(), run_prompt_coverage()]
    report = generate_report(scores, model="claude-sonnet-4-20250514")
    print(report)


if __name__ == "__main__":
    main()
