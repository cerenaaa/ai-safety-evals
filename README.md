# AI Safety Evaluations

[![CI](https://github.com/cerenaaa/ai-safety-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/ai-safety-evals/actions)

Red-teaming, jailbreak detection, and safety evaluation framework for LLMs. Assesses models across toxicity, bias, robustness, and refusal-rate dimensions.

## Eval dimensions

| Dimension | What it measures |
|---|---|
| **Toxicity** | Harmful, offensive, or abusive content in outputs |
| **Bias** | Demographic disparities across protected attributes |
| **Robustness** | Output stability under prompt perturbations |
| **Refusal rate** | Whether the model correctly declines harmful requests |
| **Jailbreak resistance** | Resistance to adversarial prompt injections |

## Structure

```
ai-safety-evals/
├── evals/
│   ├── toxicity_eval.py         # Toxicity scoring via classifier + LLM judge
│   ├── bias_eval.py             # Demographic parity across protected groups
│   ├── robustness_eval.py       # Perturbation sensitivity analysis
│   └── refusal_eval.py          # Refusal rate on harmful prompt dataset
├── datasets/
│   └── safety_prompts.py        # Curated safety evaluation prompt sets
├── reporters/
│   └── safety_report.py         # Aggregate safety scorecard generator
└── run_evals.py
```

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python run_evals.py --suite full
```
