# Evaluation

## Metrics

- **Attack success rate (ASR)**: fraction of unsafe turns that pass the guardrail. Lower is better.
- **Filter recall**: fraction of unsafe turns blocked. Higher is better.
- **Prompt harmfulness F1**: F1 for identifying unsafe prompts. Higher is better.
- **Helpfulness retention**: fraction of benign turns not filtered. Higher is better.
- **Over-refusal rate**: fraction of benign turns filtered. Lower is better.
- **Guard model parameters**: size of the guard model in millions of parameters. Lower is better when safety is comparable.

## Baselines and Targets

The paper reports large reductions in average ASR on open-source Llama-3-8B and Phi-4 models under ActorAttack, Crescendo, and Opposite-day while retaining most MTBench helpfulness and keeping XSTest over-refusal low. The local reproduction uses the same metric types on a compact multi-turn safety benchmark.

## Evaluation Protocol

Run `scripts/reproduce.sh` to train the submitted guardrail, generate attack and benign filtering decisions, run the keyword baseline, and call `scripts/evaluate.sh`. The evaluator reads only saved outputs in `tmp/outputs/` and writes `scoring/scores.json`; it does not import from `method/`.
