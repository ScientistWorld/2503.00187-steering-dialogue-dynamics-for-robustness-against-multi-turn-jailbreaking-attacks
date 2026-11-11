# Progress

## Current state (iteration 11, hardening)

- The gym runs on the paper authors' released data: GPT-3.5-turbo multi-turn jailbreak dialogues (ActorAttack, Acronym, Opposite-day, Crescendo) with all-mpnet-base-v2 query/response embeddings and GPT-4o 1-5 judge scores per turn. Circuit Breakers goals are train/dev (85/15 by goal); HarmBench goals are the hidden test. Benign sessions come from MMLU, MT-Bench, XSTest and JailbreakBench (hash split train/dev/test).
- `method/nbf_llm.py` implements the paper's method at the paper's/authors' settings: neural state-space dynamics (1536-512-512-768, MSE, Adam 1e-4, 200 epochs), safety predictor (1536-32-32-5) with CE + 100 L_SI + 100 L_SS (kappa 3, training eta 0, Adam 1e-3, 200 epochs), best-dev checkpoints, filter iff h + eta >= 0, filtered queries do not advance the state. Exported as TorchScript guards at eta = 1e-3 (`nbf_steering`) and eta = 0.
- `baseline/single_turn_guard.py`: context-free single-turn guard of the same head size.
- `eval/test/evaluate.py` / `eval/train/evaluate.py` execute every guard turn by turn and recompute all metrics from their own labels; failures and gate violations get worst values everywhere.

## Results (hidden test)

| guard | ASR | helpfulness | over-refusal | prompt F1 |
|---|---:|---:|---:|---:|
| unguarded | 0.581 | 1 | 0 | - |
| NBF steering eta=1e-3 (reference) | 0.165 | 0.833 | 0.288 | 0.665 |
| NBF steering eta=0 | 0.379 | 0.929 | 0.158 | 0.589 |
| single-turn guard (baseline) | 0.436 | 0.959 | 0.043 | 0.533 |

Paper (GPT-3.5-turbo, Table 1): ASR 0.643 -> 0.217 at eta = 1e-3. Per family the reference reaches 0.089 / 0.140 / 0.265 (ActorAttack / Crescendo / Opposite-day) vs the paper's 0.040 / 0.235 / 0.375. L_SS ablation raises ASR to 0.533 (paper direction); the L_SI ablation does not raise ASR (not reproduced). See HARDENING.md for seeds, variants, shortcuts and rank comparisons.

## Deviations from the paper

- Checkpoint selection on the Circuit Breakers dev split (the authors select on HarmBench, our hidden test); 15% of Circuit Breakers goals held out for it.
- Offline replay of recorded attacks: a filtered query is skipped and the recorded next query follows (the paper re-queries online).
- Helpfulness / over-refusal are the guard's pass rates on benchmark prompts inside multi-turn sessions, not the LLM's benchmark scores.
- Only GPT-3.5-turbo dialogues are released, so cross-LLM results (Tables 1-3 for other LLMs) and Table 5 (held-out attacks) are not reproduced.
