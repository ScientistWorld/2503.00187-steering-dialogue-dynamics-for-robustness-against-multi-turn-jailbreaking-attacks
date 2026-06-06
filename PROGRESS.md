# Progress

## What Works

- Paper analyzed and author code cloned into `method/author_repo`.
- Implemented a runnable reproduction of the paper's computational method: sentence embeddings, neural dialogue state-space dynamics, safety predictor, safe-set loss, safety-invariance loss, and threshold filtering.
- Added method-agnostic evaluation that reads saved outputs only.
- Added a keyword baseline to verify the scoring pipeline.
- CPU smoke job executed training and prediction end-to-end.
- The latest passed job validates the core safety/helpfulness claim on the local benchmark.

## Results

Latest passed job wrote `scoring/scores.json`: NBF steering achieved ASR 0.0, filter recall 1.0, prompt harmfulness F1 1.0, helpfulness retention 0.958, and over-refusal 0.042. The keyword baseline achieved ASR 0.433, filter recall 0.567, prompt harmfulness F1 0.683, helpfulness retention 0.825, and over-refusal 0.175.

This supports the core claim on the local substitute benchmark: the paper's neural barrier filtering blocks substantially more multi-turn unsafe turns while preserving more benign interactions than the lightweight lexical guard.

The eta=0 threshold variant achieved ASR 0.503, filter recall 0.497, prompt harmfulness F1 0.664, helpfulness retention 1.0, and over-refusal 0.0. This provides an additional setting for the paper's threshold trade-off: weaker filtering preserves benign prompts but allows many more unsafe turns through.

## Remaining

- A higher milestone would require broader coverage, such as leave-one-attack-out generalization or reproducing more of the closed/open-source LLM tables.
- A loss-ablation experiment for Table 6 has been added and is pending the next submitted job.
- Exact paper-scale reproduction remains out of scope without the API-generated attack transcripts, paid model access, and larger training budget.

## Issues

- The full paper setup depends on paid GPT/Claude APIs and API-generated attack transcripts. Paid APIs are disallowed, so the autonomous gym uses a local substitute dataset.
- The first job failed validation because local method names were not listed in `reference.json`; this has been fixed.
- The eta=0 threshold result is weaker than the paper's eta=0 numbers on the local substitute benchmark; the stronger eta=1e-3 threshold supports the reproduced core claim.

## Deviations from Paper

- Data: the paper trains on GPT-3.5 responses to Circuit Breakers-derived multi-turn attacks and evaluates on HarmBench-derived attacks judged by GPT-4o. This reproduction uses a local synthetic multi-turn safety dataset with the same safe-score convention because paid APIs and unreleased generated transcripts are not usable in this environment.
- Embedding scale: the paper defaults to all-mpnet-base-v2 with 768-dimensional embeddings. This reproduction uses all-MiniLM-L6-v2 with 384-dimensional embeddings to keep the first CPU smoke test small. The state-space and NBF procedure is unchanged.
- Training scale: the paper trains dynamics and NBF for 200 epochs each. The first submitted job uses fewer epochs for a smoke test and can be scaled up after results are inspected.
