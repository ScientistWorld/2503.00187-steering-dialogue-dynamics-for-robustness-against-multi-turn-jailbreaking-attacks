# Progress

## What Works

- Paper analyzed and author code cloned into `method/author_repo`.
- Implemented a runnable reproduction of the paper's computational method: sentence embeddings, neural dialogue state-space dynamics, safety predictor, safe-set loss, safety-invariance loss, and threshold filtering.
- Added method-agnostic evaluation that reads saved outputs only.
- Added a keyword baseline to verify the scoring pipeline.
- CPU smoke job executed training and prediction end-to-end.

## Results

First smoke job wrote `scoring/scores.json`: NBF steering achieved ASR 0.0, filter recall 1.0, helpfulness retention 0.958, and over-refusal 0.042 on the initial local split. The keyword baseline also had ASR 0.0 but over-refused all benign turns, revealing that the initial split was too keyword-obvious for a useful benchmark.

## Remaining

- Run the revised nontrivial local split.
- Inspect whether NBF steering beats the keyword baseline on the safety/helpfulness trade-off.
- If results are weak, tune threshold/epochs and resubmit.

## Issues

- The full paper setup depends on paid GPT/Claude APIs and API-generated attack transcripts. Paid APIs are disallowed, so the autonomous gym uses a local substitute dataset.
- The first job failed validation because local method names were not listed in `reference.json`; this has been fixed.

## Deviations from Paper

- Data: the paper trains on GPT-3.5 responses to Circuit Breakers-derived multi-turn attacks and evaluates on HarmBench-derived attacks judged by GPT-4o. This reproduction uses a local synthetic multi-turn safety dataset with the same safe-score convention because paid APIs and unreleased generated transcripts are not usable in this environment.
- Embedding scale: the paper defaults to all-mpnet-base-v2 with 768-dimensional embeddings. This reproduction uses all-MiniLM-L6-v2 with 384-dimensional embeddings to keep the first CPU smoke test small. The state-space and NBF procedure is unchanged.
- Training scale: the paper trains dynamics and NBF for 200 epochs each. The first submitted job uses fewer epochs for a smoke test and can be scaled up after results are inspected.
