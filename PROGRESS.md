# Progress

## What Works

- Paper analyzed and author code cloned into `method/author_repo`.
- Implemented a runnable reproduction of the paper's computational method: sentence embeddings, neural dialogue state-space dynamics, safety predictor, safe-set loss, safety-invariance loss, and threshold filtering.
- Added method-agnostic evaluation that reads saved outputs only.
- Added a keyword baseline to verify the scoring pipeline.

## Results

Pending compute job. The submitted job will write `scoring/scores.json`.

## Remaining

- Run the end-to-end reproduction job.
- Inspect whether NBF steering beats the baseline while keeping benign over-refusal controlled.
- If results are weak, tune threshold/epochs or dataset difficulty and resubmit.

## Issues

- The full paper setup depends on paid GPT/Claude APIs and API-generated attack transcripts. Paid APIs are disallowed, so the autonomous gym uses a local substitute dataset.

## Deviations from Paper

- Data: the paper trains on GPT-3.5 responses to Circuit Breakers-derived multi-turn attacks and evaluates on HarmBench-derived attacks judged by GPT-4o. This reproduction uses a local synthetic multi-turn safety dataset with the same safe-score convention because paid APIs and unreleased generated transcripts are not usable in this environment.
- Embedding scale: the paper defaults to all-mpnet-base-v2 with 768-dimensional embeddings. This reproduction uses all-MiniLM-L6-v2 with 384-dimensional embeddings to keep the first CPU smoke test small. The state-space and NBF procedure is unchanged.
- Training scale: the paper trains dynamics and NBF for 200 epochs each. The first submitted job uses fewer epochs for a smoke test and can be scaled up after results are inspected.
