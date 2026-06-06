# Research Direction

## Research Problem
Improve lightweight multi-turn LLM safety guardrails that account for evolving dialogue context, not only the current prompt in isolation.

## Core Contribution
The contribution area is dialogue-level safety prediction with explicit context dynamics and an inference-time safety filter. Improvements should still model or use conversation state when deciding whether to block a query.

## Approach Scope
Researchers may modify the state representation, embedding model within the size budget, classifier architecture, losses, thresholding policy, or training curriculum. They may add ablations or better calibration as long as the system remains a lightweight add-on guardrail.

## Out of Bounds
Do not replace the task with target-LLM fine-tuning, a large generative guard model, or a single-turn-only keyword/moderation classifier that ignores dialogue state. Do not solve the benchmark by refusing all prompts or by using the test labels directly.
