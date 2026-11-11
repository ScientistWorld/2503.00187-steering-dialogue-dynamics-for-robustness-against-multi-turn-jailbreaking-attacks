# Research Direction

## Research Problem
Lightweight, turn-by-turn guardrails against multi-turn jailbreaks that account for how the conversation has evolved, not only for the current prompt.

## Core Contribution
Better safety decisions from dialogue context: lower attack success on unseen goals and attack strategies, with less filtering of benign traffic and fewer refusals of benign safety-adjacent prompts.

## Approach Scope
Within the guard interface and size budget, researchers may change the state representation, how context is accumulated, model architecture, losses, training data use (including the visible benign prompts), calibration and threshold policy, ensembling, or training curriculum.

## Out of Bounds
- Using hidden-test data in any form (the HarmBench dialogues of the authors' release, the hidden benign items), or reading `eval/test/`.
- Guards that game the protocol rather than judge safety: blocking by conversation style or turn position regardless of content, refusing most traffic, or relying on side channels between dialogues.
- Replacing the guard with something outside the interface (text access, LLM judges at test time, models above the size budget).
