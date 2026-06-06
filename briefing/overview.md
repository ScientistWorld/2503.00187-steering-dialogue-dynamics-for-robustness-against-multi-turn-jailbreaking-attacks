# Overview

- **Paper ID:** 2503.00187
- **Title:** Steering Dialogue Dynamics for Robustness against Multiturn Jailbreaking Attacks
- **Domain:** LLM safety / guardrails
- **TL;DR:** The paper learns dialogue dynamics and a neural barrier safety predictor to filter risky multi-turn prompts before they can steer an LLM into unsafe behavior.

## Short Summary

The paper addresses multi-turn jailbreaks, where a conversation gradually drifts from benign context into unsafe requests. It embeds user and assistant turns, learns a latent state-space model of dialogue evolution, and trains a safety predictor with barrier-style losses. The resulting guardrail blocks queries based on the current dialogue state and a tunable safety threshold. Reported experiments show lower attack success than prompt and fine-tuning baselines while maintaining useful behavior.

## Key Results

- On Llama-3-8B, the reported average ASR across three attacks drops from about 0.427 to 0.123 with NBF steering at eta 1e-3.
- On Phi-4, the reported average ASR drops from about 0.372 to 0.097.
- The paper reports stronger prompt harmfulness F1 than several larger guardrail baselines while using a much smaller guard model.
