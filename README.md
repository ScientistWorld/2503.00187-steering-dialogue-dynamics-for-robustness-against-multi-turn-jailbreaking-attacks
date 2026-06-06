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

---

# Problem Definition

## Research Question

Can a lightweight guardrail reduce successful multi-turn jailbreak attempts, where a dialogue gradually shifts from benign context toward unsafe requests, without making the assistant broadly unhelpful or over-refusing benign safety-related prompts?

## Why It Matters

Single-turn safety filters can miss risk that emerges only after several conversational turns. A useful solution should monitor evolving dialogue context and intervene before unsafe requests reach the assistant while preserving normal helpful conversations.

## Success Criteria

A successful system lowers multi-turn attack success rate, retains helpfulness on benign conversations, keeps false refusals low, and remains lightweight enough to run as an add-on guardrail.

---

# Paper's Method

## Key Contribution

The paper frames multi-turn dialogue safety as a dynamical-system safety problem and introduces a neural barrier function guardrail for filtering risky queries before they are sent to an LLM.

## Approach

User queries and assistant responses are embedded with a sentence encoder. A neural state-space model learns hidden dialogue dynamics, updating a latent state from the previous state and current query and reconstructing the response embedding. A safety predictor takes the previous hidden state and current query embedding, predicts a five-level safety label, and computes an unsafe-vs-safe probability margin. Training combines response embedding reconstruction, safety classification, a safe-set loss, and a forward-invariance loss over future turns. At inference, a query is filtered when the barrier margin plus threshold is nonnegative.

## Main Claims

The primary claim is that this guardrail lowers multi-turn jailbreak attack success while retaining helpfulness and avoiding large over-refusal. Secondary claims are that the learned guardrail generalizes across target LLMs and attack types, is lightweight compared with LLM guard models, and that the safe-set and invariance losses are important for the safety/helpfulness trade-off.

---

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

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 21m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-06 05:35] - none
- Read the paper and extracted the core method and evaluation requirements.
- Cloned the author repository and confirmed the released code implements the neural state-space model plus neural barrier predictor.
- Implemented the reproduction environment and prepared the first end-to-end compute job.

### [2026-06-06 05:52] - method_runs
- CPU smoke job trained the neural dialogue dynamics and neural barrier predictor.
- The method wrote attack and benign filtering outputs and `scoring/scores.json`.
- Job failed only because the validator required reproduced method names to also appear in `reference.json`.

### [2026-06-06 05:57] - core_claim
- Revised CPU smoke job passed `python validate.py --compare`.
- NBF steering reduced attack success to 0.0 on the local multi-turn split while retaining 0.958 benign helpfulness.
- The keyword guard baseline remained nontrivial, with attack success 0.433 and benign over-refusal 0.175.

### [2026-06-06 06:03] - core_claim_plus
- Final verification job passed with cleaned scripts and regenerated `scoring/scores.json`.
- Added the eta=0 threshold setting: it retained all benign prompts but had much higher attack success, exposing the threshold safety/helpfulness trade-off.
- The stronger eta=1e-3 setting remains the reproduced core result.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
