# Paper's Method

## Key Contribution

The paper frames multi-turn dialogue safety as a dynamical-system safety problem and introduces a neural barrier function guardrail for filtering risky queries before they are sent to an LLM.

## Approach

User queries and assistant responses are embedded with a sentence encoder. A neural state-space model learns hidden dialogue dynamics, updating a latent state from the previous state and current query and reconstructing the response embedding. A safety predictor takes the previous hidden state and current query embedding, predicts a five-level safety label, and computes an unsafe-vs-safe probability margin. Training combines response embedding reconstruction, safety classification, a safe-set loss, and a forward-invariance loss over future turns. At inference, a query is filtered when the barrier margin plus threshold is nonnegative.

## Main Claims

The primary claim is that this guardrail lowers multi-turn jailbreak attack success while retaining helpfulness and avoiding large over-refusal. Secondary claims are that the learned guardrail generalizes across target LLMs and attack types, is lightweight compared with LLM guard models, and that the safe-set and invariance losses are important for the safety/helpfulness trade-off.
