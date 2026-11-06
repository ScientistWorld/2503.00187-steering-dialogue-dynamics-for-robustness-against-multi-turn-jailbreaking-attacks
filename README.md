# Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2503.00187"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2503.00187-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2503.00187](https://arxiv.org/abs/2503.00187) |
| **Domain** | LLM safety / guardrails |
| **TL;DR** | The paper learns dialogue dynamics and a neural barrier safety predictor to filter risky multi-turn prompts before they can steer an LLM into unsafe behavior. |

## Summary

The paper addresses multi-turn jailbreaks, where a conversation gradually drifts from benign context into unsafe requests. It embeds user and assistant turns, learns a latent state-space model of dialogue evolution, and trains a safety predictor with barrier-style losses. The resulting guardrail blocks queries based on the current dialogue state and a tunable safety threshold. Reported experiments show lower attack success than prompt and fine-tuning baselines while maintaining useful behavior.

## Problem

Can a lightweight guardrail reduce successful multi-turn jailbreak attempts, where a dialogue gradually shifts from benign context toward unsafe requests, without making the assistant broadly unhelpful or over-refusing benign safety-related prompts?

## The paper's method

The paper frames multi-turn dialogue safety as a dynamical-system safety problem and introduces a neural barrier function guardrail for filtering risky queries before they are sent to an LLM.

## Key results of the paper

- On Llama-3-8B, the reported average ASR across three attacks drops from about 0.427 to 0.123 with NBF steering at eta 1e-3.
- On Phi-4, the reported average ASR drops from about 0.372 to 0.097.
- The paper reports stronger prompt harmfulness F1 than several larger guardrail baselines while using a much smaller guard model.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +104.8 |
| `gpt-5.6-sol-xhigh` | +104.8 |
| `gpt-5.6-terra-xhigh` | +104.2 |
| `gpt-5.6-luna-xhigh` | +86.2 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install git+https://github.com/zlab-princeton/scientist_world_cli
sw research 2503.00187 --mode innovate --agent claude-code
sw evaluate 2503.00187
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

