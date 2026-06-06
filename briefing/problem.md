# Problem Definition

## Research Question

Can a lightweight guardrail reduce successful multi-turn jailbreak attempts, where a dialogue gradually shifts from benign context toward unsafe requests, without making the assistant broadly unhelpful or over-refusing benign safety-related prompts?

## Why It Matters

Single-turn safety filters can miss risk that emerges only after several conversational turns. A useful solution should monitor evolving dialogue context and intervene before unsafe requests reach the assistant while preserving normal helpful conversations.

## Success Criteria

A successful system lowers multi-turn attack success rate, retains helpfulness on benign conversations, keeps false refusals low, and remains lightweight enough to run as an add-on guardrail.
