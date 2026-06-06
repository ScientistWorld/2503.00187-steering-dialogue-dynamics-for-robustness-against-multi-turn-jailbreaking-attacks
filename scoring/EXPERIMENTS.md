# Experiment Overview

## open_source_multiturn_safety_tradeoff
This experiment tests the central trade-off: a guardrail should reduce unsafe turns that pass through during gradually shifting multi-turn conversations, while leaving benign conversations mostly untouched. It also tracks guard size so improvements cannot come only from replacing the guardrail with a much larger model.

The scored method variants include a stronger filtering threshold and an eta=0 threshold, exposing the same safety/helpfulness trade-off axis emphasized by the paper.

## loss_ablation_eta_0
This experiment checks whether removing the additional context-safety training objective weakens the guardrail at the same filtering threshold. It compares the full model against a matched variant trained without that objective while keeping the data, threshold, and model size fixed.
