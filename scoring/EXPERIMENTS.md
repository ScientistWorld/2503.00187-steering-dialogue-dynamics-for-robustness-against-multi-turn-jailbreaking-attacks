# Experiment Overview

## open_source_multiturn_safety_tradeoff
This experiment tests the central trade-off: a guardrail should reduce unsafe turns that pass through during gradually shifting multi-turn conversations, while leaving benign conversations mostly untouched. It also tracks guard size so improvements cannot come only from replacing the guardrail with a much larger model.

The scored method variants include a stronger filtering threshold and an eta=0 threshold, exposing the same safety/helpfulness trade-off axis emphasized by the paper.
