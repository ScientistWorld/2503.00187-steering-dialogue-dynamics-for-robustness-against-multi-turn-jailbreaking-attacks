# Experiment Overview

## open_source_multiturn_safety_tradeoff
This experiment tests the central trade-off: a guardrail should reduce unsafe turns that pass through during gradually shifting multi-turn conversations, while leaving benign conversations mostly untouched. It also tracks guard size so improvements cannot come only from replacing the guardrail with a much larger model.

The scored contract also includes worst-group and decision-margin diagnostics so that a method cannot look strong only by performing well on the easiest attack family or by depending on a brittle decision boundary.
