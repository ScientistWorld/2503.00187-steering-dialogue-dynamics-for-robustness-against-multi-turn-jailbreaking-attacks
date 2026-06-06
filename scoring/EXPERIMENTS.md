# Experiment Overview

## open_source_multiturn_safety_tradeoff
This experiment tests the central trade-off: a guardrail should reduce unsafe turns that pass through during gradually shifting multi-turn conversations, while leaving benign conversations mostly untouched. It also tracks guard size so improvements cannot come only from replacing the guardrail with a much larger model.

The default scored method uses the stronger filtering threshold that reproduced the safety/helpfulness trade-off. An eta=0 diagnostic output is produced by the method script, but it is not part of the default scored contract because the local substitute benchmark did not reproduce the paper's eta=0 ASR closely enough.

## heldout_attack_generalization
This experiment tests whether a guardrail can transfer to attack families excluded from its training data. The benchmark keeps the same input/output schema as the main trade-off experiment, but changes the train/test split so success requires useful multi-turn safety behavior on unfamiliar attack patterns rather than memorizing one family.
