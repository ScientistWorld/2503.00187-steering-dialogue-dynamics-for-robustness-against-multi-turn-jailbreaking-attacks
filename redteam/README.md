# Gym audit

`bash redteam/run_all.sh` runs the audit as a compute job. Individual variants
have a `run.sh` that installs their outputs and invokes the unmodified train
and hidden test scripts. Score JSON and small summaries are retained alongside
each run. Training uses the original method and visible data; shortcut code
uses only future-agent-visible inputs, never the hidden evaluator or its scores.
`redteam/.state/` holds temporary restoration backups and trained outputs.

Any scalar rank reported is normalized using the explicit fixed audit panel
in `rank_summary.json`. The future population of attempts is unknown, so its
normalizers and exact scalar rank cannot be predicted. Pareto improvements and
exact ties are independent of those normalizers.
