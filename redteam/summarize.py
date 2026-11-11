"""Audit-only analysis; never used for constructing candidate outputs."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
EXPERIMENT = 'open_source_multiturn_safety_tradeoff'
schema = json.loads((ROOT / 'scoring/reference.json').read_text())['experiments']
records = {}
for folder in sorted(RT.iterdir()):
    if not folder.is_dir() or not (folder / 'status.json').exists():
        continue
    status = json.loads((folder / 'status.json').read_text())
    scores = None
    if (folder / 'scores_test.json').exists():
        scores = json.loads((folder / 'scores_test.json').read_text())
    records[folder.name] = {'status': status, 'metrics': (
        scores['experiments'].get(EXPERIMENT, {}).get('results', {}).get('nbf_steering')
        if scores else None)}
reference = records['reference_saved']['metrics']
panel = {name: rec['metrics'] for name, rec in records.items()
         if rec['status']['test_exit'] == 0 and rec['metrics'] is not None}
spec = schema[EXPERIMENT]
spread = {m: statistics.pstdev([v[m] for v in panel.values()]) for m in spec['metrics']}
weight = spec['weight']
for name, rec in records.items():
    if name not in panel:
        rec['audit_rank_delta'] = None
        continue
    terms = {}
    for m, config in spec['metrics'].items():
        sign = 1 if config['higher_is_better'] else -1
        terms[m] = weight * abs(config['coefficient']) * sign * (rec['metrics'][m] - reference[m]) / (spread[m] or 1.0)
    rec['audit_rank_delta'] = sum(terms.values())
    rec['rank_terms'] = terms
    rec['pareto_at_least_reference'] = all(
        (rec['metrics'][m] - reference[m]) * (1 if c['higher_is_better'] else -1) >= -1e-12
        for m, c in spec['metrics'].items())
seeds = [records[f'reference_seed_{s}'] for s in [13, 14, 15]
         if f'reference_seed_{s}' in panel]
seed_spread = {}
for m in spec['metrics']:
    values = [r['metrics'][m] for r in seeds]
    seed_spread[m] = {'min': min(values), 'max': max(values), 'range': max(values) - min(values)} if values else None
ranks = [r['audit_rank_delta'] for r in seeds]
bound_metrics = [m for m, c in spec['metrics'].items()
                 if reference[m] == (1.0 if c['higher_is_better'] else 0.0)]
baseline = json.loads((RT / 'reference_saved/scores_test.json').read_text())['experiments'][EXPERIMENT]['results']['keyword_guard']
baseline_delta = sum(weight * abs(c['coefficient']) * (1 if c['higher_is_better'] else -1) *
                     (baseline[m] - reference[m]) / (spread[m] or 1.0)
                     for m, c in spec['metrics'].items())
result = {'rank_definition': 'sum weight * abs(coefficient) * direction * (candidate-reference) / audit_panel_population_std',
          'normalization_panel': list(panel), 'normalization_std': spread,
          'reference_method': 'reference_saved/nbf_steering', 'reference_metrics': reference,
          'reference_metrics_at_best_bound': bound_metrics, 'baseline_metrics': baseline,
          'baseline_audit_rank_delta': baseline_delta,
          'seed_metric_spread': seed_spread,
          'seed_rank_spread': {'min': min(ranks), 'max': max(ranks), 'range': max(ranks)-min(ranks)} if ranks else None,
          'records': records}
(RT / 'rank_summary.json').write_text(json.dumps(result, indent=2) + '\n')
columns = ['variant', 'test_exit', 'ASR', 'F1', 'helpfulness', 'zero_ASR', 'params_M', 'rank_delta']
lines = ['\t'.join(columns)]
for name, rec in records.items():
    metrics = rec['metrics'] or {}
    fields = [name, str(rec['status']['test_exit'])]
    for m in ['asr', 'prompt_harmfulness_f1', 'helpfulness_retention', 'zero_buffer_asr', 'guard_model_params_M']:
        fields.append(f"{metrics[m]:.6f}" if m in metrics else 'NA')
    fields.append(f"{rec['audit_rank_delta']:.6f}" if rec['audit_rank_delta'] is not None else 'NA')
    lines.append('\t'.join(fields))
(RT / 'results.tsv').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
