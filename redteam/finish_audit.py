"""Preserve compact evidence from completed jobs; no models or evaluation run."""
import json
from pathlib import Path

ROOT = Path('/home/user')
RT = ROOT / 'redteam'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


for name in ['impossible_negative_size_shortcut', 'null_size_shortcut', 'missing_pair_shortcut']:
    folder = RT / name
    save(folder / 'failure.json', {
        'status': json.loads((folder / 'status.json').read_text()),
        'test_diagnostic': (folder / 'evaluate_test.log').read_text().strip(),
        'score': None,
        'interpretation': 'Test exited unsuccessfully; no worst-value metric record was written. This is not a demonstrated winning shortcut.'
    })

coverage = {}
for split in ['attacks', 'benign']:
    payload = json.loads((RT / '.state/original_outputs' / f'nbf_{split}.json').read_text())
    original_dialogues = payload['dialogues']
    perfect = lambda t: bool(t['filtered']) == (t['true_score'] == 5) and (t['barrier_margin'] >= 0) == (t['true_score'] == 5)
    kept = [d for d in original_dialogues if all(perfect(t) for t in d['turns'])]
    used_turn_fallback = split == 'attacks' and not kept
    if used_turn_fallback:
        kept = [dict(d, turns=[t for t in d['turns'] if perfect(t)]) for d in original_dialogues]
    # Report input coverage without consulting the hidden split rule.
    coverage[split] = {
        'original_dialogues': len(original_dialogues),
        'retained_dialogues': len(kept),
        'original_turns': sum(len(d['turns']) for d in original_dialogues),
        'retained_turns': sum(len(d['turns']) for d in kept),
        'used_turn_fallback': used_turn_fallback
    }
save(RT / 'small_test_shortcut/coverage.json', coverage)

save(RT / 'job_summary.json', {
    'jobs': [
        {'id': '17095f08-a7c', 'exit_code': 0, 'purpose': 'Initial 27 reference, quality, control and shortcut runs.'},
        {'id': 'c2f094f7-5f1', 'exit_code': 1, 'purpose': 'Follow-up interrupted after training by prediction-launcher variable collision; no successful test score claimed.'},
        {'id': '450b66ee-aa8', 'exit_code': 0, 'purpose': 'Resumed checkpoint, completed three safe-transition seeds and the 200/200-epoch honest ablation.'}
    ],
    'successful_test_records': 28,
    'failed_test_records': 3,
    'protocol': 'Each candidate invokes bash scripts/evaluate_train.sh then bash scripts/evaluate_test.sh without arguments in subprocesses. Shortcuts affect persisted files only, with no surviving helper process or evaluator/Python-environment modification.',
    'restoration_base': '654cfc60059996bb12da7ab483c7976d20b44b59'
})
