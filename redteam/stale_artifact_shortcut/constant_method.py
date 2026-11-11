"""Constant-allow implementation for the stale-artifact shortcut probe."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('cmd', nargs='?')
parser.add_argument('--input-data', required=True)
parser.add_argument('--output', required=True)
args, _ = parser.parse_known_args()
dialogs = []
for line in Path(args.input_data).read_text().splitlines():
    row = json.loads(line)
    dialogs.append({'id': row['id'], 'split': row['split'], 'attack': row['attack'],
        'turns': [{'true_score': t['score'], 'filtered': False, 'barrier_margin': -1.0}
                  for t in row['turns']]})
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps({'guard_model_params_M': 0.0, 'dialogues': dialogs}))
