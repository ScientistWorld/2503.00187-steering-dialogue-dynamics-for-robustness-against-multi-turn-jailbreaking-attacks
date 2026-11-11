"""Job-only measurement of model size and conflicting training objectives."""
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

root = Path('/home/user')
model = SentenceTransformer(str(root / 'data/downloads/all-MiniLM-L6-v2'), device='cpu')
embedding_params = sum(p.numel() for p in model.parameters())
ckpt = torch.load(root / 'redteam/.state/checkpoints/reference_seed_13/nbf_model.pt', map_location='cpu')
ssm_params = sum(v.numel() for v in ckpt['ssm'].values())
nbf_params = sum(v.numel() for v in ckpt['nbf'].values())
rows = [json.loads(line) for line in (root / 'data/local_dialogues/train.jsonl').read_text().splitlines()]
eligible, unsafe = 0, 0
for row in rows:
    for t in range(len(row['turns']) - 3):
        eligible += 1
        unsafe += row['turns'][t+1]['score'] == 5
out = {'embedding_params': embedding_params, 'ssm_params': ssm_params, 'nbf_params': nbf_params,
       'actual_total_params_M': (embedding_params + ssm_params + nbf_params) / 1e6,
       'declared_params_M': ckpt['guard_model_params_M'],
       'invariance_loss_eligible_transitions': eligible,
       'invariance_targets_with_unsafe_next_turn': unsafe,
       'unsafe_fraction': unsafe / eligible,
       'fix': 'Only impose forward safety on training transitions with safe current and next labels; retain all other architecture, loss weights, data, thresholds and epoch counts.'}
(root / 'redteam/training_inspection.json').write_text(json.dumps(out, indent=2) + '\n')
