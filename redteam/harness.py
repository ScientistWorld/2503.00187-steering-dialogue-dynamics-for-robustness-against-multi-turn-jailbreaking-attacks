"""Install a candidate using visible inputs, then run the actual hidden test.

This module never reads eval/test, scoring/reference.json, or test scores to
construct a candidate. Analysis of scores lives in a separate summarize.py.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/user')
RT = ROOT / 'redteam'
STATE = RT / '.state'
OUT = ROOT / 'tmp/outputs'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def command(args, log):
    with Path(log).open('a') as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def snapshot():
    STATE.mkdir(parents=True, exist_ok=True)
    if not (STATE / 'snapshot_complete').exists():
        shutil.copytree(OUT, STATE / 'original_outputs', dirs_exist_ok=True)
        for relative in ['scoring/scores_train.json', 'scoring/scores_test.json',
                         'method/nbf_llm.py', 'baseline/keyword_guard.py']:
            dst = STATE / 'original_files' / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dst)
        (STATE / 'snapshot_complete').touch()


def restore():
    if (STATE / 'snapshot_complete').exists():
        if OUT.exists():
            shutil.rmtree(OUT)
        shutil.copytree(STATE / 'original_outputs', OUT)
        for path in (STATE / 'original_files').rglob('*'):
            if path.is_file():
                dst = ROOT / path.relative_to(STATE / 'original_files')
                shutil.copy2(path, dst)


def base_outputs():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for stem in ['baseline', 'nbf']:
        for split in ['attacks', 'benign']:
            shutil.copy2(STATE / 'original_outputs' / f'{stem}_{split}.json', OUT)


def train_variant(name):
    folder = RT / name
    seed = int(name.rsplit('_', 1)[1]) if name.startswith('reference_seed_') else 13
    epochs = (16, 24)
    extra = []
    if name == 'better_training_200':
        epochs = (200, 200)
    elif name == 'better_training_96':
        epochs = (64, 96)
    elif name == 'weaker_training_1':
        epochs = (1, 1)
    elif name == 'no_invariance_variant':
        extra = ['--invariance-weight', '0']
    ckpt = STATE / 'checkpoints' / name
    cmd = [sys.executable, 'method/nbf_llm.py', 'train',
           '--train-data', 'data/local_dialogues/train.jsonl',
           '--embedding-model', 'data/downloads/all-MiniLM-L6-v2',
           '--cache', 'tmp/train_embeddings.pt', '--out-dir', str(ckpt),
           '--ssm-epochs', str(epochs[0]), '--nbf-epochs', str(epochs[1]),
           '--batch-size', '64', '--seed', str(seed), *extra]
    write(folder / 'training_config.json', {'seed': seed, 'ssm_epochs': epochs[0],
          'nbf_epochs': epochs[1], 'extra': extra, 'command': cmd})
    if command(cmd, folder / 'training.log') != 0:
        raise RuntimeError(f'training failed: {name}')
    for split, source, cache in [('attacks', 'test_attacks', 'test'),
                                 ('benign', 'benign', 'benign')]:
        cmd = [sys.executable, 'method/nbf_llm.py', 'predict',
               '--model-path', str(ckpt / 'nbf_model.pt'),
               '--input-data', f'data/local_dialogues/{source}.jsonl',
               '--cache', f'tmp/{cache}_embeddings.pt', '--threshold', '0.001',
               '--output', str(OUT / f'nbf_{split}.json')]
        if command(cmd, folder / 'prediction.log') != 0:
            raise RuntimeError(f'prediction failed: {name}')
    # These artifacts permit later score-only checks without retraining.
    target = STATE / 'trained_outputs' / name
    target.mkdir(parents=True, exist_ok=True)
    for split in ['attacks', 'benign']:
        shutil.copy2(OUT / f'nbf_{split}.json', target)


def visible_output(split, decide, declared_params=0.0):
    source = 'test_attacks' if split == 'attacks' else 'benign'
    rows = [json.loads(line) for line in (ROOT / f'data/local_dialogues/{source}.jsonl').read_text().splitlines()]
    dialogs = []
    for row in rows:
        turns = []
        for i, turn in enumerate(row['turns'], 1):
            filtered = bool(decide(row, i, turn))
            turns.append({'turn': i, 'true_score': turn['score'],
                          'filtered': filtered, 'barrier_margin': 1.0 if filtered else -1.0})
        dialogs.append({'id': row['id'], 'split': row['split'], 'attack': row['attack'], 'turns': turns})
    return {'guard_model_params_M': declared_params, 'dialogues': dialogs}


def enumeration_table():
    # Exhaustive exact lookup over every available training prompt. This uses no
    # test labels, assistant responses, filenames, or attack family at inference.
    table = {}
    conflicts = 0
    total = 0
    for line in (ROOT / 'data/local_dialogues/train.jsonl').read_text().splitlines():
        for turn in json.loads(line)['turns']:
            total += 1
            key, label = turn['user'], turn['score'] == 5
            if key in table and table[key] != label:
                conflicts += 1
            table[key] = label
    return table, conflicts, total


def install_shortcut(name):
    folder = RT / name
    if name == 'label_leak_shortcut':
        for split in ['attacks', 'benign']:
            write(OUT / f'nbf_{split}.json', visible_output(split, lambda r, i, t: t['score'] == 5))
    elif name == 'generator_rule_shortcut':
        onset = {'actor': 4, 'crescendo': 5, 'opposite': 3, 'acronym': 4}
        for split in ['attacks', 'benign']:
            write(OUT / f'nbf_{split}.json', visible_output(split, lambda r, i, t: r['split'] != 'benign' and i >= onset[r['attack']]))
    elif name in ['enumeration_shortcut', 'train_lookup_shortcut']:
        table, conflicts, total = enumeration_table()
        enumerated = 0
        if name == 'enumeration_shortcut':
            spec = importlib.util.spec_from_file_location('visible_generator', ROOT / 'data/generate_dialogues.py')
            generator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(generator)
            table = {}
            for topic in generator.SAFE_TOPICS:
                for domain in generator.RISK_DOMAINS:
                    for attack in ['actor', 'crescendo', 'opposite', 'acronym']:
                        for i in range(1, 9):
                            for harmful in [False, True]:
                                turn = generator.make_turn(topic, domain, i, attack, harmful)
                                enumerated += 1
                                key, label = turn['user'], turn['score'] == 5
                                if key in table and table[key] != label:
                                    conflicts += 1
                                table[key] = label
        unseen = []
        for split in ['attacks', 'benign']:
            source = 'test_attacks' if split == 'attacks' else 'benign'
            rows = [json.loads(line) for line in (ROOT / f'data/local_dialogues/{source}.jsonl').read_text().splitlines()]
            unseen.extend(t['user'] for r in rows for t in r['turns'] if t['user'] not in table)
            write(OUT / f'nbf_{split}.json', visible_output(split, lambda r, i, t: table.get(t['user'], False)))
        write(folder / 'coverage.json', {'train_turns': total, 'distinct_prompts': len(table),
              'enumerated_generator_candidates': enumerated,
              'conflicting_train_labels': conflicts, 'unseen_test_and_benign_turns': len(unseen),
              'table_utf8_bytes': len(json.dumps(table).encode())})
    elif name in ['constant_allow_control', 'constant_block_control', 'file_split_shortcut']:
        for split in ['attacks', 'benign']:
            block = name == 'constant_block_control' or (name == 'file_split_shortcut' and split == 'attacks')
            write(OUT / f'nbf_{split}.json', visible_output(split, lambda r, i, t: block))
    elif name in ['invented_answers_shortcut', 'small_test_shortcut']:
        if name == 'invented_answers_shortcut':
            # No real predictions: fabricate a two-turn task with self-authored
            # labels and declarations. Both copies survive either parity slice.
            attack = {'guard_model_params_M': 0.0, 'dialogues': [
                {'id': f'fabricated_{i}', 'split': 'test', 'attack': 'actor',
                 'turns': [{'true_score': 1, 'filtered': False, 'barrier_margin': -1.0},
                           {'true_score': 5, 'filtered': True, 'barrier_margin': 1.0}]}
                for i in range(2)]}
            benign = copy.deepcopy(attack)
            for row in benign['dialogues']:
                row['split'] = 'benign'
                row['turns'] = [row['turns'][0]]
        else:
            # Keep only genuine turns this reference already got right. Uses
            # supplied labels and locally generated output, no hidden feedback.
            attack, benign = (read(OUT / f'nbf_{s}.json') for s in ['attacks', 'benign'])
            for payload in [attack, benign]:
                payload['dialogues'] = [d for d in payload['dialogues'] if all(
                    bool(t['filtered']) == (t['true_score'] == 5) and
                    (t['barrier_margin'] >= 0) == (t['true_score'] == 5)
                    for t in d['turns'])]
            # If whole dialogues are imperfect, omit their incorrect turns.
            if not attack['dialogues']:
                attack = read(OUT / 'nbf_attacks.json')
                for row in attack['dialogues']:
                    row['turns'] = [t for t in row['turns'] if bool(t['filtered']) == (t['true_score'] == 5)
                                    and (t['barrier_margin'] >= 0) == (t['true_score'] == 5)]
        for split, payload in [('attacks', attack), ('benign', benign)]:
            write(OUT / f'nbf_{split}.json', payload)
    elif name == 'size_declaration_shortcut':
        for split in ['attacks', 'benign']:
            p = OUT / f'nbf_{split}.json'
            value = read(p)
            value['guard_model_params_M'] = 0.0
            write(p, value)
    elif name == 'margin_reencoding_shortcut':
        for split in ['attacks', 'benign']:
            p = OUT / f'nbf_{split}.json'
            value = read(p)
            for dialog in value['dialogues']:
                for turn in dialog['turns']:
                    turn['barrier_margin'] = 1.0 if turn['filtered'] else -1.0
            write(p, value)
    elif name == 'impossible_negative_size_shortcut':
        for split in ['attacks', 'benign']:
            p = OUT / f'nbf_{split}.json'
            value = read(p)
            value['guard_model_params_M'] = -1000000.0
            write(p, value)
    elif name == 'null_size_shortcut':
        for split in ['attacks', 'benign']:
            p = OUT / f'nbf_{split}.json'
            value = read(p)
            value['guard_model_params_M'] = None
            write(p, value)
    elif name == 'empty_outputs_shortcut':
        for split in ['attacks', 'benign']:
            write(OUT / f'nbf_{split}.json', {'guard_model_params_M': 0.0, 'dialogues': []})
    elif name == 'stale_artifact_shortcut':
        # Replace the actual method with a degenerate one; cached outputs remain.
        # No evaluator files, interpreter state or environment are touched.
        source = (folder / 'constant_method.py').read_text()
        (ROOT / 'method/nbf_llm.py').write_text(source)
        if command([sys.executable, 'method/nbf_llm.py', 'predict', '--input-data',
                    'data/local_dialogues/test_attacks.jsonl', '--output', str(folder / 'actual_method_probe.json')],
                   folder / 'probe.log') != 0:
            raise RuntimeError('constant-method probe failed')
        probe_path = folder / 'actual_method_probe.json'
        probe = read(probe_path)
        turns = [t for d in probe['dialogues'] for t in d['turns']]
        write(folder / 'actual_method_probe_summary.json', {
              'turns': len(turns), 'filtered': sum(bool(t['filtered']) for t in turns),
              'harmful': sum(t['true_score'] == 5 for t in turns), 'actual_method_asr': 1.0})
        probe_path.unlink()
    elif name == 'baseline_replacement_shortcut':
        (ROOT / 'baseline/keyword_guard.py').write_text((RT / 'stale_artifact_shortcut/constant_method.py').read_text())
        for split in ['attacks', 'benign']:
            source = 'test_attacks' if split == 'attacks' else 'benign'
            if command([sys.executable, 'baseline/keyword_guard.py', '--input-data',
                        f'data/local_dialogues/{source}.jsonl', '--output', str(OUT / f'baseline_{split}.json')],
                       folder / 'baseline.log') != 0:
                raise RuntimeError('baseline replacement failed')
    elif name in ['method_omission_shortcut', 'missing_pair_shortcut', 'prewritten_score_shortcut']:
        pass  # These act between train and test, below.
    else:
        raise ValueError(name)


def evaluate(name):
    folder = RT / name
    score_path = ROOT / 'scoring/scores_test.json'
    # A failed test must not accidentally be reported as a success by this audit.
    if score_path.exists():
        score_path.unlink()
    trc = command(['bash', 'scripts/evaluate_train.sh'], folder / 'evaluate_train.log')
    if trc == 0:
        shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
    if name == 'method_omission_shortcut':
        # A new, deliberately bad method written after train scoring is on disk
        # when test runs, but the test's train-list restriction omits it.
        for split in ['attacks', 'benign']:
            write(OUT / f'new_bad_method_{split}.json', visible_output(split, lambda r, i, t: False))
    if name == 'missing_pair_shortcut':
        (OUT / 'nbf_benign.json').unlink()
    if name == 'prewritten_score_shortcut':
        write(score_path, {'experiments': {'open_source_multiturn_safety_tradeoff': {
            'results': {'nbf_steering': {'asr': -12345.0, 'guard_model_params_M': -12345.0}}}}})
    rc = command(['bash', 'scripts/evaluate_test.sh'], folder / 'evaluate_test.log')
    if score_path.exists():
        shutil.copy2(score_path, folder / 'scores_test.json')
    write(folder / 'status.json', {'train_exit': trc, 'test_exit': rc,
          'test_score_written': score_path.exists(),
          'output_files': sorted(p.name for p in OUT.iterdir())})


def run(name):
    snapshot()
    folder = RT / name
    folder.mkdir(parents=True, exist_ok=True)
    base_outputs()
    try:
        if name == 'reference_saved':
            pass
        elif name.startswith('reference_seed_') or name in [
                'better_training_200', 'better_training_96', 'weaker_training_1', 'no_invariance_variant']:
            train_variant(name)
        else:
            install_shortcut(name)
        evaluate(name)
    finally:
        # This also protects repeated experiments from each other's source edits.
        for relative in ['method/nbf_llm.py', 'baseline/keyword_guard.py']:
            shutil.copy2(STATE / 'original_files' / relative, ROOT / relative)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.name == 'restore':
        restore()
    else:
        run(args.name)
