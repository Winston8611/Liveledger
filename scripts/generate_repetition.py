#!/usr/bin/env python3
"""Generate the controlled-repetition set: identical arithmetic, four repetition levels.

For each seed, chain depth d in {12, 16}, and index k < 32, a base chain is drawn with
a in [3, 9] and b in [2, 4]: v0 = a plus b, v1 = v0 times 2, v2 = v1 plus b, ... (plus b and
times 2 alternate). The four levels rep00, rep25, rep50, and rep75 keep the givens, the
operations, the constants, the query position, and the answer, and change only the target
names: the last s targets are renamed alternately to r and s, which makes assignment lines
repeat verbatim. For every level, s is the suffix length whose fraction of repeated lines
is closest to 0, 0.25, 0.50, or 0.75 (ties go to the shorter suffix). Running this script
reproduces data/repetition byte for byte; only the standard library is used.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEEDS, execute_sequential, repeated_line_fraction, render_program, write_jsonl

METHODS = ('rep00', 'rep25', 'rep50', 'rep75')
TARGETS = dict(zip(METHODS, (0.0, 0.25, 0.50, 0.75)))
DEPTHS = (12, 16)
PER_CELL = 32


def base_program(seed, depth, index):
    identity = hashlib.sha256(f'liveledger-controlled-repetition-v1:{seed}:{depth}:{index}'.encode()).hexdigest()
    rng = random.Random(int(identity[:16], 16))
    a, b = rng.randint(3, 9), rng.randint(2, 4)
    values = {'a': a, 'b': b}
    steps = []
    for position in range(depth):
        left = 'a' if position == 0 else f'v{position - 1}'
        operation, right = ('plus', 'b') if position % 2 == 0 else ('times', 2)
        operand = values[right] if isinstance(right, str) else right
        value = values[left] + operand if operation == 'plus' else values[left] * operand
        target = f'v{position}'
        steps.append(dict(target=target, operation=operation, left=left, right=right, value=value, dead=False))
        values[target] = value
    return dict(pair_id=identity, slice=f'chain{depth:02d}', panel='short', n_givens=2,
                operator_regime='controlled_add_mul', inputs=[('a', a), ('b', b)], output=f'v{depth - 1}',
                answer=steps[-1]['value'], steps=steps, source_id=identity + ':source', variant='clean')


def renamed_program(base, suffix):
    depth = len(base['steps'])
    prefix = depth - suffix
    names = [f'v{i}' for i in range(prefix)] + [('r' if i % 2 == 0 else 's') for i in range(prefix, depth)]
    aliases = {f'v{i}': names[i] for i in range(depth)}
    steps = []
    for step in base['steps']:
        steps.append(dict(step, target=aliases[step['target']], left=aliases.get(step['left'], step['left']),
                          right=aliases.get(step['right'], step['right']) if isinstance(step['right'], str) else step['right']))
    return dict(base, steps=steps, output=aliases[base['output']])


def best_program(base, target):
    candidates = []
    for suffix in range(len(base['steps']) + 1):
        program = renamed_program(base, suffix)
        prompt = render_program(program)
        realized = repeated_line_fraction(prompt)
        candidates.append((abs(realized - target), suffix, program, prompt, realized))
    _, suffix, program, prompt, realized = min(candidates, key=lambda item: (item[0], item[1]))
    return program, prompt, realized, suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for seed in SEEDS:
        rows = {method: [] for method in METHODS}
        for depth in DEPTHS:
            for index in range(PER_CELL):
                base = base_program(seed, depth, index)
                for method in METHODS:
                    program, prompt, realized, suffix = best_program(base, TARGETS[method])
                    assert execute_sequential(prompt) == base['answer']
                    source_id = base['pair_id'] + ':' + method
                    rows[method].append(dict(source_id=source_id, pair_id=base['pair_id'],
                                             slice=f'short:chain{depth:02d}', variant='clean', answer=base['answer'],
                                             prompt=prompt, source_prompt=render_program(base),
                                             semantic_value=base['answer'], repeated_fraction=realized,
                                             target_repetition=TARGETS[method], reused_suffix=suffix, chain_depth=depth))
        for method, records in rows.items():
            write_jsonl(args.out / method / f'seed{seed}.jsonl', records)
        print(json.dumps({'seed': seed, 'rows_per_method': len(rows['rep00']),
                          'realized': {m: sum(r['repeated_fraction'] for r in rows[m]) / len(rows[m]) for m in METHODS}}), flush=True)


if __name__ == '__main__':
    main()
