#!/usr/bin/env python3
"""Generate the generalization suite: 2-4 givens and four operator regimes.

The suite has the same seeds, panels, families, and counts as the core suite:

    family              short (core / irrelevant)    long (core / irrelevant)
    base                6 / 1                         12 / 4
    deep                10 / 2                        24 / 6
    irrelevant_heavy    6 / 4                         12 / 12
    fanout              7 / 2                         20 / 6
    ordinary            5 / 0 (clean only)            6 / 0 (clean only)

For candidate index k, the number of givens is 2 + k mod 3 and the operator regime is
(addmul, subtract, divide, mixed)[(k // 3) mod 4]. Intermediate names are random distinct
three-digit names v100-v998. Step 0 updates the first given a by a constant; steps 1 .. h-1
read the remaining givens in order, so that every given is used; in the fanout family every
third later step re-reads the first intermediate value; the other steps read a constant 1-9
or a given other than a. Each step takes the first admissible (operation, operand) pair in
the regime's operator preference order and a shuffled operand order, where admissible means
an exact, nonnegative integer result below 100,000. A candidate is redrawn (up to 1,000
attempts) when the regime's required operators do not all occur. The irrelevant chain starts
from the given b; each statement applies plus, times, or minus with a constant 1-5 and is
inserted after one core statement. Candidates whose canonical fingerprint repeats an earlier
accepted program of the suite are skipped.

Running this script reproduces data/generalization_suite byte for byte; only the standard
library is used.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OPERATIONS, SEEDS, apply_operation, execute_sequential, render_program, write_jsonl  # noqa: E402

SALT = 'liveledger-followup-extended-v1'   # hash salt of the published data; do not change
SPECS = {
    'short': {'in_domain': (6, 1), 'unseen_depth': (10, 2), 'dead_branch_heavy': (6, 4), 'fanout': (7, 2), 'ordinary': (5, 0)},
    'long': {'in_domain': (12, 4), 'unseen_depth': (24, 6), 'dead_branch_heavy': (12, 12), 'fanout': (20, 6), 'ordinary': (6, 0)},
}
FAMILY_NAMES = {'in_domain': 'base', 'unseen_depth': 'deep', 'dead_branch_heavy': 'irrelevant_heavy',
                'fanout': 'fanout', 'ordinary': 'ordinary'}
COUNTS = {'short': 32, 'long': 16}
REGIMES = ('addmul', 'subtract', 'divide', 'mixed')


def rng_for(seed, panel, family, index, attempt):
    material = f'{SALT}:{seed}:{panel}:{family}:{index}:{attempt}'
    return random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'big'))


def make_pair(seed, panel, family, index):
    depth, dead_count = SPECS[panel][family]
    n_givens = 2 + index % 3
    regime = REGIMES[(index // 3) % len(REGIMES)]
    base = hashlib.sha256(f'{SALT}:{seed}:{panel}:{family}:{index}'.encode()).hexdigest()
    for attempt in range(1000):
        rng = rng_for(seed, panel, family, index, attempt)
        names = [f'v{x}' for x in rng.sample(range(100, 999), depth + dead_count)]
        if regime == 'divide':
            inputs = [(chr(97 + i), rng.choice((12, 18, 24, 36, 48)) if i == 0 else rng.choice((2, 3, 4, 6))) for i in range(n_givens)]
        else:
            inputs = [(chr(97 + i), rng.randint(12, 30) if i == 0 else rng.randint(2, 11)) for i in range(n_givens)]
        values = dict(inputs)
        core = []
        valid = True
        for position in range(depth):
            left = 'a' if position == 0 else names[position - 1]
            if position == 0:
                rights = list(range(1, 10))
            elif position < n_givens:
                rights = [inputs[position][0]]
            elif family == 'fanout' and position % 3 == 0:
                rights = [names[0]]
            else:
                rights = list(range(1, 10)) + [name for name, _ in inputs[1:]]
            rng.shuffle(rights)
            if regime == 'addmul':
                operations = ['times', 'plus'] if position % 4 == 0 else ['plus', 'times']
            elif regime == 'subtract':
                operations = ['minus', 'plus']
            elif regime == 'divide':
                operations = ['divided_by', 'times', 'plus']
            else:
                desired = OPERATIONS[position % 4]
                operations = [desired] + [op for op in OPERATIONS if op != desired]
            candidates = []
            for operation in operations:
                for right in rights:
                    operand = values[right] if isinstance(right, str) else right
                    try:
                        value = apply_operation(operation, values[left], operand)
                    except (ValueError, ZeroDivisionError):
                        continue
                    if 0 <= value < 100000:
                        candidates.append((operation, right, value))
            if not candidates:
                valid = False
                break
            operation, right, value = candidates[0]
            core.append(dict(target=names[position], operation=operation, left=left, right=right, value=value, dead=False))
            values[names[position]] = value
        if not valid:
            continue
        observed = {step['operation'] for step in core}
        required = {'addmul': {'plus', 'times'}, 'subtract': {'minus'}, 'divide': {'divided_by'}, 'mixed': set(OPERATIONS)}[regime]
        if not required <= observed:
            continue
        dead = []
        previous = 'b'
        for position in range(dead_count):
            candidates = []
            for operation in ('plus', 'times', 'minus'):
                for right in range(1, 6):
                    try:
                        value = apply_operation(operation, values[previous], right)
                    except ValueError:
                        continue
                    if 0 <= value < 100000:
                        candidates.append(dict(target=names[depth + position], operation=operation, left=previous,
                                               right=right, value=value, dead=True))
            if not candidates:
                valid = False
                break
            step = rng.choice(candidates)
            dead.append(step)
            values[step['target']] = step['value']
            previous = step['target']
        if not valid:
            continue
        interleaved = []
        for position, step in enumerate(core):
            interleaved.append(step)
            if position < len(dead):
                interleaved.append(dead[position])
        common = dict(pair_id=base, family=FAMILY_NAMES[family], panel=panel, n_givens=n_givens,
                      operator_regime=regime, inputs=inputs, output=names[depth - 1], answer=core[-1]['value'])
        clean = dict(common, source_id=base + ':clean', variant='clean', steps=core)
        distractor = dict(common, source_id=base + ':dead', variant='dead', steps=interleaved)
        return clean, distractor
    raise RuntimeError(f'generation exhausted: {seed} {panel} {family} {index} {regime}')


def fingerprint(program):
    payload = dict(inputs=program['inputs'], output=program['output'],
                   steps=[{k: s[k] for k in ('target', 'operation', 'left', 'right')} for s in program['steps']])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    seen = set()
    for seed in SEEDS:
        rows = []
        for panel in ('short', 'long'):
            for family in SPECS[panel]:
                accepted = index = 0
                while accepted < COUNTS[panel]:
                    pair = make_pair(seed, panel, family, index)
                    index += 1
                    key = fingerprint(pair[0])
                    if key in seen:
                        continue
                    seen.add(key)
                    for program in (pair[:1] if family == 'ordinary' else pair):
                        prompt = render_program(program)
                        assert execute_sequential(prompt) == program['answer']
                        rows.append(dict(program, prompt=prompt, canonical_fingerprint=key, source_seed=seed))
                    accepted += 1
        assert len(rows) == 432
        write_jsonl(args.out / f'sources_seed{seed}.jsonl', rows)
        strata = Counter(f"{r['n_givens']}:{r['operator_regime']}" for r in rows)
        print(json.dumps({'seed': seed, 'rows': len(rows), 'givens:regime': dict(sorted(strata.items()))}), flush=True)


if __name__ == '__main__':
    main()
