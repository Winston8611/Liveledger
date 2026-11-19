#!/usr/bin/env python3
"""Generate the core suite of the LiveLedger structured benchmark.

The core suite uses two givens (a, b) and the operations plus and times. For every data
seed it contains five program families in a short and a long panel:

    family              short (core / irrelevant)    long (core / irrelevant)
    base                5-6 / 1                       12 / 4
    deep                9-10 / 2                      24 / 6
    irrelevant_heavy    6 / 4                         12 / 12
    fanout              7 / 2                         20 / 6
    ordinary            4-5 / 0 (clean only)          6 / 0 (clean only)

Each paired family yields a clean program and a distractor program that interleaves an
irrelevant chain into the same core computation; both have the same answer. Candidates are
drawn with index k = 0, 1, 2, ... from a pseudo-random generator seeded by a hash of
(seed, family, k). A candidate is skipped when its canonical fingerprint (the program with
intermediate names replaced by their definition order) equals that of an accepted program of
any earlier family or seed, or one of the reserved fingerprints in
metadata/reserved_fingerprints.txt, which belong to development programs that are kept out
of the test set. Generation stops after 32 short and 16 long accepted programs per family.

The internal family keys below (in_domain, unseen_depth, dead_branch_heavy, fanout,
ordinary) only enter the hashes that seed the generator; the data files use the family
names base, deep, irrelevant_heavy, fanout, and ordinary. Running this script with the
default arguments reproduces data/core_suite byte for byte. Only the standard library is used.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1663, 1693, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777)
FAMILIES = ('in_domain', 'unseen_depth', 'dead_branch_heavy', 'fanout', 'ordinary')
COUNTS = (('short', 32), ('long', 16))
LONG_SPECS = {'in_domain': (12, 4), 'unseen_depth': (24, 6), 'dead_branch_heavy': (12, 12),
              'fanout': (20, 6), 'ordinary': (6, 0)}
CEILING = 100_000
FAMILY_NAMES = {'in_domain': 'base', 'unseen_depth': 'deep', 'dead_branch_heavy': 'irrelevant_heavy',
                'fanout': 'fanout', 'ordinary': 'ordinary'}


def step(target, operation, left, right, value, dead=False):
    return dict(target=target, operation=operation, left=left, right=right, value=value, dead=dead)


def apply(operation, left, right):
    return left * right if operation == 'mul_const' else left + right


# ---------------------------------------------------------------- short panel
def short_rng(seed, family, index):
    digest = hashlib.sha256(f'{seed}:evaluation:{family}:{index}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def short_shape(family, index):
    """(core length, irrelevant statements, fanout positions)."""
    return {'unseen_depth': (9 + index % 2, 2, True),
            'dead_branch_heavy': (6, 4, False),
            'fanout': (7, 2, True),
            'ordinary': (4 + index % 2, 0, False)}.get(family, (5 + index % 2, 1, index % 4 == 0))


def short_pair(seed, family, index):
    rng = short_rng(seed, family, index)
    inputs = {'a': rng.randint(3, 19), 'b': rng.randint(2, 11)}
    depth, dead_count, fanout = short_shape(family, index)
    values, core, previous = dict(inputs), [], 'a'
    for position in range(depth):
        if position == 0:                       # c0 = a + constant
            operation, left, right = 'add_const', 'a', rng.randint(2, 7)
        elif position == 1:                     # c1 = c0 + b
            operation, left, right = 'add_var', previous, 'b'
        elif fanout and position % 2 == 0:      # fanout: re-read the given a
            operation, left, right = 'add_var', previous, 'a'
        elif position % 3 == 0:                 # every third position doubles
            operation, left, right = 'mul_const', previous, 2
        else:
            operation, left, right = 'add_const', previous, rng.randint(1, 6)
        value = apply(operation, values[left], values[right] if isinstance(right, str) else right)
        assert value <= CEILING
        core.append(step(f'c{position}', operation, left, right, value))
        values[f'c{position}'] = value
        previous = f'c{position}'
    dead, source = [], 'b'                      # irrelevant chain d0, d1, ... starting from b
    for position in range(dead_count):
        operation = 'mul_const' if position % 2 == 0 else 'add_const'
        right = rng.randint(2, 4)
        value = apply(operation, values[source], right)
        dead.append(step(f'd{position}', operation, source, right, value, dead=True))
        values[f'd{position}'] = value
        source = f'd{position}'
    mixed, pending = [], list(dead)             # insert after core statements 1, 3, 5, ...
    for position, item in enumerate(core):
        mixed.append(item)
        if pending and position % 2 == 0:
            mixed.append(pending.pop(0))
    mixed.extend(pending)
    pair_id = hashlib.sha256(f'liveledger-delta:{seed}:evaluation:{family}:{index}'.encode()).hexdigest()
    return pair_id, inputs, core, mixed


# ----------------------------------------------------------------- long panel
def long_pair(seed, family, index):
    pair_id = hashlib.sha256(f'liveledger-longreuse-v1:{seed}:{family}:{index}'.encode()).hexdigest()
    rng = random.Random(int(pair_id[:16], 16))
    depth, dead_count = LONG_SPECS[family]
    names = [f'v{number}' for number in rng.sample(range(100, 999), depth + dead_count)]
    inputs = {'a': rng.randint(3, 19), 'b': rng.randint(2, 11)}
    values, core = dict(inputs), []
    for position in range(depth):
        if position == 0:                       # first anchor = a + constant
            left, right, operation = 'a', rng.randint(2, 7), 'add_const'
        elif position == 1:                     # second anchor = b + constant
            left, right, operation = 'b', rng.randint(1, 6), 'add_const'
        elif position == 2:                     # first anchor + second anchor
            left, right, operation = names[0], names[1], 'add_var'
        elif position == depth - 1 or (family == 'fanout' and position % 4 == 0):
            anchor = names[(position // 4) % 2] if family == 'fanout' else names[0]
            left, right, operation = names[position - 1], anchor, 'add_var'
        elif position == depth // 2:            # re-read the second anchor
            left, right, operation = names[position - 1], names[1], 'add_var'
        elif position % 7 == 0:
            left, right, operation = names[position - 1], 2, 'mul_const'
        else:
            left, right, operation = names[position - 1], rng.randint(1, 9), 'add_const'
        value = apply(operation, values[left], values[right] if isinstance(right, str) else right)
        values[names[position]] = value
        core.append(step(names[position], operation, left, right, value))
    dead, previous = [], 'b'                    # irrelevant chain starting from b
    for position in range(dead_count):
        target, increment = names[depth + position], rng.randint(1, 9)
        value = values[previous] + increment
        dead.append(step(target, 'add_const', previous, increment, value, dead=True))
        values[target] = value
        previous = target
    mixed = []                                  # insert after every core statement
    for position, item in enumerate(core):
        mixed.append(item)
        if position < len(dead):
            mixed.append(dead[position])
    assert 0 <= core[-1]['value'] < CEILING
    return pair_id, inputs, core, mixed


# --------------------------------------------------------- rendering & checks
def render(inputs, steps, output):
    words = {'add_const': 'plus', 'add_var': 'plus', 'mul_const': 'times'}
    lines = ['Track the named quantities and compute the requested value.',
             f"Initially, a = {inputs['a']} and b = {inputs['b']}."]
    lines += [f"{i}. set {s['target']} to {s['left']} {words[s['operation']]} {s['right']}."
              for i, s in enumerate(steps, 1)]
    lines += [f'What is {output}?', 'Show the calculation and end with #### followed by the integer only.']
    return '\n'.join(lines)


def execute(inputs, steps, output):
    values = dict(inputs)
    for s in steps:
        assert s['target'] not in values and s['left'] in values
        right = values[s['right']] if isinstance(s['right'], str) else s['right']
        assert apply(s['operation'], values[s['left']], right) == s['value']
        values[s['target']] = s['value']
    return values[output]


def fingerprint(inputs, core, output):
    """Hash of the program with intermediate names replaced by their definition order."""
    names = {'a': 0, 'b': 1}
    program = []
    for s in core:
        right = ['variable', names[s['right']]] if isinstance(s['right'], str) else ['constant', s['right']]
        program.append(['times' if s['operation'] == 'mul_const' else 'plus', names[s['left']], right])
        names[s['target']] = len(names)
    payload = [[['a', inputs['a']], ['b', inputs['b']]], program, names[output]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def candidate(seed, panel, family, index):
    pair_id, inputs, core, mixed = (short_pair if panel == 'short' else long_pair)(seed, family, index)
    output, answer = core[-1]['target'], core[-1]['value']
    key = fingerprint(inputs, core, output)
    variants = [('clean', core)] if family == 'ordinary' else [('clean', core), ('dead', mixed)]
    rows = []
    for variant, steps in variants:
        assert execute(inputs, steps, output) == answer
        rows.append(dict(source_id=f'{pair_id}:{variant}', pair_id=pair_id, family=FAMILY_NAMES[family], variant=variant,
                         answer=answer, prompt=render(inputs, steps, output), panel=panel,
                         program=dict(inputs=inputs, steps=steps, output=output),
                         canonical_fingerprint=key, source_seed=seed))
    return key, rows


def generate(seeds, reserved):
    seen = set(reserved)
    for seed in seeds:
        rows = []
        for panel, count in COUNTS:
            for family in FAMILIES:
                accepted = index = 0
                while accepted < count:
                    key, group = candidate(seed, panel, family, index)
                    index += 1
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.extend(group)
                    accepted += 1
        assert len(rows) == 432
        yield seed, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reserved', type=Path, default=ROOT / 'metadata/reserved_fingerprints.txt')
    args = parser.parse_args()
    reserved = {line.strip() for line in args.reserved.read_text().splitlines() if line.strip()}
    args.out.mkdir(parents=True, exist_ok=True)
    for seed, rows in generate(SEEDS, reserved):
        with (args.out / f'sources_seed{seed}.jsonl').open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps({'seed': seed, 'rows': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
