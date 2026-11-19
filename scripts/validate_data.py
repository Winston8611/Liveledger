#!/usr/bin/env python3
"""Check the released test data and the worked example of the paper; no GPU is needed.

    python scripts/validate_data.py            # check data and metadata/SHA256SUMS.json
    python scripts/validate_data.py --write    # (re)write metadata/SHA256SUMS.json

Checks: (1) the running example (a = 13, b = 6, answer 76) is sliced and renamed exactly as in
the paper; (2) every structured source executes to its stored answer, and so does every
compressed version of it; (3) clean and distractor rows of a pair share the answer, and the
counts per seed, panel, and family match the paper; (4) every repetition prompt executes to
its answer; (5) every natural-language oracle program executes to the reference answer;
(6) file hashes match metadata/SHA256SUMS.json.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'scripts'))
import liveledger as ll  # noqa: E402
from build_prompts import CORE_METHODS, GENERALIZATION_METHODS  # noqa: E402
from common import SEEDS, execute_sequential, load_jsonl, transform_program  # noqa: E402

EXAMPLE = """Track the named quantities and compute the requested value.
Initially, a = 13 and b = 6.
1. set c0 to a plus 5.
2. set d0 to b times 3.
3. set c1 to c0 plus b.
4. set c2 to c1 plus 1.
5. set c3 to c2 times 2.
6. set c4 to c3 plus c2.
7. set c5 to c4 plus 1.
What is c5?"""
FAMILY_COUNTS = {'base': 96, 'deep': 96, 'irrelevant_heavy': 96, 'fanout': 96, 'ordinary': 48}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_example():
    program = ll.parse_prompt(EXAMPLE)
    assert ll.evaluate(program) == Fraction(76)
    sliced = ll.compress(program, 'slice_only')
    unique = ll.compress(program, 'unique_short')
    live = ll.compress(program, 'liveledger')
    assert [s.target for s in sliced.steps] == ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
    assert [s.target for s in unique.steps] == list('cdefgh') and unique.query == 'h'
    assert [s.target for s in live.steps] == ['a', 'a', 'a', 'b', 'a', 'a'] and live.query == 'a'
    assert live.steps[3] == ll.Step('b', 'a', 'times', '2') and live.steps[4] == ll.Step('a', 'b', 'plus', 'a')
    assert all(ll.evaluate(p) == Fraction(76) for p in (sliced, unique, live))


def check_suite(suite):
    total = 0
    for seed in SEEDS:
        records = load_jsonl(ROOT / f'data/{suite}_suite/sources_seed{seed}.jsonl')
        assert len(records) == 432 and Counter(r['panel'] for r in records) == {'short': 288, 'long': 144}
        assert Counter(r['family'] for r in records) == FAMILY_COUNTS
        pairs = defaultdict(list)
        for record in records:
            pairs[record['pair_id']].append(record)
            if suite == 'core':
                program = ll.parse_prompt(record['prompt'])
                assert ll.evaluate(program) == record['answer']
                for method in CORE_METHODS:
                    assert ll.evaluate(ll.compress(program, method)) == record['answer']
            else:
                assert execute_sequential(record['prompt']) == record['answer']
                for method in GENERALIZATION_METHODS:
                    assert execute_sequential(transform_program(record, method)) == record['answer']
        for group in pairs.values():
            assert len({r['answer'] for r in group}) == 1
            expected = {'clean'} if group[0]['family'] == 'ordinary' else {'clean', 'dead'}
            assert {r['variant'] for r in group} == expected and len(group) == len(expected)
        total += len(records)
    return total


def check_repetition():
    total = 0
    for seed in SEEDS:
        for level in ('rep00', 'rep25', 'rep50', 'rep75'):
            records = load_jsonl(ROOT / f'data/repetition/{level}/seed{seed}.jsonl')
            assert len(records) == 64
            assert all(execute_sequential(r['prompt']) == r['answer'] for r in records)
            total += len(records)
    return total


def check_natural_language():
    records = load_jsonl(ROOT / 'data/natural_language/nl_items.jsonl')
    datasets = Counter(r['dataset'] for r in records)
    assert len(records) == 3229 and datasets == {'gsmplus': 2638, 'gsmic': 591}
    oracle = 0
    for record in records:
        if record.get('oracle'):
            assert ll.evaluate(ll.parse_prompt(record['oracle'])) == record['answer']
            oracle += 1
    return dict(rows=len(records), oracle_programs=oracle, **datasets)


def file_hashes():
    manifest = ROOT / 'metadata/SHA256SUMS.json'
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and p != manifest and '__pycache__' not in p.parts and not p.name.startswith('.')
            and 'work' not in p.relative_to(ROOT).parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true', help='write metadata/SHA256SUMS.json')
    args = parser.parse_args()
    check_example()
    summary = dict(core_suite=check_suite('core'), generalization_suite=check_suite('generalization'),
                   repetition=check_repetition(), natural_language=check_natural_language())
    hashes = file_hashes()
    manifest = ROOT / 'metadata/SHA256SUMS.json'
    if args.write:
        manifest.write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    else:
        assert json.loads(manifest.read_text()) == hashes, 'file hashes differ from metadata/SHA256SUMS.json'
    print(json.dumps(dict(summary, files_hashed=len(hashes), status='all checks passed'), indent=2))


if __name__ == '__main__':
    main()
