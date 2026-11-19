#!/usr/bin/env python3
"""Render the model inputs of every compared method from the released source programs.

    python scripts/build_prompts.py --suite core --out work/prompts_core
    python scripts/build_prompts.py --suite generalization --out work/prompts_generalization

The core suite yields the inputs of Direct, Instruction-only, Full-unique-short, Slice-only,
Unique-short, and LiveLedger; the generalization suite yields Direct, Slice-only,
Unique-short, and LiveLedger. Every rendered program is executed exactly before it is
written, and a program whose answer differs from the source answer aborts the build.
Output: <out>/<method>/seed<seed>.jsonl.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'scripts'))
import liveledger as ll  # noqa: E402
from common import SEEDS, execute_sequential, load_jsonl, transform_program, write_jsonl  # noqa: E402

CORE_METHODS = ('direct', 'instruction_only', 'full_unique_short', 'slice_only', 'unique_short', 'liveledger')
GENERALIZATION_METHODS = ('direct', 'slice_only', 'unique_short', 'liveledger')


def render(source, suite, method):
    if suite == 'core':
        program = ll.parse_prompt(source['prompt'])
        prompt = ll.build_prompt(program, method)
        value = ll.evaluate(ll.compress(program, method))
        return prompt, int(value) if value is not None and value.denominator == 1 else value
    prompt = transform_program(source, method)
    return prompt, execute_sequential(prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--suite', choices=('core', 'generalization'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    methods = CORE_METHODS if args.suite == 'core' else GENERALIZATION_METHODS
    for seed in args.seeds:
        sources = load_jsonl(ROOT / f'data/{args.suite}_suite/sources_seed{seed}.jsonl')
        rows = {method: [] for method in methods}
        for source in sources:
            for method in methods:
                prompt, value = render(source, args.suite, method)
                if value != source['answer']:
                    raise AssertionError((seed, source['source_id'], method, value, source['answer']))
                row = {k: source[k] for k in ('source_id', 'pair_id', 'family', 'panel', 'variant', 'answer')}
                row.update({k: source[k] for k in ('n_givens', 'operator_regime') if k in source})
                rows[method].append(dict(row, prompt=prompt))
        for method, records in rows.items():
            write_jsonl(args.out / method / f'seed{seed}.jsonl', records)
        print(json.dumps({'seed': seed, 'sources': len(sources), 'methods': len(methods)}), flush=True)


if __name__ == '__main__':
    main()
