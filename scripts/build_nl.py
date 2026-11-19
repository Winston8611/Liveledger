#!/usr/bin/env python3
"""Build the natural-language test items and their oracle programs.

    python scripts/build_nl.py --raw /path/to/raw --out work/nl_items.jsonl

The raw directory holds the public files gsm8k_train.parquet, gsm8k_test.parquet,
gsmplus_test.jsonl, GSM-IC_2step.json, and GSM-IC_mstep.json.
GSM-Plus "distraction insertion": every pair (seed_question, question) shares one integer
answer; the original question is the clean variant and the perturbed one the distractor.
GSM-IC 2-step / m-step: each base problem with an integer answer contributes up to five
distractor variants, drawn with a fixed random seed from different combinations of the
GSM-IC role, number, and sentence labels (fewer when fewer combinations exist).
Oracle programs are derived from the GSM8K reference solutions (<<expr=value>> annotations):
numbers that equal an earlier result reference that step, the remaining numbers become givens,
and every expression is decomposed into binary steps. An oracle program is kept only if the
exact interpreter reproduces the reference answer.
"""
import argparse
import ast
import json
import random
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import liveledger as ext  # noqa: E402

OPS = {ast.Add: 'plus', ast.Sub: 'minus', ast.Mult: 'times', ast.Div: 'divided by'}


def norm(text):
    return ' '.join(text.split())


def fmt(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    as_float = float(value)
    text = repr(as_float)
    if Fraction(text) == value and 'e' not in text:
        return text
    raise ValueError('non-terminating decimal')


def oracle_program(solution: str):
    """Return an ext.Program built from GSM8K calculator annotations, or None."""
    annotations = re.findall(r'<<([^=<>]+)=([^<>]+)>>', solution)
    if not annotations:
        return None
    givens, steps = [], []
    given_of = {}           # value -> given name
    result_of = {}          # value -> most recent step name with this result
    counter = {'s': 0}

    def operand(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = Fraction(str(node.value))
            if value in result_of:
                return result_of[value], value
            if value not in given_of:
                name = f'x{len(givens) + 1}'
                given_of[value] = name
                givens.append((name, fmt(value)))
            return given_of[value], value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left_name, left_value = operand(node.left)
            right_name, right_value = operand(node.right)
            op = OPS[type(node.op)]
            value = {'plus': left_value + right_value, 'minus': left_value - right_value,
                     'times': left_value * right_value}.get(op)
            if op == 'divided by':
                if right_value == 0:
                    raise ValueError('division by zero')
                value = left_value / right_value
            counter['s'] += 1
            name = f's{counter["s"]}'
            steps.append(ext.Step(name, left_name, op, right_name))
            return name, value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            raise ValueError('negative literal')
        raise ValueError('unsupported expression')

    last = None
    for expr, result in annotations:
        expr = expr.replace('x', '*').replace('%', '/100').replace(',', '').replace('$', '').strip()
        try:
            tree = ast.parse(expr, mode='eval').body
            name, value = operand(tree)
        except (SyntaxError, ValueError, ZeroDivisionError):
            return None
        if isinstance(tree, ast.Constant):
            continue  # annotation without an operation
        try:
            if Fraction(result.replace(',', '').strip()) != value:
                return None
        except (ValueError, ZeroDivisionError):
            return None
        result_of[value] = name
        last = name
    if last is None or not givens:
        return None
    program = ext.Program(tuple(givens), tuple(steps), last)
    try:
        ext.validate(program)
    except ext.ProgramError:
        return None
    return program


def program_text(program):
    lines = [f'Initially, {ext.render_givens(program.givens)}.']
    lines += [f'{i}. set {s.target} to {s.left} {s.operation} {s.right}.' for i, s in enumerate(program.steps, 1)]
    lines.append(f'What is {program.query}?')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gsmic-per-base', type=int, default=5)
    args = parser.parse_args()
    import pandas as pd
    gsm8k = {}
    for split in ('train', 'test'):
        df = pd.read_parquet(args.raw / f'gsm8k_{split}.parquet')
        for question, answer in zip(df.question, df.answer):
            gsm8k[norm(question)] = (split, answer)
    rows, report = [], defaultdict(int)

    def add(dataset, pair_id, variant, question, answer, solution, extra):
        program = oracle_program(solution) if solution else None
        if program is not None and ext.evaluate(program) != Fraction(int(answer)):
            program = None
        report[f'{dataset}_oracle_ok'] += program is not None
        rows.append(dict(dataset=dataset, pair_id=pair_id, source_id=f'{pair_id}:{variant}', variant=variant,
                         question=question, answer=int(answer),
                         oracle=program_text(program) if program is not None else None, **extra))

    # GSM-Plus distraction insertion.
    for index, line in enumerate(open(args.raw / 'gsmplus_test.jsonl', encoding='utf-8')):
        record = json.loads(line)
        if record['perturbation_type'] != 'distraction insertion':
            continue
        assert str(record['answer']).strip() == str(record['seed_answer']).strip()
        pair_id = f'gsmplus-{index:05d}'
        add('gsmplus', pair_id, 'clean', record['seed_question'], record['seed_answer'], record['seed_solution'], {})
        add('gsmplus', pair_id, 'dead', record['question'], record['answer'], record['seed_solution'], {})
        report['gsmplus_pairs'] += 1
    # GSM-IC: five variants per base, spread over label combinations.
    rng = random.Random(20261113)
    for name in ('2step', 'mstep'):
        records = json.load(open(args.raw / f'GSM-IC_{name}.json', encoding='utf-8'))
        by_base = defaultdict(list)
        for record in records:
            by_base[record['original_question']].append(record)
        for base_index, base in enumerate(sorted(by_base)):
            variants = by_base[base]
            answer = str(variants[0]['answer']).strip()
            if not re.fullmatch(r'\d+', answer):
                report[f'gsmic_{name}_nonint_bases'] += 1
                continue
            split, solution = gsm8k.get(norm(base), (None, None))
            by_label = defaultdict(list)
            for v in variants:
                by_label[(v['role_label'], v['number_label'], v['sentence_label'])].append(v)
            labels = sorted(by_label)
            rng.shuffle(labels)
            chosen = [rng.choice(by_label[label]) for label in labels[:args.gsmic_per_base]]
            base_id = f'gsmic{name}-{base_index:03d}'
            add('gsmic', base_id, 'clean', base, answer, solution, dict(gsm8k_split=split))
            for k, v in enumerate(chosen):
                labels_text = '/'.join((v['role_label'], v['number_label'], v['sentence_label']))
                add('gsmic', base_id, f'dead{k}', v['new_question'], answer, solution,
                    dict(gsm8k_split=split, labels=labels_text))
            report[f'gsmic_{name}_bases'] += 1
            report[f'gsmic_{name}_variants'] += len(chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    report['rows'] = len(rows)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
