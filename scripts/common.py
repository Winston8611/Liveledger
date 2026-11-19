"""Rendering, exact execution, and prompt transforms for the generalization and repetition sets.

These programs may use two to four givens and the operations plus, minus, times, and
divided_by (exact integer division); the operation word is rendered verbatim in the prompt.
"""
import json
import re
import sys
from pathlib import Path

CONCISE = ('Write concise working. Do not restate the question, instructions, or initial '
           'inputs. Use short lines of calculations or useful state values, without prose.')
FORMAT = ('Final answer format: put the answer on a new last line beginning with '
          'exactly four hash signs and ONE SPACE, then the integer. '
          'Do not put the integer on a separate line. Use no other hash-sign lines. '
          'Stop immediately after the integer.')

def with_answer_format(prompt):
    """Append the shared output instructions used by every structured prompt."""
    return prompt.rstrip() + '\n\n' + CONCISE + '\n\n' + FORMAT

SEEDS = (1663, 1693, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777)
OPERATIONS = ('plus', 'minus', 'times', 'divided_by')
METHODS = ('direct', 'slice_only', 'unique_short', 'liveledger')
_ASSIGNMENT = re.compile(r'^[0-9]+\. ')
_STEP = re.compile(r'set ([A-Za-z][A-Za-z0-9_]*) to ([A-Za-z][A-Za-z0-9_]*) (plus|minus|times|divided_by) ([A-Za-z][A-Za-z0-9_]*|[0-9]+)\.')


def load_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def apply_operation(operation, left, right):
    if operation == 'plus':
        return left + right
    if operation == 'minus':
        return left - right
    if operation == 'times':
        return left * right
    if operation == 'divided_by':
        if right == 0 or left % right:
            raise ValueError('division must be exact and nonzero')
        return left // right
    raise ValueError(operation)


def initial_text(inputs):
    entries = [f'{name} = {value}' for name, value in inputs]
    if len(entries) == 2:
        body = ' and '.join(entries)
    else:
        body = ', '.join(entries[:-1]) + ', and ' + entries[-1]
    return f'Initially, {body}.'


def render_program(program, steps=None, output=None, aliases=None):
    steps = program['steps'] if steps is None else steps
    output = program['output'] if output is None else output
    aliases = {} if aliases is None else aliases
    lines = ['Track the named quantities and compute the requested value.', initial_text(program['inputs'])]
    for index, step in enumerate(steps, 1):
        left = aliases.get(step['left'], step['left'])
        right = step['right'] if isinstance(step['right'], int) else aliases.get(step['right'], step['right'])
        target = aliases.get(step['target'], step['target'])
        lines.append(f"{index}. set {target} to {left} {step['operation']} {right}.")
    lines.extend((f'What is {aliases.get(output, output)}?',
                  'Process the assignments in order. Use the latest value of each quantity on the right before updating the quantity on the left.',
                  'Show the calculation and end with #### followed by the integer only.'))
    return with_answer_format('\n'.join(lines))


def split_program(prompt):
    lines = prompt.split('\n')
    index = [i for i, line in enumerate(lines) if _ASSIGNMENT.match(line)]
    if not index or index != list(range(index[0], index[-1] + 1)):
        raise ValueError('assignment block is not contiguous')
    return lines[:index[0]], lines[index[0]:index[-1] + 1], lines[index[-1] + 1:]


def execute_sequential(prompt):
    """Execute plus/minus/times/exact-division prompts with arbitrary given counts."""
    initial_lines = [line for line in prompt.splitlines() if line.startswith('Initially, ')]
    queries = re.findall(r'What is ([A-Za-z][A-Za-z0-9_]*)\?', prompt)
    if len(initial_lines) != 1 or len(queries) != 1:
        return None
    values = {name: int(value) for name, value in re.findall(r'([A-Za-z][A-Za-z0-9_]*) = ([0-9]+)', initial_lines[0])}
    if len(values) < 2:
        return None
    try:
        _, block, _ = split_program(prompt)
    except ValueError:
        return None
    for position, line in enumerate(block, 1):
        number, _, body = line.partition('. ')
        match = _STEP.fullmatch(body)
        if number != str(position) or match is None:
            return None
        target, left, operation, right = match.groups()
        if left not in values or (not right.isdigit() and right not in values):
            return None
        operand = int(right) if right.isdigit() else values[right]
        try:
            values[target] = apply_operation(operation, values[left], operand)
        except (ValueError, ZeroDivisionError):
            return None
    return values.get(queries[0])


def answer_slice(program):
    selected = {program['output']}
    for step in reversed(program['steps']):
        if step['target'] in selected:
            selected.add(step['left'])
            if isinstance(step['right'], str):
                selected.add(step['right'])
    return [step for step in program['steps'] if step['target'] in selected]


def short_name(index):
    result = ''
    while True:
        index, digit = divmod(index, 26)
        result = chr(97 + digit) + result
        if index == 0:
            return result
        index -= 1


def transform_program(program, method):
    if method not in METHODS:
        raise ValueError(method)
    steps = program['steps'] if method == 'direct' else answer_slice(program)
    if method in ('direct', 'slice_only'):
        return render_program(program, steps=steps)
    aliases = {name: name for name, _ in program['inputs']}
    occupants = dict(aliases)
    rewritten = []
    next_name = 2
    input_names = {name for name, _ in program['inputs']}
    for index, step in enumerate(steps):
        left = aliases[step['left']]
        right = step['right'] if isinstance(step['right'], int) else aliases[step['right']]
        if method == 'unique_short':
            slot = short_name(next_name)
            while slot in input_names:
                next_name += 1
                slot = short_name(next_name)
            next_name += 1
        else:
            required = {program['output']}
            for future in steps[index + 1:]:
                required.add(future['left'])
                if isinstance(future['right'], str):
                    required.add(future['right'])
            slot = next((name for name in 'abcdefgh' if occupants.get(name) not in required), None)
            if slot is None:
                raise ValueError('more than eight simultaneously live values')
            old = occupants.get(slot)
            if old is not None:
                aliases.pop(old, None)
            occupants[slot] = step['target']
        aliases[step['target']] = slot
        rewritten.append(dict(step, target=slot, left=left, right=right))
    # Rewritten steps already use final names; render without applying aliases again.
    return render_program(program, steps=rewritten, output=aliases[program['output']])


def repeated_line_fraction(prompt):
    _, block, _ = split_program(prompt)
    bodies = [line.partition('. ')[2] for line in block]
    seen = set()
    repeated = 0
    for body in bodies:
        repeated += body in seen
        seen.add(body)
    return repeated / len(block) if block else 0.0
