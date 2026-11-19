"""LiveLedger: semantics-preserving compression of straight-line arithmetic prompts.

A program declares named givens, a numbered list of assignments `set T to L OP R.`
(OP in plus / minus / times / divided by; operands are names or decimal constants), and one
query. `compress` applies the two passes of the method:

1. Backward dependency slicing (`slice_program`) keeps only the assignments whose values can
   reach the query; givens that are no longer read are dropped as well.
2. Liveness-based renaming (`rename` with method 'liveledger') keeps a list of short name
   slots a, b, c, ...  Live givens take the first slots in declaration order; each
   assignment reads its operands first and then writes its result to the first slot whose
   occupant is dead after the step, opening a new slot only when every slot is live.

`evaluate` executes a program exactly under sequential-update semantics, so every
transformation can be checked: evaluate(compress(p, m)) == evaluate(p) for every method m.
The control methods used in the experiments are also provided: 'direct', 'instruction_only',
'slice_only', 'unique_short' (slicing + one fresh short name per assignment), and
'full_unique_short' (fresh short names without slicing).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

HEADER = 'Track the named quantities and compute the requested value.'
UPDATE_RULE = ('Process the assignments in order. Use the latest value of each quantity '
               'on the right before updating the quantity on the left.')
SHOW = 'Show the calculation and end with #### followed by the integer only.'
INSTRUCTION_ONLY = 'Ignore irrelevant operations and keep a compact ledger of values needed later.'
CONCISE = ('Write concise working. Do not restate the question, instructions, or initial '
           'inputs. Use short lines of calculations or useful state values, without prose.')
FORMAT = ('Final answer format: put the answer on a new last line beginning with '
          'exactly four hash signs and ONE SPACE, then the integer. '
          'Do not put the integer on a separate line. Use no other hash-sign lines. '
          'Stop immediately after the integer.')
METHODS = ('liveledger', 'slice_only', 'unique_short', 'full_unique_short', 'instruction_only', 'direct')
OPERATIONS = ('plus', 'minus', 'times', 'divided by')
NAME = r'[A-Za-z][A-Za-z0-9_]*'
NUMBER = r'[0-9]+(?:\.[0-9]+)?(?:/[1-9][0-9]*)?'
_STEP = re.compile(rf'set ({NAME}) to ({NAME}|{NUMBER}) (plus|minus|times|divided by) ({NAME}|{NUMBER})\.')
_GIVEN = re.compile(rf'({NAME}) = ({NUMBER})')


class ProgramError(ValueError):
    pass


def is_constant(token: str) -> bool:
    return re.fullmatch(NUMBER, token) is not None


@dataclass(frozen=True)
class Step:
    target: str
    left: str
    operation: str
    right: str

    @property
    def sources(self):
        return tuple(x for x in (self.left, self.right) if not is_constant(x))


@dataclass(frozen=True)
class Program:
    givens: tuple  # ((name, value_string), ...)
    steps: tuple
    query: str


def parse_givens(text: str):
    """'a = 13 and b = 6' or 'x = 1, y = 2, and z = 3' or 'x = 5'."""
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', text.strip())
    givens = []
    for part in parts:
        match = _GIVEN.fullmatch(part.strip())
        if match is None:
            raise ProgramError(f'bad given: {part!r}')
        givens.append((match[1], match[2]))
    return tuple(givens)


def parse_prompt(prompt: str) -> Program:
    initial = re.findall(r'^Initially, (.+)\.$', prompt, flags=re.MULTILINE)
    queries = re.findall(rf'^What is ({NAME})\?$', prompt, flags=re.MULTILINE)
    if len(initial) != 1 or len(queries) != 1:
        raise ProgramError('ambiguous givens or query')
    givens = parse_givens(initial[0])
    lines = re.findall(r'^([0-9]+)\. (.+)$', prompt, flags=re.MULTILINE)
    steps = []
    for position, (number, line) in enumerate(lines, 1):
        match = _STEP.fullmatch(line)
        if int(number) != position or match is None:
            raise ProgramError(f'unsupported instruction {line!r}')
        steps.append(Step(*match.groups()))
    program = Program(givens, tuple(steps), queries[0])
    validate(program)
    return program


def validate(program: Program) -> None:
    defined = set()
    for name, _ in program.givens:
        if name in defined:
            raise ProgramError('duplicate given')
        defined.add(name)
    for step in program.steps:
        if step.target in defined or not set(step.sources) <= defined:
            raise ProgramError('undefined source or duplicate assignment')
        defined.add(step.target)
    if not program.steps or program.query not in defined:
        raise ProgramError('missing instructions or query')


def evaluate(program: Program):
    """Exact sequential-update semantics; None for division by zero."""
    values = {name: Fraction(value) for name, value in program.givens}
    for step in program.steps:
        left = Fraction(step.left) if is_constant(step.left) else values[step.left]
        right = Fraction(step.right) if is_constant(step.right) else values[step.right]
        if step.operation == 'plus':
            result = left + right
        elif step.operation == 'minus':
            result = left - right
        elif step.operation == 'times':
            result = left * right
        else:
            if right == 0:
                return None
            result = left / right
        values[step.target] = result
    return values[program.query]


def slice_program(program: Program) -> Program:
    """Backward dependency slicing; unused givens are dropped as well."""
    needed = {program.query}
    kept = []
    for step in reversed(program.steps):
        if step.target in needed:
            needed.discard(step.target)
            needed.update(step.sources)
            kept.append(step)
    givens = tuple((n, v) for n, v in program.givens if n in needed)
    if not givens and program.givens:
        givens = program.givens[:1]  # keep one given so that the program still has an initial line
    return Program(givens, tuple(reversed(kept)), program.query)


def short_name(index: int) -> str:
    result = ''
    while True:
        index, digit = divmod(index, 26)
        result = chr(97 + digit) + result
        if index == 0:
            return result
        index -= 1


def live_sets(program: Program):
    """L[j] = names defined after step j that are read later (or the query), j = 0..n."""
    n = len(program.steps)
    future = [set() for _ in range(n + 1)]
    future[n] = {program.query}
    for j in range(n - 1, -1, -1):
        future[j] = future[j + 1] | set(program.steps[j].sources)
    defined = [name for name, _ in program.givens]
    lives = [set(defined) & future[0]]
    for j, step in enumerate(program.steps, 1):
        defined.append(step.target)
        lives.append(set(defined) & future[j])
    return lives


def rename(program: Program, method: str) -> Program:
    if method in ('direct', 'instruction_only', 'slice_only'):
        return program
    alias = {}
    givens = []
    for index, (name, value) in enumerate(program.givens):
        alias[name] = short_name(index)
        givens.append((alias[name], value))
    steps = []
    if method in ('unique_short', 'full_unique_short'):
        offset = len(program.givens)
        for index, step in enumerate(program.steps):
            left = step.left if is_constant(step.left) else alias[step.left]
            right = step.right if is_constant(step.right) else alias[step.right]
            alias[step.target] = short_name(index + offset)
            steps.append(Step(alias[step.target], left, step.operation, right))
        return Program(tuple(givens), tuple(steps), alias[program.query])
    assert method == 'liveledger'
    lives = live_sets(program)
    slots = [alias[name] for name, _ in program.givens]
    occupant = {alias[name]: name for name, _ in program.givens}
    for j, step in enumerate(program.steps, 1):
        left = step.left if is_constant(step.left) else alias[step.left]   # resolve operands first
        right = step.right if is_constant(step.right) else alias[step.right]
        live = lives[j]
        slot = next((s for s in slots if occupant.get(s) not in live), None)
        if slot is None:  # every slot holds a live value: open the next one
            slot = short_name(len(slots))
            slots.append(slot)
        old = occupant.get(slot)
        if old is not None:
            alias.pop(old, None)
        occupant[slot] = step.target
        alias[step.target] = slot
        steps.append(Step(slot, left, step.operation, right))
    return Program(tuple(givens), tuple(steps), alias[program.query])


def render_givens(givens) -> str:
    parts = [f'{n} = {v}' for n, v in givens]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f'{parts[0]} and {parts[1]}'
    return ', '.join(parts[:-1]) + ', and ' + parts[-1]


def render(program: Program, method: str) -> str:
    lines = [HEADER, f'Initially, {render_givens(program.givens)}.']
    lines.extend(f'{i}. set {s.target} to {s.left} {s.operation} {s.right}.' for i, s in enumerate(program.steps, 1))
    lines.extend((f'What is {program.query}?', UPDATE_RULE, SHOW))
    if method == 'instruction_only':
        lines.append(INSTRUCTION_ONLY)
    return '\n'.join(lines) + '\n\n' + CONCISE + '\n\n' + FORMAT


def compress(program: Program, method: str) -> Program:
    if method not in METHODS:
        raise ValueError(method)
    if method in ('liveledger', 'slice_only', 'unique_short'):
        program = slice_program(program)
    return rename(program, method)


def build_prompt(program: Program, method: str) -> str:
    return render(compress(program, method), method)

