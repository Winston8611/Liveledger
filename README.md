# LiveLedger: code and test data

LiveLedger compresses structured prompts that describe a straight-line arithmetic program.
It applies two compiler analyses: backward dependency slicing removes every assignment
that cannot reach the queried value, and liveness-based renaming stores each new result
under the short name of a value that is no longer needed. Under sequential-update semantics,
the compressed prompt always computes the same answer as the original one.

This repository contains the implementation of the method, every test input used in the
paper, and the scripts that generate these inputs. The core code and the data generators
use only the Python standard library (Python 3.10 or later).

## Contents

```
core/liveledger.py              parser, exact interpreter, slicing, liveness-based renaming
scripts/generate_core.py        generates data/core_suite
scripts/generate_generalization.py  generates data/generalization_suite
scripts/generate_repetition.py  generates data/repetition
scripts/build_nl.py             builds data/natural_language from GSM-Plus, GSM-IC, and GSM8K
scripts/build_prompts.py        renders the model input of every compared method
scripts/validate_data.py        checks the data, the worked example, and the file hashes
scripts/common.py               rendering and exact execution for programs with 2-4 givens
data/                           all test inputs (no model outputs)
metadata/reserved_fingerprints.txt  development programs kept out of the core suite
metadata/SHA256SUMS.json        SHA-256 hash of every file
```

## Quick check

```bash
python scripts/validate_data.py
```

This takes a few seconds. It reproduces the worked example of the paper, executes every
source program and every compressed version of it, checks pairs and counts, and verifies
the file hashes.

## Using the method

```python
import sys; sys.path.insert(0, 'core')
import liveledger as ll

program = ll.parse_prompt(open('example.txt').read())
compact = ll.compress(program, 'liveledger')       # slicing + liveness-based renaming
assert ll.evaluate(compact) == ll.evaluate(program)
print(ll.render(compact, 'liveledger'))
```

A program consists of a line `Initially, a = 13 and b = 6.`, numbered assignments
`1. set c0 to a plus 5.` (operations `plus`, `minus`, `times`, `divided by`), and a query
`What is c5?`. The control methods of the paper are also available: `direct`,
`instruction_only`, `slice_only`, `unique_short`, and `full_unique_short`.

## Test data

| Directory | Rows | Content |
|---|---:|---|
| `data/core_suite` | 4,320 | two givens, `plus` and `times`; 10 data seeds x 432 rows |
| `data/generalization_suite` | 4,320 | 2-4 givens; addition/multiplication, subtraction, exact division, or all four |
| `data/repetition` | 2,560 | four repetition levels x 10 seeds x 2 chain depths x 32 chains |
| `data/natural_language/nl_items.jsonl` | 3,229 | 1,319 GSM-Plus pairs; 99 GSM-IC base problems with 492 distractor variants |

The ten data seeds are 1663, 1693, 1721, 1723, 1733, 1741, 1747, 1753, 1759, and 1777.
For every seed, each structured suite has a short and a long panel and five families:

| Family | Short: core / irrelevant | Long: core / irrelevant | What it tests |
|---|---|---|---|
| `base` | 5-6 / 1 | 12 / 4 | reference chain with a few irrelevant statements |
| `deep` | 9-10 / 2 | 24 / 6 | roughly twice as many dependent steps |
| `irrelevant_heavy` | 6 / 4 | 12 / 12 | as much irrelevant computation as useful computation |
| `fanout` | 7 / 2 | 20 / 6 | values read by several later steps, so more values stay live |
| `ordinary` | 4-5 / 0 | 6 / 0 | simple chain without a distractor version |

Core statements lie on the dependency chain of the query. Irrelevant statements form a
separate chain that starts from the given `b` and has no path to the query. Each paired
family yields a clean program (`variant = "clean"`) and a distractor program
(`variant = "dead"`) in which the irrelevant chain is interleaved with the same core
statements; both have the same answer. Each seed contributes 32 short and 16 long programs
per family, i.e., 4 x 48 pairs plus 48 ordinary programs = 432 rows. The generalization suite
uses the same families and quotas with short-panel sizes 6/1, 10/2, 6/4, 7/2, and 5/0 and
the long-panel sizes above; the docstring of each generator gives its exact rules.

Fields of a structured source row: `prompt` (program text), `answer`, `family`, `panel`,
`variant`, `pair_id` (shared by the clean and distractor rows), `source_id`, `source_seed`,
`canonical_fingerprint` (hash of the program with intermediate names replaced by their
definition order; no two programs share it), and the program in structured form
(`program` in the core suite; `inputs`, `steps`, `output`, `n_givens`, `operator_regime` in
the generalization suite). Each step records whether it is irrelevant (`dead`).

Repetition rows give the prompt of one repetition level, the unrenamed `source_prompt`, the
realized `repeated_fraction`, and the `chain_depth`. Natural-language rows give the
`dataset`, `question`, `answer`, `variant`, `pair_id`, and, when one exists, an `oracle`
program derived from the GSM8K reference solution.

## Regenerating the data

```bash
python scripts/generate_core.py --out work/core_suite
python scripts/generate_generalization.py --out work/generalization_suite
python scripts/generate_repetition.py --out work/repetition
diff -r work/core_suite data/core_suite   # identical
```

Each generator draws candidate programs from a pseudo-random generator seeded by a hash of
(data seed, panel, family, candidate index), checks every candidate with the exact
interpreter, skips candidates whose canonical fingerprint was already accepted, and stops
when the family quota is reached. The rules for each family are documented at the top of
each script. All three generators reproduce the released files byte for byte.

The natural-language items are rebuilt from the public datasets (requires `pandas` and
`pyarrow`):

```bash
python scripts/build_nl.py --raw /path/to/raw --out work/nl_items.jsonl
```

where `/path/to/raw` contains `gsm8k_train.parquet`, `gsm8k_test.parquet`,
`gsmplus_test.jsonl`, `GSM-IC_2step.json`, and `GSM-IC_mstep.json`.

## Model inputs

```bash
python scripts/build_prompts.py --suite core --out work/prompts_core
python scripts/build_prompts.py --suite generalization --out work/prompts_generalization
```

These commands write the exact model inputs of every compared method
(`<out>/<method>/seed<seed>.jsonl`). Every rendered program is executed before it is written,
and the build stops if any answer differs from the source answer.

## Upstream datasets

The natural-language items are taken from GSM-Plus, GSM-IC, and GSM8K. Please cite these
datasets and follow their licenses when using `data/natural_language`.
