# Gate control test, 15 November 2026

A keyed control test of the gate in `abhaya_gate(1).ipynb`, and CPU checks of its kernel.
Status of the claim "the gate prevents false statements": **not established by anything here.**
The single-GPU run on the notebook's own hardware class has not been made yet.

## What is in this folder

| File | What it is |
|---|---|
| `gate_control_colab.py` | One Colab cell. Paste the whole file and run it on one GPU. No token is needed. |
| `kernel_checks.py`, `kernel_checks.json` | NumPy float32 form of the notebook's two kernels and the checks run on it. No model is loaded. |
| `cpu_run_qwen2.5-3b-instruct_bf16.json` | The cell run on this CPU with the notebook's model. Not a GPU record. |
| `cpu_run_qwen2.5-1.5b-instruct_f32.json`, `cpu_run_qwen2.5-0.5b-instruct_f32.json` | The same cell on two smaller models. |

## What the v8 sheets say about the gate

Nothing. `v8_final_provisional_run_B1_2026-11-15T092041Z.json` and
`v8_final_provisional_run_B4_2026-11-15T092049Z.json` hold 95 distinct key paths. None is a
logit, a token, a question, an answer, a model name, or a threshold. The sheets measure
arithmetic, memory traffic, power and clock on one and four boards.

The v8 selection step does bear on the gate's formula, in one way. It keeps the map
`A(y) = mu + (y - mu) exp(-xi)` and refuses `R(y) = y exp(-xi)`. `kernel_checks.py` runs the
same add-5 test, in float32, on the same seed-11 series:

| Map | Drift under a shift of 5 |
|---|---|
| v8 map A, selected | 9.5367431640625e-07, which is the logged value |
| v8 map R, rejected | 2.757575511932373, which is the logged value |
| `sri.omlang.abhaya.abhaya_damping` | 2.757575511932373 |
| Notebook kernel, cell 0 | 555.5492553710938 |

The repository's damping function is the map v8 rejected. The notebook's kernel is a third
map: it multiplied that series by 112.1, where map A multiplies departures by 0.4485. A
softmax ignores a uniform shift, so this drift does not by itself change a token. It shows
that v8 did not test, and does not select, the notebook's formula.

## What the kernel does to a row of logits

- It never changed the largest logit: 0 of 288 rows, both kernels, 17 to 151,936 entries.
  A greedy decode therefore picks the same token with or without it.
- On random logits it raised the leading probability from between 7.6% and 37% to about 100%.
  That is the notebook's own cell 0 result, entropy 6.86 nats in and 0 out.
- Its score reads how peaked the row is. A wrong token 22 logits ahead scores 1.0.

## The control test

Forty two-option questions with textbook keys, the key first in twenty. Prompts are the
notebook's two formats. Decoding is greedy. No noise is injected. Each answer ends at its
first full stop in every arm, so the trigger is not counted for firing on later text.

| Arm | What runs |
|---|---|
| plain | The model alone |
| plain again | The same, a second time. This is the noise floor. |
| kernel only | The v4 kernel on every step's logits, no re-prompt |
| true reference always | The reference sentence in the prompt, no gate |
| false reference always | The same sentence naming the other option, no gate |
| gate + true reference | `abhaya_v4_generate_clean` from cell 9 |
| gate + false reference | The same, handed the false sentence |

Conditions fixed before any run, printed PASS or FAIL by the cell:

- **T1** The kernel changes at least one chosen token and the kernel-only arm beats plain.
- **T2** The trigger fires more often on questions the plain model gets wrong than on those
  it gets right, one-sided Fisher p below 0.05.
- **T2b** The same, reading the trigger's condition at the token that names the answer and
  setting aside its three-token rule. This is the most favourable reading of the trigger.
- **T3** Gate + true reference beats plain, and beats the reference with no gate, each by a
  paired sign test with p below 0.05.
- **T4** Gate + false reference scores no lower than plain.

The gate counts as a detector of wrong answers only if T2 or T2b passes. It counts as more
than a re-prompt with a supplied sentence only if T1 or T3 passes.

## CPU results

These were run on a 4-core CPU with the PyTorch form of the kernel, which matches the NumPy
form to 7e-11. The Triton kernel needs a GPU and was not run. Correct answers out of 40:

| | Qwen2.5-3B, bfloat16 | Qwen2.5-1.5B, float32 | Qwen2.5-0.5B, float32 |
|---|---|---|---|
| plain | 28 | 25 | 23 |
| plain again | 29 | 25 | 23 |
| kernel only | 29 | 25 | 23 |
| true reference always | 38 | 40 | 19 |
| false reference always | 11 | 1 | 13 |
| gate + true reference | 32 | 30 | 21 |
| gate + false reference | 26 | 19 | 18 |
| Steps where the kernel changed the chosen token | 0 of 632 | 0 of 409 | 0 of 712 |
| Trigger fired, plain answer wrong | 5 of 12 | 5 of 15 | 12 of 17 |
| Trigger fired, plain answer right | 5 of 28 | 6 of 25 | 11 of 23 |
| Fisher p for that split | 0.117 | 0.387 | 0.132 |
| Wrong answers that passed with no trigger | 7 | 10 | 5 |
| Firings at the two answer words | 1 of 10 | 5 of 11 | 0 of 23 |
| False reference: output named the false answer after firing | 6 of 8 | 11 of 11 | 17 of 23 |
| AUROC of the collision score at the answer token | 0.890, p = 0.00005 | 0.589, p = 0.18 | 0.481, p = 0.58 |
| T1, T2, T2b, T3, T4 | all FAIL | all FAIL | all FAIL |

Reading, limited to these runs:

1. The kernel changed no token on any of 1,753 steps. On the 1.5B model it never rescaled
   the logits at all.
2. Every answer the gate corrected came from the re-prompt that carries the reference. The
   reference with no gate scored higher on the two larger models: 38 against 32 and 40
   against 30.
3. Handed a false sentence, the gate's output repeated it in 34 of 42 firings across the
   three models.
4. The trigger let most wrong answers through on the two larger models, 7 of 12 and 10 of 15.
5. One result favours the idea behind the trigger. On the 3B model the collision score at
   the answer token was lower for wrong answers, mean 0.73 against 0.95. The trigger as
   built does not use it there: its condition held at 1 of 12 wrong answers. The two
   smaller models did not show the separation.

On the notebook's own three questions the 3B model gave the same sentence with the true
reference and no gate as with the full gate on the first two, character for character.
With a false reference the full gate wrote that diamond conducts heat five times worse than
copper, and that nitrogen boils at 97.36 K, above oxygen.

## Limits

- The 3B CPU run is in bfloat16 and is not repeatable to the token. Plain run twice gave
  the same text on 34 of 40 and the same mark on 39. The float32 runs repeated exactly.
- bfloat16 logits near 40 move in steps of 0.25. The trigger's ratio test, below 1.6, then
  passes only for a gap of 0 or 0.25 between the two leading logits.
- Forty questions, one prompt format, greedy decoding, three sizes of one model family.
- The questions were written for this test. The notebook's thresholds were not retuned.
- The scored 20-question exam of 15 November
  (`docs/benchmark-results/nov15-foundation/language_exam_15nov.json`, on the branch
  `cursor/nov15-foundation-report-862c`) recorded NOT_MET for the same claim.

## How to run

Colab, one GPU: paste `gate_control_colab.py` into one cell and run it. It prints the table,
the five tests, and a `JSON_SUMMARY` line, and writes `abhaya_gate_control_<UTC>.json`.

CPU checks: `python docs/benchmark-results/gate-control-15nov/kernel_checks.py`
