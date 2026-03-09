# Co-Training on Bijective and Non-Bijective Inversion: A Detailed Comparative Analysis (v2)

## 1. Experimental Setup

### 1.1 Models

We trained three models from the same base (Qwen2.5-7B-Instruct) using GRPO for 160 steps, differing only in training data:

| Model Name | Training Data | Description |
|---|---|---|
| **Both model** | 50k bijective + 50k non-bijective samples | Trained jointly on functions with exactly 1 preimage (bijective, e.g., `func_00`–`func_23`) and functions with multiple preimages (non-bijective, e.g., `nb_00`–`nb_23`) |
| **NB-only model** | 50k non-bijective samples | Trained exclusively on non-bijective functions |
| **Bij-only model** | 50k bijective samples | Trained exclusively on bijective functions |

All models were trained with identical hyperparameters (lr=1e-6, batch_size=16, n=16 rollouts, GRPO).

### 1.2 Task Description

Each task gives the model a Python function and an output string, then asks it to find N different input strings that produce that output (N is typically 2). The prompt includes the full source code of the function. Here is an example prompt:

```
You are given a code:
[... Python source code of the function ...]

You are told that `main_solution(x)` returns the string "21".
Find 2 DIFFERENT digit-string inputs `x` such that `main_solution(x)` returns exactly that string.
If fewer than 2 valid answers exist, output only the valid answers.
```

**Bijective functions** (e.g., `func_23`) have exactly one valid preimage for each output, so the correct behavior is to find 1 input and recognize no second input exists. **Non-bijective functions** (e.g., `nb_23`) have multiple valid preimages.

### 1.3 Evaluation Sets

We focus our analysis on **out-of-distribution (OOD) evaluations** that test generalization:

| Eval Set | Description | N (# answers requested) | # Samples | OOD? |
|---|---|---|---|---|
| NB more candidates | Non-bijective functions, 3–6 preimages, short strings | 3–6 | 1024 | Yes (more answers than training) |
| NB long outputs | Non-bijective functions, output strings length 6–10 | 2 | 1024 | Yes (longer than training) |
| Lenchange functions | Novel function family (`lenop_*`) that changes string length | 3–5 | 1024 | Yes (unseen function type) |
| Bijective test | Bijective functions, asked for 2 inputs (only 1 exists) | 2 | 2048 | In-dist for Both model, OOD for NB-only |
| NB test | Non-bijective functions, 2 preimages | 2 | 1024 | In-dist for both |

### 1.4 Evaluation Methodology

For each evaluation, we generate 4 rollouts per prompt at temperature 1.0. We report **pass@1** (the framework's built-in metric, which averages per-sample reward scores across rollouts) and conduct qualitative analysis by reading model responses.

---

## 2. Quantitative Results

### 2.1 OOD Evaluation Scores (pass@1)

| Evaluation Set | Both model | NB-only model | Absolute delta |
|---|---|---|---|
| NB more candidates (3–6 preimages) | **0.284** | 0.254 | +0.030 |
| NB long outputs (len 6–10) | **0.237** | 0.210 | +0.027 |
| Lenchange functions (novel family) | **0.117** | 0.051 | +0.066 |
| Bijective test | **0.244** | 0.034 | +0.210 |

On the **in-distribution NB test**, the both model scores 0.332 vs NB-only 0.227 (+0.105).

### 2.2 Differential Analysis (Sample-Level)

We count how many samples each model gets correct (at least 1 correct preimage in any of the 4 rollouts) while the other does not:

| Eval Set | Both wins (only Both correct) | NB-only wins (only NB correct) | Ratio |
|---|---|---|---|
| NB test (in-dist) | 269 | 19 | 14:1 |
| NB more candidates (OOD) | 291 | 25 | 12:1 |
| Lenchange (OOD) | 10 | 0 | ∞ |

The both model wins overwhelmingly in differential comparisons.

### 2.3 Function-Level Accuracy Breakdown

The improvement is **not uniform across all function types**. We break down accuracy by function on the NB test and NB more candidates evaluations:

| Function | Structure | Both (NB test) | NB-only (NB test) | Both (MC) | NB-only (MC) |
|---|---|---|---|---|---|
| `nb_18` | Suffix minimum + shift | 10.9% | 10.9% | 14.8% | 16.9% |
| `nb_19` | Suffix maximum + shift | 12.7% | 11.6% | 14.4% | 15.6% |
| `nb_20` | **Prefix sum** + half-state + shift | **49.4%** | 5.8% | **69.0%** | 14.3% |
| `nb_21` | **Prefix sum** + fold + shift | **46.9%** | 24.6% | **53.0%** | 35.1% |
| `nb_22` | **Prefix sum** + square mod 10 + shift | **39.8%** | 10.8% | **32.5%** | 11.8% |
| `nb_23` | **Prefix sum** + distance to 5 | **74.5%** | 16.3% | **76.5%** | 13.4% |

**Key finding**: The both model's advantage is **entirely concentrated on prefix-sum-based functions** (`nb_20`–`nb_23`), where algebraic inversion via modular arithmetic is possible. On suffix min/max functions (`nb_18`–`nb_19`), both models perform equally poorly. This is a critical nuance: co-training does not provide a universal improvement — it specifically enables algebraic inversion skills for function families where such inversion is structurally feasible.

### 2.4 Answer Format on Bijective Tasks

On bijective tasks (where only 1 preimage exists), the models differ dramatically in output format:

| Behavior | Both model | NB-only model |
|---|---|---|
| Outputs a **single** answer (recognizing only 1 preimage exists) | **66.8%** | 0.0% |
| Outputs a **comma-separated pair** (always trying to find 2) | 33.0% | **99.8%** |

On non-bijective tasks, both models output pairs ~99% of the time, which is correct.

---

## 3. Qualitative Analysis of Reasoning Strategies

### 3.1 Methodology

We classified reasoning strategies by examining model responses for structural features (not just keyword matching). The classification was done as follows:

- **Algebraic inversion**: The response contains at least 2 instances of explicit modular arithmetic expressions (e.g., `(state + d) % 10`, `\pmod{10}`, `mod 10`), indicating the model is solving for inputs using equations.
- **Case analysis**: The response explicitly considers multiple cases (e.g., "Case 1: state = 2, Case 2: state = 8") arising from the non-injectivity of the emit function.
- **Trial and error**: The response tests 2+ specific candidate inputs by forward-computing the function on them.
- **Forward verification**: The response computes the function forward on its candidate answer to verify correctness.

These categories are not mutually exclusive — a single response can use multiple strategies. Classification was performed programmatically across all 4096+ responses per model per evaluation set.

### 3.2 Strategy Distribution

| Strategy | Both model (NB test) | NB-only (NB test) | Both model (Bij test) | NB-only (Bij test) |
|---|---|---|---|---|
| Algebraic inversion | 46.5% | 46.4% | 34.2% | 37.6% |
| Case analysis | **24.0%** | 0.5% | **6.9%** | 13.3% |
| Trial and error | **30.7%** | 8.1% | 12.4% | **29.8%** |
| Forward verification | 6.3% | 5.0% | 1.2% | 1.6% |

**Honest assessment of the data**: Superficially, both models use "algebraic inversion" markers at similar rates (~46%). The keyword-level analysis alone does not fully capture the quality difference. The real difference becomes apparent only through close reading (Section 3.3), where we see the NB-only model often writes algebraic-looking text but applies it incorrectly, while the both model's algebraic steps are more rigorous.

The clearest **quantitative** signal is **case analysis**: the both model uses it 24% of the time on NB tasks vs 0.5% for the NB-only model. This is the dominant reasoning strategy that distinguishes the two models.

### 3.3 Detailed Examples: Same Problem, Different Reasoning

#### Example 1: nb_23 (Prefix sum, emit distance to 5)

**Prompt** (abbreviated):
```
def nb_23(s: str) -> str:
    """Prefix sum, then emit distance to 5. y[i] = |state - 5|."""
    return _state_emit(s, lambda state, digit, i: abs(state - 5))

main_solution(x) returns "21". Find 2 DIFFERENT inputs.
```
**Ground truth**: Known preimages are `"79"` and `"77"`.

**Both model** (gets 1 correct: `"77"`):
```
y[i] = |state - 5|, where state = (state + digit) % 10.

For the first character '2':
  |d₁ mod 10 − 5| = 2
  → d₁ ≡ 3 (mod 10) or d₁ ≡ 7 (mod 10)

For the second character '1':
  |(d₁ + d₂) mod 10 − 5| = 1

  If d₁ = 3: |(3 + d₂) mod 10 − 5| = 1
    → (3 + d₂) mod 10 = 4 or 6 → d₂ = 1 or d₂ = 3

  If d₁ = 7: |(7 + d₂) mod 10 − 5| = 1
    → (7 + d₂) mod 10 = 4 or 6 → d₂ = 7 or d₂ = 9

Possible inputs: "31", "33", "77", "79"
```
**Answer**: `31,77`

**NB-only model** (gets 0 correct):
```
Let's test some possible inputs:

Input 1: "54"
- state = abs(0 - 5) % 10 = 5    ← WRONG: confuses emit with update!
  Emit: _c(5) = "5"
- state = abs(5 - 5) % 10 = 0
  Emit: _c(0) = "0"
Output: "50", not "21".

Input 2: "46"
- state = abs(0 - 5) % 10 = 5    ← Same error repeated
  Emit: "5"  ...
Output: "50", not "21".

[continues trying 8+ inputs, all producing "50" due to the same systematic error]
```
**Answer**: `52,47`

**Analysis**: The both model does backward algebraic inversion with explicit case analysis (2 cases for each position). The NB-only model attempts forward trial-and-error but has a **systematic misunderstanding**: it applies the emit function (`abs(state - 5)`) as if it were the state update rule, when in fact the update rule is `state = (state + digit) % 10`. This causes every forward computation to produce `"50"`. After many failed attempts, it outputs a guess.

#### Example 2: nb_22 (Prefix sum, emit state² mod 10, shift +1)

**Prompt** (abbreviated):
```
def nb_22(s: str) -> str:
    """Prefix sum, then emit state squared mod 10, then add 1.
    y[i] = (state² mod 10) + 1 (mod 10)."""

main_solution(x) returns "152". Find 2 DIFFERENT inputs.
```
**Ground truth**: Known preimages are `"083"` and `"081"`.

**Both model** (gets 1 correct: `"081"`):
```
For y[0] = 1: (state² mod 10) + 1 mod 10 = 1 → state² ≡ 0 → state = 0
For y[1] = 5: (state² mod 10) + 1 mod 10 = 5 → state² ≡ 4 → state = 2 or 8
For y[2] = 2: (state² mod 10) + 1 mod 10 = 2 → state² ≡ 1 → state = 1 or 9

Testing x = "021": state sequence 0→2→3, emits (0+1)=1, (4+1)=5, (9+1)=0 ✗ [third digit wrong]
Testing x = "081": state sequence 0→8→9, emits (0+1)=1, (4+1)=5, (1+1)=2 ✓
```
**Answer**: `021,081`

**NB-only model** (gets 0 correct):
```
Step 1: Reverse the shift: subtract 1 from each digit of "152" → "041"   ✓ [correct]
Step 2: Find inputs whose _state_emit produces "041"

Testing "041": state=0, emit 0²%10=0 → "0" ✓
  state=(0+4)%10=4, emit 4²%10=6 → "6" (not "4") ✗

Testing "042": same first digit issue...
Testing "040": same issue...
Testing "031": state=(0+3)%10=3, emit 3²%10=9 → "9" (not "4") ✗

[repeats "041" again, gets stuck in a loop]
```
**Answer**: `041,032`

**Analysis**: The NB-only model correctly reverses the shift operation (subtract 1) but fails at the harder step of inverting the inner emit function. It uses forward trial-and-error, repeatedly testing candidates without algebraically solving for the correct inputs. The both model, by contrast, solves for the required state at each position algebraically (state² ≡ 4 mod 10 → state = 2 or 8), which directly gives it the branching options.

#### Example 3: nb_20 (Prefix sum, emit state//2, shift +3)

**Prompt** (abbreviated):
```
def nb_20(s: str) -> str:
    """Prefix sum, then emit half-state with a small drift.
    y[i] = floor(state / 2) + 3 (mod 10)."""

main_solution(x) returns "67". Find 2 DIFFERENT inputs.
```
**Ground truth**: Known preimages are `"71"` and `"62"`.

**Both model** (gets 1 correct: `"62"`):
Performs algebraic inversion with explicit modular arithmetic and case analysis to find inputs `"62"` and `"72"`.

**NB-only model** (gets 0 correct):
```
Step 1: Reverse the shift: subtract 3 from each digit of "67" → "34"   ✓ [correct]
Step 2: Test inputs:

Input "34": state=3, emit 3//2=1 → "1" (not "3") ✗
Input "35": state=3, emit 3//2=1, state=(3+5)=8, emit 8//2=4 → "14" [wait, says "67"]

[NB-only model's forward computation is WRONG here:
 it claims "35" maps to "67" but its own step-by-step shows emit produces "1" then "4",
 which would be "14" not "34", and after adding shift+3 would be "47" not "67"]
```
**Answer**: `34,35`

**Analysis**: The NB-only model correctly reverses the shift but then makes a **forward computation error** when verifying its trial inputs. It claims `"35"` maps to the target output but the arithmetic doesn't actually check out. This pattern — correct shift reversal followed by erroneous forward verification — is the dominant error mode on `nb_20`–`nb_22`.

---

## 4. Error Taxonomy

### 4.1 Methodology

We categorized errors by reading hundreds of responses and identifying recurring patterns. The categories below are manually defined based on observed patterns, then counted programmatically using regex-based heuristics across all responses.

### 4.2 NB-only Model's Dominant Error Patterns

The NB-only model exhibits three dominant error types on prefix-sum functions:

**Error Type 1: Correct Shift Reversal, Failed Inner Inversion** (dominant on `nb_22`, ~91% of errors)

The model correctly identifies the outermost `_shift_all` operation and reverses it (e.g., subtracting 1 from each digit). However, it cannot algebraically invert the inner `_state_emit` function. Instead, it falls into forward trial-and-error, testing candidate inputs one by one. Since the search space is large and the model doesn't systematically constrain its search, most trials fail.

**Error Type 2: Function Semantics Confusion** (dominant on `nb_23`, ~30% of errors)

The model confuses the emit function with the state update function. For `nb_23`, the state update is `state = (state + digit) % 10` but the model writes `state = abs(state - 5)` (which is the emit rule). This causes all forward computations to produce the same wrong output regardless of input.

**Error Type 3: Computation Errors During Forward Verification** (common on `nb_20`, `nb_21`)

The model correctly identifies the reversal steps but makes arithmetic errors when computing the function forward on candidate inputs. For example, it may claim an input produces the target output when its own step-by-step computation shows a different result, then proceeds to output that incorrect input as an answer.

### 4.3 Both Model's Error Patterns

The both model's errors are qualitatively different:

- **Incomplete case analysis**: The model identifies the correct algebraic framework but misses some branches (e.g., finds 2 out of 4 possible inputs).
- **Algebraic errors in modular arithmetic**: Makes a mistake in solving modular equations (e.g., `(7 + d₂) mod 10 = 4` → writes `d₂ = 6` instead of `d₂ = 7`).
- **Similar suffix min/max failures**: On `nb_18`/`nb_19`, the both model fails at the same rate as the NB-only model because suffix min/max inversion requires different reasoning (ordering constraints) that neither model has learned well.

---

## 5. Bijective Tasks: Format Adaptation and Structural Recognition

### 5.1 How the Both Model Handles Bijective Functions

On bijective tasks, the both model has learned to recognize that only 1 preimage exists and often outputs a **single answer** (66.8% of responses). It does this by performing the algebraic inversion, finding the unique input, and explicitly stating "there are no other possible inputs."

Example (func_23, output "43", true preimage "02"):
```
The function performs right-to-left running sum with +1 drift.
Inverse: subtract the next output digit and 1.

For last digit '3': input = (3 - 1) mod 10 = 2
For next digit '4': input = (4 - 1) mod 10 = 3

The input is "32". Since the process is deterministic and direct,
there are no other valid inputs.
```
**Answer**: `32`

Note: This particular answer is actually wrong (the true preimage is "02"), revealing that even when the both model uses the correct inversion framework, computation errors still occur.

### 5.2 How the NB-only Model Handles Bijective Functions

The NB-only model **never** outputs a single answer on bijective tasks (0.0%). It always produces pairs, using strategies such as:

1. **Mutate-last-digit**: Finds one candidate, then increments or decrements the last digit to produce a second. Example answer: `"4472,4473"`.
2. **Fabricated pair**: Produces two candidates that appear unrelated, neither of which is correct.

Error breakdown on bijective tasks (2000 responses):

| Category | Both model | NB-only model |
|---|---|---|
| Correct single answer | **21.1%** | 0.0% |
| Correct preimage found in pair | 4.9% | 6.9% |
| Both answers wrong, pair format | 28.1% | **92.9%** |
| Wrong single answer | 45.6% | 0.0% |

---

## 6. Discussion

### 6.1 What Co-Training Does and Does Not Do

**What it does**: Co-training with bijective functions appears to strengthen **algebraic inversion skills** specifically for prefix-sum-based functions. The both model learns to invert emission functions by solving modular equations and performing case analysis when the emission function is non-injective (e.g., `|state - 5| = 2` has two solutions). It also learns **output format flexibility** — recognizing when a function is bijective and producing only 1 answer.

**What it does NOT do**:
- It does not improve performance on suffix min/max functions (`nb_18`, `nb_19`), which require fundamentally different reasoning about orderings rather than modular arithmetic.
- It does not teach the model to solve inverse tasks in a different prompt format (pass@1 = 0 on the inverse test set).
- It does not eliminate computation errors — the both model still makes arithmetic mistakes ~45-50% of the time.

### 6.2 Honest Assessment of the "Reasoning Hacking" Thesis

The original thesis proposes that the NB-only model collapses into a "mutate-and-enumerate" strategy that is structurally incompatible with bijective functions.

**Evidence supporting this thesis:**
- On bijective tasks, the NB-only model outputs pairs 99.8% of the time (vs 33% for both model), suggesting it has learned a rigid "always output two answers" strategy.
- The NB-only model makes more trial-and-error attempts on bijective tasks (29.8%) compared to NB tasks (8.1%), suggesting it struggles when its default strategy fails.
- The both model significantly outperforms the NB-only model even on NB tasks (+46% in-dist, +12-13% OOD), suggesting genuine reasoning improvement.

**Nuances and caveats:**
- The "simple mutate-last-digit" pattern (explicitly incrementing the last digit) is actually rare in the data: only 0.0–1.3% of responses across both models. The NB-only model's second candidate typically comes from a failed inversion attempt, not from explicitly mutating a found answer.
- The NB-only model does not completely lack algebraic reasoning — it uses algebraic markers at a similar rate to the both model (~46%). The difference is in the *quality and correctness* of that reasoning, particularly in case analysis (24% vs 0.5%) and function semantics understanding.
- The improvement is **function-family-specific**: concentrated entirely on prefix-sum functions where algebraic inversion applies. This suggests the mechanism is not general "better reasoning" but specifically "better modular arithmetic inversion."
- A more precise description might be: the NB-only model learned a **shallow inversion procedure** (reverse the shift, then guess) that breaks down when the inner function is complex, while the both model learned a **deeper algebraic inversion** that handles the inner function by solving modular equations.

### 6.3 Why Case Analysis Is the Key Differentiator

The most robust quantitative signal distinguishing the two models is the use of **case analysis** (24% vs 0.5% on NB tasks, 20.7% vs 0.8% on OOD more-candidates). Case analysis is specifically required when inverting non-injective emit functions (e.g., `|state - 5| = 2` → state = 3 or 7; `state² mod 10 = 4` → state = 2 or 8).

We hypothesize that bijective training teaches the model to carefully track state through sequential operations (since bijective functions have unique inversions that demand exact computation). When this skill transfers to non-bijective tasks, the model is able to additionally identify branching points where the emit function is non-injective, leading to case analysis.

---

## 7. Summary of Findings

1. **The both model outperforms the NB-only model across all evaluations**, including OOD sets. The advantage ranges from +12% (NB more candidates) to +618% (bijective test).

2. **The improvement is concentrated on prefix-sum-based functions** (`nb_20`–`nb_23`) where algebraic inversion via modular arithmetic is feasible. On suffix min/max functions (`nb_18`–`nb_19`), both models perform equally.

3. **The key reasoning skill gained from co-training is case analysis** (24% vs 0.5%), which enables the model to handle non-injective emission functions by systematically enumerating all valid states at each position.

4. **The NB-only model's dominant failure mode is "correct shift reversal, failed inner inversion"** — it successfully reverses outer shift operations but cannot algebraically invert the inner state-emit function, falling back to forward trial-and-error that rarely finds the correct input.

5. **On bijective tasks, the NB-only model is rigidly locked into producing pairs** (99.8%), while the both model has learned output format flexibility (66.8% singles on bijective, 98.1% pairs on non-bijective).

6. **Neither model achieves high absolute accuracy** — even the both model gets less than 50% correct on most evaluations. The primary difference is in the *type* of reasoning employed, which could compound with longer training or larger models.

---

## Appendix: Experimental Details

- **Base model**: Qwen2.5-7B-Instruct
- **Training**: GRPO, 160 steps, batch size 16, n=16 rollouts per prompt, lr=1e-6, param_offload + optimizer_offload
- **Evaluation**: n=4 rollouts per prompt, temperature=1.0
- **Checkpoint**: Step 160 for both and NB-only models
- **Functions**: 24 bijective (`func_00`–`func_23`) and 24 non-bijective (`nb_00`–`nb_23`) from the v3 atomic function library
- **HTML reports**: Available in `analysis/htmls/` for manual inspection of individual responses
- **Rollout data**: Available in `analysis/rollouts/{both_model,nb_only_model}/` as parquet files
