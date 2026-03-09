# Co-Training on Structurally Diverse Inversion Tasks: How Bijective Functions Teach Algebraic Reasoning

## 1. Introduction and Motivation

### 1.1 Research Question

When training language models with reinforcement learning on function inversion tasks, does the composition of the training data affect the *type of reasoning* the model learns — not just its accuracy?

Specifically: if a model is trained only on **non-bijective** functions (many-to-one, where multiple inputs produce the same output), it could learn a shortcut — find one correct input, then slightly perturb it to produce a second candidate. This shortcut exploits the many-to-one structure but does not require genuine understanding of the function. We ask whether **co-training with bijective functions** (one-to-one, where each output has exactly one preimage) prevents this shortcut by forcing the model to develop genuine algebraic inversion skills.

### 1.2 Summary of Findings

We trained two models (plus one control) and found:

1. **Co-training dramatically improves performance** — not just on bijective tasks (where the NB-only model was never trained), but also on non-bijective tasks themselves (+46% mean reward on in-distribution test).

2. **The improvement is concentrated on a specific function family** — prefix-sum functions with non-injective emission (`nb_20`–`nb_23`), where algebraic inversion via modular arithmetic is feasible. On suffix min/max functions (`nb_18`–`nb_19`), both models perform identically.

3. **The key emergent reasoning skill is case analysis** — the co-trained model learns to recognize when a non-injective operation creates branching (e.g., `|x - 5| = 2` → `x = 3 or x = 7`) and systematically explores both branches. The NB-only model almost never does this (24% vs 0.5% of responses). This skill is both statistically significant (Fisher's OR = 55.5, p ≈ 0) and functionally causal (5× accuracy improvement when used).

4. **The NB-only model's failure mode is not "simple mutate-and-enumerate"** as initially hypothesized — it is more nuanced. The NB-only model correctly reverses the outer shift operation but fails at the inner function inversion, falling back to forward trial-and-error that rarely succeeds.

---

## 2. Experimental Setup

### 2.1 Task Description

Each task presents the model with the **full Python source code** of a digit-string function and an output string, then asks it to find N different input strings that produce that output.

**Example prompt** (abbreviated):
```
You are given a code:

def _state_emit(s, emit_fn):
    state = 0
    out = []
    for i, ch in enumerate(s):
        digit = int(ch)
        out.append(str(emit_fn(state, digit, i) % 10))
        state = (state + digit) % 10
    return ''.join(out)

def _shift_all(s, k):
    return ''.join(str((int(c) + k) % 10) for c in s)

def nb_23(s: str) -> str:
    """Prefix sum, then emit distance to 5. y[i] = |state - 5|."""
    return _state_emit(s, lambda state, digit, i: abs(state - 5))

def main_solution(x):
    return nb_23(x)

You are told that `main_solution(x)` returns the string "21".
Find 2 DIFFERENT digit-string inputs `x` such that `main_solution(x)` returns exactly that string.
If fewer than 2 valid answers exist, output only the valid answers.

Please reason between <think> </think> tags, and put answers between <answer> </answer> tags
separated by commas.
```

The model must reason through the function logic, invert it, and produce valid preimages. The full source code is always included in the prompt — the model has complete information about the function's mechanics.

### 2.2 Function Families

We use 48 digit-string functions from a synthetic "atomic function library" (v3), split into 24 bijective and 24 non-bijective. Each function takes a digit string (e.g., `"4813"`) and returns a digit string of the same length.

**Bijective functions** (`func_00` to `func_23`) — each output has exactly one preimage:

| Tier | Functions | Description | Example |
|---|---|---|---|
| 1: Permutation + constant shift | `func_00`–`func_05` | Reverse/rotate/swap + add constant to all digits | `func_00`: reverse, then +1 per digit |
| 2: Permutation + structured shift | `func_06`–`func_11` | Reorder + alternating shift patterns | `func_10`: reverse, then +1/+2 alternating |
| 3: Position-dependent shift | `func_12`–`func_15` | Add position-dependent offset | `func_12`: add (i+1) at position i |
| 4: Permutation + position shift | `func_16`–`func_19` | Reorder + position-dependent offset | `func_16`: reverse, then add (i+1) |
| 5: Stateful running sum | `func_20`–`func_23` | Sequential state: `state = (state + digit + k) % 10` | `func_20`: running sum with +1 drift |

**Non-bijective functions** (`nb_00` to `nb_23`) — each output has multiple preimages:

| Tier | Functions | Description | Example |
|---|---|---|---|
| 1: Cyclic neighbor | `nb_00`–`nb_03` | Average/min/max of adjacent digits | `nb_00`: floor-avg of cyclic neighbors, +1 |
| 2: Cyclic triples | `nb_04`–`nb_07` | Summary of length-3 windows | `nb_04`: floor-avg of cyclic triples, +1 |
| 3: Mirror pairs | `nb_08`–`nb_11` | Average/min/max of mirror-position pairs | `nb_08`: mirror floor-avg, +1 |
| 4: Block sorting | `nb_12`–`nb_15` | Sort adjacent pairs/triples | `nb_12`: sort pairs ascending, +1 |
| 5: Cumulative extrema | `nb_16`–`nb_19` | Prefix/suffix min/max | `nb_18`: suffix min, +2 |
| 6: Lossy state emission | `nb_20`–`nb_23` | Prefix sum → non-injective emit | `nb_23`: prefix sum → emit \|state − 5\| |

**Train/test split**: Functions `*_00` to `*_17` are used for training data generation; functions `*_18` to `*_23` are reserved for test evaluation. All test-set evaluations use **held-out functions** that the model has never seen during training.

### 2.3 Why Non-Bijective Functions Are Structured This Way

The non-bijective test functions (`nb_18`–`nb_23`) fall into two structurally different sub-families that are crucial for interpreting results:

**Suffix min/max functions** (`nb_18`, `nb_19`):
- `nb_18(s)` = suffix_minimum(s) + 2 (mod 10 per digit)
- `nb_19(s)` = suffix_maximum(s) + 2 (mod 10 per digit)
- Inversion requires reasoning about **ordering constraints** — which digits must be ≤ or ≥ each other
- Non-injectivity comes from the min/max collapsing different orderings to the same result

**Prefix-sum + lossy emission functions** (`nb_20`–`nb_23`):
- All share the pattern: maintain running state via `state = (state + digit) % 10`, then emit a **non-injective function of state**
- `nb_20`: emit `floor(state/2) + 3`
- `nb_21`: emit `min(state, 9-state) + 1`
- `nb_22`: emit `(state² mod 10) + 1`
- `nb_23`: emit `|state - 5|`
- Inversion requires **modular arithmetic** to recover state from emission, plus **case analysis** because each non-injective emit has multiple pre-states (e.g., `|state - 5| = 2` → state ∈ {3, 7})

This structural distinction is critical: algebraic inversion is feasible for `nb_20`–`nb_23` but requires a fundamentally different approach for `nb_18`–`nb_19`.

### 2.4 Models Trained

All models start from **Qwen2.5-7B-Instruct** and are trained with GRPO (Group Relative Policy Optimization) for 160 steps.

| Model | Training Data | Purpose |
|---|---|---|
| **Both model** | 50k bijective + 50k non-bijective multi-response samples (shuffled together) | Test whether co-training improves reasoning |
| **NB-only model** | 50k non-bijective multi-response samples | Baseline: train only on the target task domain |
| **Bij-only model** | 50k bijective multi-response samples | Control: does bijective-only training transfer? (checkpoint at step 150 only; no rollouts generated) |

### 2.5 Training Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | GRPO (no KL penalty, `kl_coef=0`) |
| Base model | Qwen2.5-7B-Instruct |
| Training steps | 160 |
| Batch size | 16 prompts per step |
| Rollouts per prompt | 16 |
| Learning rate | 1e-6 |
| Max prompt length | 1024 tokens |
| Max response length | 8192 tokens |
| PPO clip ratio | 0.2 |
| Token-level loss | Yes |
| GPUs | 8 × (single node) |
| Checkpoint frequency | Every 50 steps |

### 2.6 Reward Function

The reward uses **optional shortfall accuracy** (pure accuracy — no format bonus). The codebase contains a legacy scoring path that adds a 0.1 format weight, but our data uses the newer `multi_response_scoring` dispatch (`"nonbijective_preimage"` or `"bijective_inverse"`), which returns `(acc, acc)` — i.e., the RL reward signal equals the accuracy directly:

```
reward = accuracy = min(correct_unique, required) / max(required, submitted)
```

Where:
- `correct_unique` = number of distinct submitted answers that, when executed through the function, produce the target output
- `required` = number of answers requested in the prompt (typically 2)
- `submitted` = number of answers the model actually outputs

**Examples**:
- 2 correct out of 2 submitted, 2 required → `min(2,2)/max(2,2)` = **1.0**
- 1 correct out of 2 submitted, 2 required → `min(1,2)/max(2,2)` = **0.5**
- 1 correct out of 1 submitted, 2 required → `min(1,2)/max(2,1)` = **0.5**
- 0 correct out of 2 submitted, 2 required → **0.0**
- 2 correct out of 3 submitted, 2 required → `min(2,2)/max(2,3)` = **0.67** (penalized for over-generation)

For bijective functions asked for 2 inputs (where only 1 exists): the model is rewarded for finding the 1 valid input, and penalized if it submits a fabricated second answer.

### 2.7 Evaluation Sets

| Eval Set | Functions | N requested | Output length | # Samples | What it tests |
|---|---|---|---|---|---|
| NB test | `nb_18`–`nb_23` | 2 | 2–5 | 1024 | In-distribution for NB-only; standard difficulty |
| Bij test | `func_18`–`func_23` | 2 | 2–5 | 2048 | OOD for NB-only (never trained on bijective) |
| NB more candidates | `nb_18`–`nb_23` | 3–6 | 2–6 | 1024 | OOD: more answers than training asked for |
| NB long outputs | `nb_18`–`nb_23` | 2 | 6–10 | 1024 | OOD: longer strings than training |
| Lenchange | `lenop_*` (novel family) | 3–5 | 6–10 | 1024 | OOD: entirely unseen function type |
| Inverse test | `func_18`–`func_23` | 1 | 2–5 | 2048 | Different prompt format (single answer) |

**Evaluation protocol**: 4 rollouts per prompt, temperature 1.0. We report two metrics:
- **Mean reward**: average optional-shortfall-accuracy score across all rollouts and samples (ranges 0–1; this is the primary metric reflecting answer quality)
- **Sample pass rate**: fraction of samples where at least 1 of 4 rollouts achieves reward > 0 (a more lenient measure of coverage)

---

## 3. Quantitative Results

### 3.1 Overall Performance

| Eval Set | Both model (mean reward) | NB-only (mean reward) | Δ | Both (pass rate) | NB-only (pass rate) |
|---|---|---|---|---|---|
| NB test (in-dist) | **0.328** | 0.226 | +45.2% | **80.6%** | 49.1% |
| Bij test | **0.243** | 0.032 | +659% | **50.3%** | 15.6% |
| NB more candidates (OOD) | **0.318** | 0.292 | +8.9% | 100%* | 100%* |
| NB long outputs (OOD) | **0.273** | 0.251 | +8.8% | 100%* | 100%* |
| Lenchange (OOD) | **0.160** | 0.100 | +60.0% | 100%* | 100%* |
| Inverse test | 0.000 | N/A | — | 0.0% | N/A |

*100% pass rate means every sample has at least 1 rollout with some partial credit (reward > 0), but the mean reward shows the both model still achieves higher-quality answers.

**Key observations**:
- The both model outperforms the NB-only model on **every evaluation**, including the NB-only model's own training domain (+45% on NB test).
- The advantage is largest on bijective tasks (+659%), where the NB-only model was never trained.
- On OOD non-bijective sets, the gap is smaller in mean reward (+9%) but the both model is consistently better.
- The lenchange set (entirely novel function family) shows +60%, suggesting the reasoning improvement partially transfers beyond the trained function types.
- Neither model can solve the inverse test (different prompt format requiring a single answer), indicating the learned skills are tied to the multi-response prompt format.

### 3.2 Function-Level Breakdown on NB Test

This is the most revealing analysis. The 6 test functions split into two groups with starkly different outcomes:

| Function | Type | Both (mean rwd) | NB-only (mean rwd) | Δ | Both (pass rate) | NB-only (pass rate) |
|---|---|---|---|---|---|---|
| `nb_18` | Suffix min + shift | 0.506 | 0.509 | −0.6% | 100% | 98.9% |
| `nb_19` | Suffix max + shift | 0.501 | 0.507 | −1.1% | 100% | 100% |
| `nb_20` | **Prefix sum** + half + shift | **0.241** | 0.033 | +630% | **77.3%** | 7.6% |
| `nb_21` | **Prefix sum** + fold + shift | **0.211** | 0.158 | +34% | **67.0%** | 40.8% |
| `nb_22` | **Prefix sum** + sq mod 10 + shift | **0.153** | 0.035 | +337% | **56.3%** | 15.9% |
| `nb_23` | **Prefix sum** + dist-to-5 | **0.361** | 0.074 | +388% | **83.0%** | 24.8% |

**Critical finding**: The co-training advantage is **entirely concentrated on prefix-sum functions** (`nb_20`–`nb_23`). On suffix min/max functions (`nb_18`–`nb_19`), both models achieve virtually identical performance (~0.50 mean reward, ~100% pass rate).

This tells us:
1. Co-training does **not** provide a universal reasoning improvement — it specifically enables **algebraic inversion via modular arithmetic** for the prefix-sum function family.
2. Suffix min/max functions are equally easy (or equally hard at the inner inversion step) for both models, suggesting both have learned to reverse the outer `_shift_all` operation effectively.
3. The overall aggregate improvement (+45% on NB test) is driven entirely by `nb_20`–`nb_23`.

### 3.3 Differential Analysis

For each sample in the NB test (1024 samples), we check whether each model gets at least 1 rollout correct:

| Outcome | Count | % |
|---|---|---|
| Both models correct | 479 | 46.8% |
| **Both model only** correct | 334 | 32.6% |
| **NB-only model only** correct | 12 | 1.2% |
| Neither correct | 199 | 19.4% |

The ratio is **28:1** in favor of the both model — for every problem the NB-only model uniquely solves, the both model uniquely solves 28. This near-total dominance indicates the both model has strictly more capable reasoning on this task distribution.

### 3.4 Bijective Task Behavior: Output Format Flexibility

On bijective tasks (where only 1 preimage exists, but the prompt asks for 2), the models reveal fundamentally different understanding:

| Behavior | Both model | NB-only model |
|---|---|---|
| Outputs **single** answer (recognizing only 1 exists) | **66.8%** | 0.0% |
| Outputs **pair** (always tries to find 2) | 33.0% | **99.8%** |

The both model has learned to **adapt its output format based on the function structure** — recognizing bijective functions and correctly reporting that only 1 preimage exists. The NB-only model is rigidly locked into always producing pairs, because it has never encountered a situation where only 1 answer is correct.

Error breakdown on bijective tasks:

| Category | Both model | NB-only |
|---|---|---|
| Correct single answer | **21.1%** | 0.0% |
| Correct answer found within pair | 4.9% | 6.9% |
| Both answers wrong (pair) | 28.1% | **92.9%** |
| Wrong single answer | 45.6% | 0.0% |

---

## 4. Reasoning Strategy Analysis

### 4.1 Classification Methodology

We classified reasoning strategies by examining model responses for structural features. The categories are defined as follows:

- **Algebraic inversion**: The response contains at least 2 instances of explicit modular arithmetic expressions (e.g., `(state + d) % 10`, `\pmod{10}`, `mod 10`), indicating the model is solving for inputs using equations.
- **Case analysis**: The response explicitly considers multiple cases (e.g., "Case 1: state = 2, Case 2: state = 8") arising from the non-injectivity of the emit function. Detected via patterns like "Case 1"/"Case 2", "state = X or state = Y", "two possible", "state can be".
- **Trial and error**: The response tests 2+ specific candidate inputs by forward-computing the function on them. Detected via patterns like "Let's try", "Testing input", "Input 1:".
- **Forward verification**: The response computes the function forward on its final candidate answer to verify correctness before submitting.

These categories are **not mutually exclusive** — a single response can use multiple strategies (e.g., algebraic inversion with case analysis, followed by forward verification). Classification was performed programmatically across all 4096 responses per model per evaluation set.

### 4.2 Strategy Distribution

| Strategy | Both model (NB test) | NB-only (NB test) | Both model (Bij test) | NB-only (Bij test) |
|---|---|---|---|---|
| Algebraic inversion | 46.5% | 46.4% | 34.2% | 37.6% |
| Case analysis | **24.0%** | 0.5% | **6.9%** | 13.3% |
| Trial and error | **30.7%** | 8.1% | 12.4% | **29.8%** |
| Forward verification | 6.3% | 5.0% | 1.2% | 1.6% |

**Reading this table — what each column tells us**:
- *NB test columns*: Both models' strategies on non-bijective (multi-preimage) tasks, which is the primary evaluation domain.
- *Bij test columns*: How each model behaves when encountering bijective functions (single preimage). The NB-only model has never been trained on these.

**Key observations**:

1. **Algebraic inversion markers are surprisingly similar** (~46% for both models on NB test). This superficial similarity is misleading — close reading reveals the NB-only model often writes algebraic-looking text (e.g., "subtract the shift") but applies it incorrectly or only to the outer shift operation. The both model's algebraic reasoning extends deeper into the inner function inversion. Keyword-based detection cannot capture this quality difference.

2. **Case analysis is the starkest differentiator**: 24.0% vs 0.5% on NB tasks (48× ratio). The both model has learned to recognize when non-injective operations create branching and systematically explores both branches. The NB-only model almost never does this.

3. **Trial and error shows an interesting pattern**: The both model uses more trial-and-error on NB tasks (30.7% vs 8.1%), while the NB-only model uses more on bijective tasks (29.8% vs 12.4%). This suggests the NB-only model's default strategy fails on bijective functions, forcing it to fall back to trial-and-error; conversely, the both model uses trial-and-error on NB tasks as a secondary strategy when algebraic inversion is insufficient.

4. **Forward verification is rare** for both models (5–6%), suggesting neither has strongly learned to self-check answers.

### 4.3 Implication: Surface-Level Metrics Can Be Misleading

The near-identical algebraic inversion rates (~46%) would suggest both models reason similarly — but the accuracy gap (0.328 vs 0.226) tells a different story. This highlights a methodological caution: **keyword-based reasoning pattern detection can miss qualitative differences in reasoning depth**. The real distinction only becomes apparent through (a) the case analysis metric, which captures a genuinely different reasoning structure, and (b) close reading of individual responses (Section 6).

---

## 5. Key Discovery: Case Analysis as an Emergent Reasoning Skill

### 5.1 What Is Case Analysis?

The prefix-sum non-bijective functions (`nb_20`–`nb_23`) share a common inversion challenge: the **emission function is non-injective**, meaning multiple internal states produce the same output digit. For example:

- `nb_23` emits `|state - 5|`: the equation `|state - 5| = 2` has two solutions: **state = 3** or **state = 7**
- `nb_22` emits `(state² mod 10) + 1`: the equation `state² ≡ 4 (mod 10)` has two solutions: **state = 2** or **state = 8**
- `nb_21` emits `min(state, 9-state) + 1`: the equation `min(state, 9-state) = 3` has two solutions: **state = 3** or **state = 6**

**Case analysis** means the model explicitly recognizes this branching and reasons through both possibilities: "state could be 3 or 7, so we have two cases..." This is the algebraically correct approach to inverting a non-injective function.

**Trial and error**, by contrast, means the model picks a candidate input, runs the function forward to check if it works, and if not, tries another candidate — without algebraic reasoning about *why* a particular input should work.

### 5.2 Quantitative Results

| Strategy | Both model (NB test) | NB-only (NB test) | Ratio |
|---|---|---|---|
| **Case analysis** | **23.1%** | 0.5% | **46×** |
| Trial and error | 30.7% | 8.1% | 3.8× |
| Algebraic inversion markers | 46.5% | 46.4% | 1.0× |

**Statistical significance of case analysis difference**: Chi-squared = 996.75, Fisher's exact OR = 55.5, p ≈ 0. The difference is overwhelmingly significant.

**Algebraic markers are misleading**: Both models use `mod 10` and similar expressions at nearly identical rates (~46%). This superficial similarity masks a deep qualitative difference — the NB-only model writes algebraic-looking text but applies it incorrectly (see examples in Section 6). The case analysis rate is the only reliable quantitative differentiator.

### 5.3 Case Analysis Is Causally Linked to Accuracy

Within the both model's own responses on the NB test:

| Condition | Accuracy (mean reward) | N responses |
|---|---|---|
| Responses **with** case analysis | **30.4% correct** | 945 |
| Responses **without** case analysis | **6.3% correct** | 3151 |

This 5× accuracy gap, combined with the fact that 58.9% of all correct responses use case analysis (vs 18.2% of incorrect ones), indicates case analysis is not merely a surface marker but a functionally necessary reasoning step for these problems.

### 5.4 Which Functions Trigger Case Analysis?

| Function | Both model case analysis rate | Emission function | # of pre-states per emission |
|---|---|---|---|
| `nb_23` | **64.2%** | `\|state - 5\|` | 2 (symmetric around 5) |
| `nb_20` | 32.3% | `floor(state/2) + 3` | 2 (e.g., states 6,7 both → 6) |
| `nb_22` | 26.3% | `(state² mod 10) + 1` | 2 (e.g., states 2,8 both → 5) |
| `nb_21` | 22.8% | `min(state, 9-state) + 1` | 2 (fold around 4.5) |
| `nb_18` | 1.2% | suffix minimum + 2 | N/A (different structure) |
| `nb_19` | 0.6% | suffix maximum + 2 | N/A (different structure) |

Case analysis is triggered precisely by the functions where the emission is non-injective and creates a clean 2-way branch. The `nb_23` function (`|state - 5|`) has the most obvious branching structure and triggers case analysis most frequently.

### 5.5 OOD Generalization of Case Analysis

| Eval Set | Both model case analysis | NB-only | Helps accuracy? |
|---|---|---|---|
| NB test (in-dist) | 23.1% | 0.5% | Yes (5× improvement) |
| NB more candidates (OOD) | 20.4% | 1.1% | Yes (15.6% vs 1.3%) |
| NB long outputs (OOD) | 13.5% | 6.3% | No (2.9% vs 2.7%) |

Case analysis transfers to the OOD more-candidates setting and remains effective. On long outputs (length 6–10), the both model still uses case analysis more, but it no longer helps — the combinatorial explosion of cases across 6–10 positions exceeds what the model can manage in its reasoning chain.

---

## 6. Detailed Qualitative Analysis

### 6.1 Side-by-Side Comparison: Same Problem, Different Strategies

#### Example 1: `nb_23` (emit `|state - 5|`), target output "21"

**Ground truth preimages**: `"79"`, `"77"`

**Both model** (finds 1 correct: `"77"`):
```
The function computes y[i] = |state - 5|, where state = (state + digit) % 10.

For y[0] = 2:
  |d₁ mod 10 − 5| = 2
  → d₁ = 3 or d₁ = 7

For y[1] = 1:
  |(d₁ + d₂) mod 10 − 5| = 1

  Case 1: d₁ = 3 → |(3 + d₂) mod 10 − 5| = 1
    → (3 + d₂) mod 10 = 4 or 6
    → d₂ = 1 or d₂ = 3

  Case 2: d₁ = 7 → |(7 + d₂) mod 10 − 5| = 1
    → (7 + d₂) mod 10 = 4 or 6
    → d₂ = 7 or d₂ = 9

Possible inputs: "31", "33", "77", "79"
```
**Answer**: `31,77` (1 correct — "77" is valid; "31" is also valid but not in the known preimage list, which only stored 2 of the 4 valid preimages)

**What this shows**: The both model works **backward** from the output, setting up algebraic equations at each position, recognizing the 2-way branching from `|state - 5|`, and systematically constructing all 4 possible inputs. This is genuine mathematical inversion.

**NB-only model** (finds 0 correct):
```
Let's test some possible inputs:

Input 1: "54"
- state = abs(0 - 5) % 10 = 5    ← WRONG: confuses emit with update!
  Emit: "5"
- state = abs(5 - 5) % 10 = 0
  Emit: "0"
Output: "50", not "21".

Input 2: "46"
- state = abs(0 - 5) % 10 = 5    ← Same systematic error
  Emit: "5"  ...
Output: "50", not "21".

[continues trying 8+ inputs, all producing "50" due to the same error]
```
**Answer**: `52,47` (both wrong)

**What this shows**: The NB-only model attempts **forward trial-and-error** but has a **systematic misunderstanding** — it applies the emit function (`abs(state - 5)`) as if it were the state update rule (`state = (state + digit) % 10`). This causes every forward computation to produce the same wrong output. After exhausting its token budget, it outputs a guess.

#### Example 2: `nb_22` (emit `(state² mod 10) + 1`), target output "152"

**Ground truth preimages**: `"083"`, `"081"`

**Both model** (finds 1 correct: `"081"`):
```
For y[0] = 1: (state² mod 10) + 1 mod 10 = 1 → state² ≡ 0 → state = 0
For y[1] = 5: (state² mod 10) + 1 mod 10 = 5 → state² ≡ 4 → state = 2 or 8
For y[2] = 2: (state² mod 10) + 1 mod 10 = 2 → state² ≡ 1 → state = 1 or 9

Testing x = "021": state sequence 0→2→3, emits 1, 5, 0 ✗
Testing x = "081": state sequence 0→8→9, emits 1, 5, 2 ✓
```
**Answer**: `021,081`

**What this shows**: The both model solves the modular equation `state² ≡ 4 (mod 10)` → state ∈ {2, 8}, then traces through both cases to find valid full inputs.

**NB-only model** (finds 0 correct):
```
Step 1: Reverse the shift: subtract 1 from each digit of "152" → "041"  ✓ correct

Step 2: Find inputs whose _state_emit produces "041"
Testing "041": state=0, emit 0²%10=0 → "0" ✓
  state=(0+4)%10=4, emit 4²%10=6 → "6" (not "4") ✗

Testing "042": same first position issue...
Testing "031": state=(0+3)%10=3, emit 3²%10=9 → "9" (not "4") ✗

[tries "041" again, gets stuck in loop]
```
**Answer**: `041,032` (both wrong)

**What this shows**: The NB-only model correctly identifies and reverses the outer shift (+1), but then cannot invert the inner emission function. It falls into forward trial-and-error, testing candidates one by one without algebraic reasoning about what state values are needed.

#### Example 3: `nb_23`, target "23" — Clearest contrast

**Ground truth preimages**: `"39"`, `"75"`

**Both model** (finds 2 correct: `"71"`, `"39"`):
Brief algebraic verification: digit 7 → state 7 → |7−5|=2, digit 1 → state 8 → |8−5|=3 for "71". Digit 3 → state 3 → |3−5|=2, digit 9 → state 2 → |2−5|=3 for "39".

**NB-only model** (finds 1 correct: `"35"`):
Tests ~10 different inputs: "57", "48", "52", "53", "44", "51", "56", "45", "52", "46"... Eventually settles on `53, 35`. But "53" actually gives "03" not "23" — the model claims it works despite its own trace showing otherwise.

**Key difference**: The both model constructs answers algebraically in 1–2 steps. The NB-only model searches through 10+ candidates and frequently **hallucinates verification** — claiming an input works when its own step-by-step trace shows it doesn't.

### 6.2 Summary: Case Analysis vs Trial-and-Error

| Aspect | Case Analysis (both model) | Trial and Error (NB-only model) |
|---|---|---|
| **Direction** | Backward: output → equations → solve for inputs | Forward: guess input → run function → check |
| **Uses function structure?** | Yes — recognizes non-injective branching | No — treats function as black box to test against |
| **Typical # attempts** | 1–2 (direct algebraic construction) | 5–20 (sequential search) |
| **Failure mode** | Arithmetic errors in modular equations | Hallucinated verification ("this works" when trace shows it doesn't) |
| **Typical outcome** | 2/2 preimages correct, or 1/2 with arithmetic error | 1/2 correct (finds one by luck), 0/2 on harder problems |

---

## 7. Error Taxonomy

### 7.1 NB-only Model's Error Patterns

**Error Type 1: Correct Shift Reversal + Failed Inner Inversion** (dominant on `nb_20`–`nb_22`)

The model correctly identifies and reverses the outer `_shift_all` operation (e.g., subtract 1 from each digit). But it cannot algebraically invert the inner `_state_emit` function. Instead it falls into forward trial-and-error. Since the search space is large (10^n for n-digit strings) and the model doesn't constrain its search, most trials fail.

**Error Type 2: Function Semantics Confusion** (dominant on `nb_23`, ~30% of errors)

The model confuses the emit function with the state update function. For `nb_23`, it writes `state = abs(state - 5)` (the emit rule) instead of `state = (state + digit) % 10` (the actual update). This causes all forward computations to produce identical wrong outputs regardless of input.

**Error Type 3: Hallucinated Verification** (common across `nb_20`–`nb_23`)

The model performs a forward computation on a candidate input, the computation clearly shows the output doesn't match, but the model writes "this matches the target" and proceeds. This suggests the model has learned to *simulate* the form of verification without actually checking the result.

### 7.2 Both Model's Error Patterns

**Error Type 1: Incomplete Case Analysis** — The model identifies the correct algebraic framework but misses some branches (e.g., finds 2 of 4 possible inputs, but picks the 2 that don't include a valid preimage in the reference set).

**Error Type 2: Arithmetic Errors** — The model sets up the correct modular equation but makes a computational mistake (e.g., `(7 + d₂) mod 10 = 4` → writes `d₂ = 6` instead of `d₂ = 7`).

**Error Type 3: Suffix Min/Max Failures** — On `nb_18`/`nb_19`, the both model fails at the same rate as the NB-only model. These functions require reasoning about ordering constraints, which neither model has learned well. This is the clearest evidence that co-training improves a specific skill (modular arithmetic inversion), not general reasoning.

---

## 8. Discussion

### 8.1 What Co-Training Does

Co-training with bijective functions teaches the model a specific skill: **algebraic inversion of stateful digit-string functions via modular arithmetic and case analysis**. The mechanism is:

1. Bijective functions (especially `func_20`–`func_23`, the running-sum family) require exact algebraic inversion — there is exactly one preimage, so the model must compute it precisely.
2. This trains the model to track state through sequential operations using modular arithmetic.
3. When applied to non-bijective prefix-sum functions, this same skill enables the model to recognize **branching points** where the non-injective emission creates multiple valid states, leading to case analysis.
4. The result is that the model can systematically construct multiple preimages algebraically, rather than relying on trial-and-error.

### 8.2 What Co-Training Does Not Do

- **Does not improve suffix min/max inversion**: `nb_18` and `nb_19` require reasoning about digit-ordering constraints, not modular arithmetic. Co-training provides no benefit here (~0.50 mean reward for both models).
- **Does not transfer across prompt formats**: The inverse test (single-answer format) gets 0% for both models. The learned skills are tied to the multi-response prompt template.
- **Does not eliminate computation errors**: The both model still makes arithmetic mistakes, achieving <50% mean reward on most evaluations. The improvement is in *strategy*, not in *execution reliability*.
- **Does not generalize to all function types equally**: The lenchange evaluation (+60% improvement) suggests some transfer to novel function types, but the effect is much smaller than on the prefix-sum family where the trained skill directly applies.

### 8.3 Revising the "Reasoning Hacking" Thesis

**Original thesis**: "The NB-only model collapses into a mutate-and-enumerate strategy that is structurally incompatible with bijective functions. Co-training prevents this."

**Revised, more accurate description**: The NB-only model learns a **shallow two-step inversion procedure**: (1) reverse the outer shift, (2) attempt to find the inner preimage via forward trial-and-error. This procedure works reasonably well on suffix min/max functions (where the inner inversion is simpler) but fails on prefix-sum functions (where the inner inversion requires algebraic reasoning).

The "mutate-last-digit" pattern (explicitly incrementing the last digit of a found answer to fabricate a second one) is actually **rare** — only 0–1.3% of responses across both models. The NB-only model's second candidate typically comes from a failed inversion attempt with a different starting guess, not from explicitly mutating a correct answer.

The more precise framing is: **co-training teaches deeper algebraic inversion** (specifically modular arithmetic + case analysis) that replaces the shallow "reverse shift then guess" procedure. This is not about preventing a specific hack but about teaching a more powerful reasoning tool.

### 8.4 Why Does Bijective Training Help Non-Bijective Tasks?

We hypothesize:

1. **Bijective running-sum functions demand exact inversion**: Functions like `func_20` (`state = (state + digit + 1) % 10`) have a unique preimage, so the model must solve `digit = (next_state - current_state - 1) mod 10` exactly. This trains precise modular arithmetic.

2. **This skill transfers directly**: Non-bijective prefix-sum functions use the same state update rule (`state = (state + digit) % 10`). The inversion of the state update is identical; the only additional challenge is the non-injective emission, which creates branching.

3. **The model then discovers case analysis**: Having learned to invert the state update precisely, the model encounters the emission inversion and recognizes the branching — "state could be 3 or 7" — leading to case analysis as an emergent strategy.

This hypothesis explains why the improvement is specific to prefix-sum functions: they share the same underlying state mechanism as bijective running-sum functions, so the algebraic skills transfer directly.

### 8.5 Absolute Performance Remains Low

Even with co-training, the both model's mean reward on the NB test is only 0.328 (roughly 1 out of 2 answers correct on average). The model's arithmetic is unreliable, and many problems remain unsolved. This suggests:

- The model has learned the *right strategy* but cannot execute it reliably
- Longer training, larger models, or additional scaffolding (e.g., code execution for verification) could substantially improve accuracy
- The gap between "knows the right approach" and "gets the right answer" is significant for this task

---

## 9. Open Questions and Next Steps

### 9.1 High-Priority Questions

1. **Does the bij-only model show any transfer to NB tasks?** We trained a bij-only model but have not generated rollouts for it. If the bij-only model shows *some* improvement on prefix-sum NB functions (even without ever training on NB data), this would strongly confirm that bijective training teaches transferable algebraic inversion skills. If it shows *no* improvement, the benefit may require the model to encounter both task types during training.

2. **Is 160 steps sufficient?** Both models are at an early stage of training. The both model may be further from convergence because it has twice the data diversity. Training curves (currently unavailable — only wandb logs without exported CSVs) would reveal whether the gap is widening or narrowing over time.

3. **Does the case analysis rate increase over training?** Checkpoints at steps 50, 100, 150 are available. Comparing case analysis rates across checkpoints would show whether this reasoning skill emerges gradually or appears suddenly.

### 9.2 Directions for Future Work

4. **Suffix min/max functions**: Neither model handles `nb_18`/`nb_19` well beyond the shift-reversal step. Can we design auxiliary training tasks that teach **ordering-constraint reasoning** (similar to how bijective functions teach modular arithmetic)?

5. **Longer sequences**: Case analysis breaks down on output lengths 6–10 (no accuracy benefit despite still being used). The combinatorial explosion of cases at each position is the bottleneck. Can the model learn to prune infeasible branches earlier?

6. **Prompt format transfer**: Neither model can solve the inverse test (different prompt format). Training on diverse prompt formats (not just multi-response) might improve robustness.

7. **Scaling**: Does the co-training benefit persist (or grow) with larger models (14B, 32B)? Does it persist with longer training (500+ steps)?

8. **Other domains**: The principle — "co-train with a structurally constraining auxiliary task to prevent shortcut learning" — may apply beyond function inversion. What other RL tasks suffer from reward hacking that could be mitigated by structural diversity in the training mix?

---

## 10. Summary

| Finding | Evidence | Confidence |
|---|---|---|
| Co-training improves NB task performance | +45% mean reward on in-dist test, +9% on OOD | **High** |
| Improvement concentrated on prefix-sum functions | Function-level breakdown: Δ=0 on nb_18/19, Δ=34–630% on nb_20–23 | **High** |
| Case analysis is the key differentiating skill | 23% vs 0.5%, OR=55.5, 5× accuracy when used | **High** |
| Case analysis is causally helpful | Within-model comparison: 30.4% vs 6.3% accuracy | **High** |
| Both model learns output format flexibility | 66.8% single answers on bijective vs 0% for NB-only | **High** |
| Bijective running-sum training teaches transferable modular arithmetic | Hypothesis consistent with function-level data | **Medium** — needs bij-only model rollouts to confirm |
| NB-only model's failure is "shallow inversion" not "mutate-and-enumerate" | Mutate-last-digit is 0–1.3%; real failure is trial-and-error after correct shift reversal | **High** |
| The effect is specific to modular arithmetic, not general reasoning | No improvement on suffix min/max; moderate improvement on lenchange | **Medium-High** |
| Case analysis breaks down on long sequences | No accuracy benefit on len 6–10 despite being used | **Medium** — sample size limited |

---

## Appendix A: Reproduction Details

**Checkpoint locations**:
- Both model: `disk/checkpoints/numerical-task-v3-rl/rl-code-in-prompt-both-bijective-and-nonbijective/global_step_160/actor/huggingface`
- NB-only model: `disk/checkpoints/numerical-task-v3-rl/rl-code-in-prompt-nonbijective-only/global_step_160/actor/huggingface`
- Bij-only model: `disk/checkpoints/numerical-task-v3-rl/rl-code-in-prompt-bijective-only/global_step_150/` (step 160 not saved; FSDP format, not yet merged to HuggingFace)

**Rollout generation**: `python3 -m verl.trainer.main_generation` with `N_SAMPLES=4, TEMPERATURE=1.0, PROMPT_LENGTH=1024, RESPONSE_LENGTH=8192`

**Rollout data**: `analysis/rollouts/{both_model,nb_only_model}/{eval_set}.parquet`

**HTML reports for manual inspection**: `analysis/htmls/` (9 files)

**Analysis scripts**: `analysis/comprehensive_results.py`, `analysis/case_analysis_stats.py`

## Appendix B: Reward Function Details

The scoring system dispatches based on `gt_info.multi_response_scoring`:

- **`"nonbijective_preimage"`**: Each submitted answer is executed through the reference code. An answer is "correct" if the executed output exactly matches the reference output. Score = `min(correct_unique, required) / max(required, submitted)`.

- **`"bijective_inverse"`**: Same formula, but `required=1` (only 1 valid preimage exists). The model is penalized for submitting extra wrong answers.

Answer parsing extracts contiguous digit spans from the `<answer>...</answer>` block using regex `\d+`.

## Appendix C: Complete Function Definitions (Test Set)

### Bijective Test Functions

```python
# func_18: Move odd-indexed chars to front, then add half-based pattern
# func_19: Swap halves, then add repeating pattern (1,2,3,...)
# func_20: Running sum with +1 drift: state = (state + digit + 1) % 10, emit state
# func_21: Running sum with +2 drift: state = (state + digit + 2) % 10, emit state
# func_22: Running sum with alternating +1/+2 drift
# func_23: Right-to-left running sum with +1 drift
```

### Non-Bijective Test Functions

```python
def nb_18(s: str) -> str:
    """Suffix minimum of each digit with all digits to its right, then add 2."""
    # For position i: out[i] = min(s[i], s[i+1], ..., s[n-1]) + 2 (mod 10)

def nb_19(s: str) -> str:
    """Suffix maximum of each digit with all digits to its right, then add 2."""
    # For position i: out[i] = max(s[i], s[i+1], ..., s[n-1]) + 2 (mod 10)

def nb_20(s: str) -> str:
    """Prefix sum, then emit floor(state/2) + 3."""
    # state = (state + digit) % 10; emit (state // 2 + 3) % 10

def nb_21(s: str) -> str:
    """Prefix sum, then emit min(state, 9-state) + 1."""
    # state = (state + digit) % 10; emit (min(state, 9-state) + 1) % 10

def nb_22(s: str) -> str:
    """Prefix sum, then emit (state^2 mod 10) + 1."""
    # state = (state + digit) % 10; emit (state*state % 10 + 1) % 10

def nb_23(s: str) -> str:
    """Prefix sum, then emit |state - 5|."""
    # state = (state + digit) % 10; emit abs(state - 5)
```
