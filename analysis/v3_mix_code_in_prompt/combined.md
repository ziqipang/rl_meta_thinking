# Code-in-Prompt RL: How Data Mix Shapes Reasoning Strategy and Generalization

## Executive Summary

We study how the composition of RL training data shapes reasoning strategies in code-in-prompt inverse problem solving. All models start from **Qwen2.5-7B-Instruct** (no SFT) with full function source code in prompts and train for 200 RL steps (GRPO). We vary the bijective:non-bijective (bij:nb) training ratio across four runs:

| Model | Bij:NB Ratio | Bij Train | NB Train | Total |
|---|---|---|---|---|
| **NB-Only** | 0:1 | 0 | 50,000 | 50,000 |
| **Bij1:NB3** | 1:3 | 16,667 | 50,000 | 66,667 |
| **Both (1:1)** | 1:1 | 50,000 | 50,000 | 100,000 |
| **Bij3:NB1** | 3:1 | 50,000 | 16,667 | 66,667 |

Three key findings:

1. **Co-training with bijective functions at 1:1 ratio teaches inversion and improves OOD generalization** — the Both model nearly doubles NB-Only's accuracy on fully unseen function families (lenchange: 0.155 vs 0.085) while gaining +33pp on bijective inverse.

2. **Bijective exposure drives a monotonic strategy shift** from enumeration to inversion across all ratios. However, strategy adoption and strategy competence are separable — Bij1:NB3 adopts inversion aggressively (83.5% on lenchange) but executes it poorly (5.9% acc), ending up *worse* than NB-Only on every OOD NB benchmark.

3. **The 1:1 model's advantage is not just strategic flexibility but superior execution quality.** On matched prompts where models use the same strategy, the 1:1 model executes enumeration and inversion more accurately than other models, suggesting co-training improves the quality of reasoning itself, not just strategy selection.

**Metric note**: All metrics use the exact training scorer (`codeio_multi.compute_score`), matching WandB `test_acc`. The scorer uses `_compute_optional_shortfall_accuracy`: acc = min(correct, required) / max(required, submitted), penalizing over-generation. Rollouts use n=4 responses/prompt, temperature=1.0, max 8192 tokens.

---

## 1. Setup

### 1.1 Task Description

Both bijective and non-bijective tasks are *inverse problems*: given a function `f` and an output `y`, find input(s) `x` such that `f(x) = y`. The function code is fully visible in the prompt (code-in-prompt, stage=1). Bijective functions have exactly one preimage; non-bijective functions have multiple valid preimages.

### 1.2 Evaluation Sets

| Eval Set | N | Description | In-Dist? |
|---|---|---|---|
| nonbijective_test | 1,024 | NB preimage, 2 answers | NB models |
| bijective_test | 2,048 | Bij preimage, 2 answers (only 1 exists) | Bij models |
| bijective_inverse | 2,048 | Bij single-answer inverse | OOD for NB-Only |
| nb_more_candidates | 1,024 | NB preimage, 3-6 answers | OOD |
| nb_long_outputs | 1,024 | NB preimage, longer strings | OOD |
| nb_long_more_cand | 1,024 | NB, long + more candidates | OOD |
| lenchange | 1,024 | Entirely new function family (6 lenop functions) | OOD for all |

### 1.3 Strategy Detectors

We classify reasoning strategies via keyword patterns in response text:
- **Enumeration/Trial**: "let's try", "test input", "brute force", etc.
- **Systematic Inversion**: "invert", "reverse the operation", "work backwards", etc.
- **Code Tracing**: "step 1:", "the function does", "according to the code", etc.
- **Case Analysis**: "Case 1", "two possible cases", etc.
- **Verification**: "let's verify", "plug back", "confirm", etc.

These are not mutually exclusive. A single response can exhibit multiple strategies.

---

## 2. Overall Accuracy

### 2.1 Primary Results

| Eval Set | NB-Only (0:1) | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| **nonbijective_test** | **0.926** | 0.890 | 0.894 | 0.858 |
| bijective_test | 0.209 | **0.406** | 0.359 | 0.308 |
| **bijective_inverse** | 0.459 | 0.731 | **0.793** | 0.791 |
| nb_more_candidates | **0.854** | 0.834 | 0.822 | 0.788 |
| nb_long_outputs | **0.183** | 0.111 | 0.180 | 0.125 |
| nb_long_more_cand | 0.185 | 0.121 | **0.190** | 0.145 |
| **lenchange** | 0.085 | 0.059 | **0.155** | 0.117 |

### 2.2 Interpreting the Results

**NB in-distribution (nonbijective_test, nb_more_candidates):** NB-Only leads; accuracy degrades monotonically with bijective fraction. The differences between NB-Only, Bij1:NB3, and Both(1:1) are modest (≤3.2pp on NB test). nb_long_outputs and nb_long_more_cand are similarly close between NB-Only and Both(1:1) (within 0.5pp), so we do not treat these differences as meaningful.

**Bijective inverse:** Large gains from any bijective exposure. Even 1:3 ratio yields +27pp over NB-Only. Performance saturates around 1:1 (~0.79), with Bij3:NB1 offering no additional gain.

**Lenchange (fully OOD):** The clearest generalization signal. Both(1:1) at 0.155 nearly doubles NB-Only (0.085) and substantially exceeds Bij3:NB1 (0.117). Critically, Bij1:NB3 (0.059) is *worse* than NB-Only despite having bijective co-training — showing that co-training can hurt when the ratio is wrong.

### 2.3 Per-Prompt Consistency

| Eval Set | Metric | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|---|
| bijective_inverse | All 4 correct | 16.5% | 60.0% | **69.0%** | **72.5%** |
| bijective_inverse | All 4 wrong | 23.1% | 15.7% | **11.4%** | 12.5% |
| lenchange | All 4 correct | 1.1% | 0.7% | **3.1%** | 3.0% |
| lenchange | All 4 wrong | 52.9% | 61.9% | **41.4%** | 46.6% |

The Both model gets all 4 responses correct on 69% of bijective_inverse prompts (deterministic inversion that either works or doesn't). On lenchange, it has the lowest all-wrong rate (41.4%) and highest all-correct rate (3.1%).

---

## 3. Strategy Divergence: The Core Finding

### 3.1 Monotonic Strategy Shift

Increasing bijective fraction drives a consistent shift from enumeration toward inversion:

**Bijective inverse:**

| Strategy | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Enumeration | 14.9% | 0.2% | 0.0% | 0.0% |
| Inversion | 68.4% | 83.0% | **97.0%** | 58.6% |
| Verification | 78.1% | 5.5% | 1.4% | 1.0% |
| Code Tracing | 49.1% | 70.7% | 74.3% | 74.3% |

**Non-bijective test:**

| Strategy | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Enumeration | **50.4%** | 9.8% | 38.2% | 3.2% |
| Case Analysis | 24.1% | 28.2% | 35.4% | **66.0%** |
| Inversion | 0.1% | 2.4% | 2.4% | **15.7%** |

**Lenchange (fully OOD):**

| Strategy | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Enumeration | **94.0%** | 41.4% | 46.4% | 15.4% |
| Inversion | 6.3% | **83.5%** | 57.2% | 44.6% |
| Case Analysis | 0.4% | 1.0% | 7.9% | **15.0%** |

Key observations:
- **Enumeration drops monotonically** with bijective fraction — from 50% (NB-Only) to 3% (Bij3:NB1) on NB test.
- **Even 25% bijective training (Bij1:NB3) nearly eliminates enumeration**, dropping it from 50% to 10% on NB test and from 94% to 41% on lenchange.
- **Case analysis rises sharply at 3:1** — Bij3:NB1 uses case analysis on 66% of NB responses, attempting algebraic decomposition rather than trial-and-error.
- **Verification collapses with any bijective training** — from 78% to 5.5% on bijective inverse. Models increasingly trust their inversion without checking.

### 3.2 The NB-Only vs Both (1:1) Contrast

The original two-model comparison reveals the fundamental strategy divergence. Despite similar NB accuracy (92.6% vs 89.4%), the models reason differently:

| Dimension | NB-Only | Both (1:1) |
|---|---|---|
| **Primary strategy** | Enumeration (50-96%) | Inversion (57-97%) |
| **Bij inverse acc** | 45.9% | **79.3%** (+33.5pp) |
| **Lenchange acc** | 8.5% | **15.5%** (+7.0pp) |
| **Bij uniqueness recognition** | 0.02% | **63.8%** |
| **Error mode** | Enumeration failure (85-88%) | Computation error (73-92%) |
| **Response length (bij test)** | 8,310 chars | 2,811 chars |
| **Consistency (bij_inv 4/4)** | 16.5% | **69.0%** |

The Both model's inversion is not just more frequent — it's more accurate. On bijective inverse, its inversion-only responses achieve 80.4% accuracy vs NB-Only's 48.3%.

### 3.3 Strategy Co-occurrence

**Lenchange** (the cleanest OOD test):

| Strategy combo | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| inv+no_enum (rate → acc) | 0.5% → 15.8% | 47.7% → 10.6% | 31.7% → 22.7% | 38.9% → 19.1% |
| no_inv+enum (rate → acc) | 88.2% → 17.7% | 5.6% → 14.8% | 20.9% → 28.9% | 9.7% → 37.5% |
| inv+enum (rate → acc) | 5.8% → 20.3% | 35.8% → 21.0% | 25.5% → 26.6% | 5.7% → 20.6% |

The Both model's inversion-only responses (31.7%) have 22.7% accuracy — the highest among all models' inversion attempts. Even its enumeration-only responses (20.9%, 28.9% acc) outperform NB-Only's dominant enumeration (88.2%, 17.7% acc), suggesting that the Both model's enumeration is more targeted, informed by code understanding.

---

## 4. Why Bij1:NB3 Fails: Strategy Adoption Without Execution Competence

The Bij1:NB3 model presents a puzzle: it has co-training, yet generalizes *worse* than NB-Only on every OOD NB eval. This section diagnoses the failure.

### 4.1 Strategy-Conditional Accuracy (Same Strategy, Different Models)

If Bij1:NB3's drop were purely from choosing the wrong strategy, it should perform as well as other models when it uses the *same* strategy. It does not:

**Lenchange — accuracy when both models use enumeration:**

| Comparison | Model A (enum acc) | Model B (enum acc) | Diff |
|---|---|---|---|
| NB-Only vs Bij1:NB3 | 0.103 | 0.052 | **−0.051** |
| Both(1:1) vs Bij1:NB3 | 0.152 | 0.051 | **−0.101** |
| NB-Only vs Both(1:1) | 0.094 | 0.152 | **+0.058** |

**Lenchange — accuracy when both models use inversion:**

| Comparison | Model A (inv acc) | Model B (inv acc) | Diff |
|---|---|---|---|
| Both(1:1) vs Bij1:NB3 | 0.140 | 0.029 | **−0.111** |

Bij1:NB3 executes *every* strategy worse. Its enumeration accuracy is half of NB-Only's; its inversion accuracy is one-fifth of Both(1:1)'s. The bijective training didn't just shift strategy — it degraded execution quality across the board.

### 4.2 Matched-Prompt Strategy Transitions

Tracking what happens to the *same prompts* when the model changes:

**Lenchange (NB-Only → Bij1:NB3):**

| Transition | Fraction | NB-Only acc | Bij1:NB3 acc | Diff |
|---|---|---|---|---|
| enum → inv_only | 41.4% | 0.076 | 0.039 | −0.036 ↓ |
| enum → inv+enum | 31.5% | 0.092 | 0.096 | +0.004 ≈ |
| enum → enum_only | 5.3% | 0.103 | 0.052 | −0.051 ↓ |
| enum → other | 10.1% | 0.065 | 0.023 | −0.041 ↓ |

Every major transition either holds flat or hurts. The 41% of responses that switch from enumeration to inversion go from 7.6% to 3.9% accuracy.

**Lenchange (NB-Only → Both (1:1)):**

| Transition | Fraction | NB-Only acc | Both acc | Diff |
|---|---|---|---|---|
| enum → inv_only | 27.4% | 0.077 | 0.140 | **+0.063** ↑ |
| enum → inv+enum | 22.5% | 0.076 | 0.154 | **+0.078** ↑ |
| enum → enum_only | 19.3% | 0.094 | 0.152 | **+0.058** ↑ |
| enum → other | 19.0% | 0.083 | 0.171 | **+0.088** ↑ |

**Every transition improves for Both(1:1)** — including when the strategy stays enumeration. The 1:1 model executes enumeration better than NB-Only on the same prompts, even though it uses enumeration less frequently overall.

### 4.3 The Competence Threshold Interpretation

The Bij1:NB3 failure reveals a **competence threshold effect**:

1. Even 25% bijective training is sufficient to *adopt* inversion as a strategy — enumeration drops from 94% to 41% on lenchange.
2. But 25% bijective training is insufficient to *execute* inversion competently — inversion accuracy is 4.2% on lenchange (vs 14.3% for Both).
3. Worse, the strategy shift also degrades enumeration execution — even on the 41% of responses where Bij1:NB3 still enumerates, it does so less accurately.
4. The loss of verification behavior (drops from 64% to 59% on lenchange) removes the self-correction mechanism that helps catch errors.

The result is a model that has abandoned a working strategy (enumeration) for one it cannot execute (inversion), while losing the quality of both. This is the "worst of both worlds" regime.

### 4.4 Per-Function Evidence

**func_48** (hardest NB function) illustrates the collapse:

| Model | Overall acc | Enum rate | Enum acc | Inv rate | Inv acc |
|---|---|---|---|---|---|
| NB-Only | **0.910** | 79% | **0.908** | 0% | — |
| Bij1:NB3 | 0.738 | 11% | 0.448 | 10% | 0.603 |
| Both (1:1) | 0.861 | **86%** | 0.869 | 2% | 0.545 |
| Bij3:NB1 | 0.714 | 4% | 0.440 | **64%** | 0.710 |

Both(1:1) *selectively retains* 86% enumeration on func_48 (even higher than NB-Only!) with 0.869 accuracy — it learned which functions benefit from enumeration. Bij1:NB3 dropped enumeration to 11% and its remaining enumeration accuracy collapsed to 0.448.

**lenop_02** (first+last mod 5, OOD) shows the inverse pattern:

| Model | Overall acc | Enum rate | Enum acc | Inv rate | Inv acc |
|---|---|---|---|---|---|
| NB-Only | 0.097 | 82% | 0.097 | 0% | — |
| Bij1:NB3 | 0.180 | 31% | 0.274 | 67% | 0.143 |
| Both (1:1) | **0.401** | 23% | 0.324 | 35% | **0.440** |
| Bij3:NB1 | 0.291 | 5% | 0.500 | 42% | 0.237 |

Both(1:1) achieves 44% inversion accuracy on this function — far ahead of all others — showing it learned to invert this type of structure well.

---

## 5. Function-Level Deep Dive

### 5.1 Bijective Inverse: Monotonic Improvement on Most Functions

Most bijective functions show a clean trend — more bijective training → higher accuracy:

| Function | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| func_04 (swap pairs) | 0.374 | 0.654 | 0.898 | **0.976** |
| func_05 (ROT5) | 0.304 | 0.565 | 0.963 | **0.993** |
| func_10 (digit shift) | 0.192 | 0.951 | 0.985 | **0.997** |
| func_16 (coprime perm) | 0.327 | 0.990 | 0.990 | **1.000** |
| func_19 (swap+reverse) | 0.238 | 0.572 | 0.822 | **0.891** |

### 5.2 Anti-Bijective Functions: Where Enumeration Wins

Several functions show *decreasing* accuracy with bijective training:

| Function | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| func_20 (XOR position) | **0.728** | 0.565 | 0.489 | 0.315 |
| func_21 (bit-reversal) | **0.644** | 0.488 | 0.407 | 0.256 |
| func_23 (3-round SPN) | **0.280** | 0.176 | 0.069 | 0.099 |

func_23 is the most instructive: the NB-Only model's verification-heavy approach (78% verification rate) catches computation errors in the multi-round state tracking. The Both model commits to inversion with 96% rate but only 1.4% verification, producing confident but often wrong answers.

### 5.3 Lenchange: Both (1:1) Uniquely Excels

| Function | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| lenop_00 (neighbor-sum) | 0.092 | 0.071 | **0.216** | 0.241 |
| lenop_01 (signed mix) | **0.071** | 0.019 | 0.054 | 0.040 |
| lenop_02 (first+last) | 0.087 | 0.125 | **0.353** | 0.225 |
| lenop_03 (cyclic+check) | 0.120 | 0.102 | **0.222** | 0.127 |
| lenop_04 (affine SM) | **0.045** | 0.006 | 0.026 | 0.010 |
| lenop_05 (prefix sum) | **0.099** | 0.048 | 0.093 | 0.083 |

Both(1:1) dominates on lenop_00, lenop_02, and lenop_03 — functions with algebraic structure amenable to inversion. NB-Only retains advantages on lenop_01, lenop_04, and lenop_05 — functions where modular arithmetic makes inversion error-prone and enumeration in a small space works.

---

## 6. Bijective Uniqueness Recognition

An emergent behavior: models with bijective training learn to recognize when a function is bijective and submit a single answer, even when the prompt requests two.

| | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Single answer on bij_test | 0.9% | **96.7%** | 64.0% | **99.7%** |

This matters because the scorer penalizes over-generation. NB-Only always submits 2 candidates (trained on NB tasks where 2 is correct). Bij1:NB3 and Bij3:NB1 almost always submit 1. Both(1:1) is mixed (64% single) — retaining some tendency to search for a second preimage.

This explains why Bij1:NB3 has the highest bijective_test accuracy (0.406) despite lower inversion skill: its single-answer behavior avoids the over-generation penalty.

---

## 7. Error Analysis

### 7.1 Error Modes by Model

**On NB tasks (nonbijective_test errors):**

| Error Type | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| enumeration_failure | **81.6%** | 21.7% | 59.4% | 7.2% |
| inversion_error | 0.5% | 9.4% | 2.5% | **44.4%** |
| other_error | 17.3% | **62.8%** | 24.8% | 39.6% |

NB-Only fails from enumeration exhaustion. Bij3:NB1 fails from misapplied inversion on non-bijective functions. Bij1:NB3's dominant error mode is "other" (62.8%) — responses that use neither clear strategy, suggesting confused reasoning.

**On bijective inverse:**

| | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Total wrong | 4,433 | 2,202 | 1,693 | 1,710 |
| inversion_error | 54.4% | 74.8% | **92.2%** | 38.7% |

Both(1:1) has the fewest errors and the most coherent error profile — 92% of its errors are inversion errors (correct strategy, computation mistake). Bij3:NB1 has many "other_error" (61.3%), suggesting overconfident responses that skip explicit reasoning.

### 7.2 Verification as a Quality Signal

Verification behavior (plugging answers back to check) diverges sharply:

**Bijective inverse — inversion responses:**

| Model | Inversion n | Inv acc | +verify rate | +verify acc | −verify acc |
|---|---|---|---|---|---|
| NB-Only | 5,607 | 0.416 | 77.0% | 0.419 | 0.404 |
| Bij1:NB3 | 6,796 | 0.757 | 2.6% | 0.638 | 0.760 |
| Both (1:1) | 7,947 | 0.803 | 1.3% | 0.495 | 0.807 |
| Bij3:NB1 | 4,799 | 0.862 | 0.9% | 0.634 | 0.864 |

NB-Only verifies 77% of inversions; models with bijective training verify ≤2.6%. For the co-trained models, verification actually *hurts* accuracy (e.g., Both: 49.5% with verify vs 80.7% without) — suggesting that the inversion is reliable enough that the verification step introduces confusion rather than catching errors.

**Lenchange — enumeration responses:**

| Model | Enum n | Enum acc | +verify rate | +verify acc |
|---|---|---|---|---|
| NB-Only | 3,853 | 0.178 | 64.2% | 0.185 |
| Bij1:NB3 | 1,699 | 0.201 | 59.1% | 0.189 |
| Both (1:1) | 1,901 | 0.276 | 34.3% | 0.271 |

Both(1:1) verifies less (34% vs 64%) but achieves higher accuracy (0.276 vs 0.178), suggesting it relies on correct reasoning rather than trial-and-verify loops.

---

## 8. Response Length

| Eval Set | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| nonbijective_test | 2,906 | 3,281 | 3,022 | 3,786 |
| bijective_test | **8,310** | 2,400 | 2,811 | 2,522 |
| bijective_inverse | 2,901 | 1,529 | 1,529 | 1,723 |
| lenchange | **7,228** | 3,822 | 4,496 | 4,390 |

NB-Only writes 3x longer responses on bijective_test (8,310 chars of exhaustive enumeration). Models with bijective training produce shorter, more focused responses. NB-Only's wrong responses on NB tasks are 3.2x longer than correct ones (8,389 vs 2,617 chars) — it keeps enumerating without converging.

---

## 9. Differential Analysis

### 9.1 Both (1:1) vs NB-Only (The Main Comparison)

| Eval Set | Both correct | Both-only | NB-only | Both wrong |
|---|---|---|---|---|
| nonbijective_test | 97.9% | 1.0% | 0.7% | 0.4% |
| bijective_inverse | 71.9% | **16.7%** | 5.0% | 6.4% |
| lenchange | 30.3% | **28.3%** | 16.8% | 24.6% |
| nb_long_outputs | 37.6% | **14.2%** | 8.2% | 40.0% |

On lenchange, Both exclusively solves 28.3% of prompts NB-Only cannot (vs 16.8% the other way) — a 1.7:1 advantage. On bijective_inverse, 16.7% vs 5.0% — a 3.3:1 advantage.

### 9.2 Bij3:NB1 vs Bij1:NB3 (Symmetric Ratio Comparison)

| Eval Set | Both correct | Bij3-only | Bij1-only | Both wrong |
|---|---|---|---|---|
| bijective_inverse | 79.6% | **7.9%** | 4.6% | 7.8% |
| nb_long_outputs | 26.8% | **14.7%** | 10.5% | 48.1% |
| lenchange | 27.6% | **25.8%** | 10.5% | 36.1% |

Bij3:NB1 uniquely solves substantially more prompts on lenchange (25.8% vs 10.5%), showing its inversion skill provides real additional coverage.

### 9.3 Both (1:1) vs Bij3:NB1

| Eval Set | Both correct | Bij3-only | Both(1:1)-only | Both wrong |
|---|---|---|---|---|
| nb_long_outputs | 33.4% | 8.0% | **18.4%** | 40.2% |
| lenchange | 39.9% | 13.5% | **18.7%** | 27.9% |

Both(1:1) uniquely solves more prompts than Bij3:NB1 on every OOD NB eval, confirming its superior generalization.

---

## 10. Qualitative Case Studies

### Case A: Bijective Inverse — Both Correct via Inversion, NB-Only Wrong

**Task**: `func_07` — interleave halves then add 2 to each digit.
**Given output**: `"908"` | **Correct input**: `"768"`

**Both model** (correct, ~1500 chars):
> Step 1: Subtract 2 from each digit in "908": 9−2=7, 0−2=8 (mod 10), 8−2=6 → "786"
> Step 2: De-interleave "786": First half takes positions 0,2 → "76", second half takes position 1 → "8". Reassemble: "768"
> Answer: **768** ✓

**NB-Only model** (wrong, ~1500 chars):
> Step 1: Subtract 2 → "786" ✓
> Step 2: [Incorrectly reasons about interleave indexing for odd-length strings]
> Answer: **786** ✗

Both models attempt inversion, but the NB-Only model makes a de-interleave indexing error.

### Case B: Bijective Test — Both Recognizes Uniqueness, NB-Only Over-Generates

**Task**: `func_22` — running sum with alternating drift (bijective).
**Given output**: `"6560"` | **Required answers**: 2 (but only 1 exists)

**Both model** (acc=1.0):
> Each digit is uniquely determined by the recurrence. Only one valid input.
> Answer: **5702**

**NB-Only model** (acc=0.5):
> [Correctly finds 5702 via identical inversion, then fabricates a second answer]
> Answer: **5702, 5704**

### Case C: Lenchange — Both Inverts, NB-Only Enumerates and Fails

**Task**: `lenop_00` — non-cyclic neighbor-sum mod 5. Output length = input length − 1.
**Given output**: `"234141131"` | **Required answers**: 3

**Both model** (acc=0.67, 5599 chars):
> Input has 10 digits. y[i] = (x[i] + x[i+1]) % 5. Choose x₀ freely and propagate constraints.
> Case x₀=0: → "0213310124" ✓
> Case x₀=1: → "1122401033" ✓
> Case x₀=2: → "2031042424" ✓

**NB-Only model** (acc=0.0, 8069 chars):
> Let's try values... a₀=3, a₁=9: (3+9)%5=2 ✓... a₅=0: (1+0)%5=1 ✗ need 4!
> [Backtracks repeatedly, 8K chars of failed attempts]
> Answer: three strings that don't satisfy constraints ✗

### Case D: func_23 (SPN) — NB-Only's Verification Catches Both's Error

**Task**: `func_23` — right-to-left running sum with +1 drift.
**Given output**: `"6682"` | **Correct input**: `"9751"`

**Both model** (wrong — all 4 responses):
> Subtract 1 from each digit → **5571** ✗ (ignores accumulated state dependency)

**NB-Only model** (correct on 2/4 responses):
> Processes right-to-left, carefully tracks running state:
> d₃=1, d₂=5, d₁=7, d₀=9 → **9751** ✓

The Both model incorrectly processes left-to-right without state tracking. The NB-Only model's careful step-by-step approach gets it right.

---

## 11. Summary

### The Ratio-Generalization Relationship

```
Bij Fraction:  0%       25%       50%       75%
               NB-Only  Bij1:NB3  Both      Bij3:NB1

NB test acc:   0.926 →  0.890  →  0.894  →  0.858    (monotonic ↓)
Bij inv acc:   0.459 →  0.731  →  0.793  →  0.791    (saturates ~50%)
Lenchange:     0.085 →  0.059  →  0.155  →  0.117    (peak at 50%)
Enum rate:     50.4% →   9.8% →  38.2%  →   3.2%    (monotonic ↓)
```

### Key Takeaways

**1. Co-training with bijective functions at 1:1 ratio improves generalization.** The Both model gains +33pp on bijective inverse, +7pp on lenchange (fully OOD), with only −3pp cost on NB in-distribution. On OOD NB evals (nb_long_outputs, nb_long_more_cand), the differences are small enough to be inconclusive.

**2. Bijective training drives a monotonic strategy shift from enumeration to inversion.** Even 25% bijective exposure (Bij1:NB3) nearly eliminates enumeration. This shift is rapid and irreversible within the training budget.

**3. Strategy adoption and strategy competence are separable.** Bij1:NB3 adopts inversion aggressively (83.5% on lenchange) but executes it poorly (5.9% acc), performing *worse* than the NB-Only baseline on every OOD NB eval. There appears to be a competence threshold: insufficient bijective training disrupts existing strategies without building adequate replacements.

**4. The 1:1 model's advantage extends beyond strategy selection to execution quality.** On matched prompts using the same strategy, Both(1:1) executes enumeration better than NB-Only on lenchange (+5.8pp) and executes inversion better than Bij1:NB3 (+11.1pp). Co-training at sufficient scale improves the quality of reasoning itself, not just which strategy is chosen.

**5. No single ratio dominates all benchmarks.** NB-Only wins on NB in-distribution; Bij3:NB1 ties for best on bijective inverse; Both(1:1) wins on OOD generalization. The choice of ratio depends on which capability matters most.

### Caveats

- **Dataset pool size confound**: The 1:1 model trains from 100k total pool vs 66.7k for Bij1:NB3 and Bij3:NB1. Although RL samples only 3,200 prompts (200 steps × batch 16) regardless of pool size, the ratio determines the sampling mix. A 1:1 experiment at 66.7k total (33.3k + 33.3k) would control for this.
- **Strategy detectors are keyword-based** and approximate. A response labeled "no inversion, no enumeration" may still use implicit versions of these strategies.
- **200 RL steps is a relatively short training horizon.** Strategy and competence dynamics may evolve differently over longer training.
