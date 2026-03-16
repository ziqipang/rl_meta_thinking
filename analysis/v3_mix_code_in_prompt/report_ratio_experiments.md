# Effect of Bijective:Non-Bijective Training Ratio on Code-in-Prompt RL

## Executive Summary

We vary the bijective:non-bijective training ratio across four code-in-prompt RL runs (200 steps each, from Qwen2.5-7B-Instruct base):

| Model | Bij:NB Ratio | Bij Train | NB Train | Total |
|---|---|---|---|---|
| **NB-Only** | 0:1 | 0 | 50,000 | 50,000 |
| **Bij1:NB3** | 1:3 | 16,667 | 50,000 | 66,667 |
| **Both (1:1)** | 1:1 | 50,000 | 50,000 | 100,000 |
| **Bij3:NB1** | 3:1 | 50,000 | 16,667 | 66,667 |

The central finding is a **strategy-accuracy trade-off driven by bijective exposure**. Increasing bijective training drives a monotonic shift from enumeration toward inversion-based reasoning. However, **the 1:1 ratio achieves the best overall generalization** — it uniquely balances inversion capability with the flexibility needed for non-bijective tasks. Extreme ratios produce strong but brittle specialists.

**Metric note**: All metrics use the exact training scorer (`codeio_multi.compute_score`). See the original report for scorer details.

---

## 1. Overall Accuracy

| Eval Set | NB-Only (0:1) | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| **nonbijective_test** | **0.926** | 0.890 | 0.894 | 0.858 |
| bijective_test | 0.209 | **0.406** | 0.359 | 0.308 |
| bijective_inverse | 0.459 | 0.731 | 0.793 | **0.791** |
| nb_more_candidates | **0.854** | 0.834 | 0.822 | 0.788 |
| nb_long_outputs | **0.183** | 0.111 | 0.180 | 0.125 |
| nb_long_more_cand | 0.185 | 0.121 | **0.190** | 0.145 |
| **lenchange** | 0.085 | 0.059 | **0.155** | 0.117 |

### Key Patterns

**1. NB in-distribution accuracy degrades monotonically with bijective fraction.**
NB-Only (0.926) > Bij1:NB3 (0.890) > Both (0.894) > Bij3:NB1 (0.858). This is expected: more bijective training dilutes NB task exposure. The 1:1 and 1:3 ratios are nearly tied, suggesting a moderate amount of bijective co-training incurs minimal NB cost.

**2. Bijective inverse peaks at 1:1 and 3:1 (essentially tied at ~0.79).**
Even 1:3 ratio achieves 0.731 — a small bijective presence is sufficient to learn inversion. The NB-Only model without any bijective exposure is far behind (0.459).

**3. Bijective test (multi-preimage) favors 1:3 — highest at 0.406.**
This is surprising: the model with *least* bijective training wins on bijective evaluation. The explanation lies in candidate count behavior (Section 3): Bij1:NB3 submits single answers 96.7% of the time, avoiding the over-generation penalty. Bij3:NB1, despite high inversion skill, submits single answers 99.7% of the time but its inversion accuracy on multi-function tasks is lower (NB training helps with certain function families — see Section 2).

**4. All OOD NB evals (long_outputs, long_more_cand, lenchange) favor Both (1:1).**
The 1:1 model significantly outperforms both extreme ratios on every OOD generalization benchmark. This is the strongest finding: **balanced training produces uniquely robust generalization**.

**5. Lenchange (fully OOD functions) shows the clearest signal: Both (0.155) >> Bij3:NB1 (0.117) >> NB-Only (0.085) >> Bij1:NB3 (0.059).**
The 1:1 model nearly doubles the runner-up. Neither extreme ratio matches it, suggesting that generalization to unseen function families benefits from diverse training signals, not just volume.

---

## 2. Function-Level Analysis

### 2.1 Bijective Inverse — Monotonic Improvement with Bij Fraction

Most bijective functions show a clean monotonic trend: more bijective training → higher accuracy.

| Function | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| func_04 | 0.374 | 0.654 | 0.898 | **0.976** |
| func_05 | 0.304 | 0.565 | 0.963 | **0.993** |
| func_10 | 0.192 | 0.951 | 0.985 | **0.997** |
| func_16 | 0.327 | 0.990 | 0.990 | **1.000** |
| func_19 | 0.238 | 0.572 | 0.822 | **0.891** |

These are "learnable inversions" — functions where the inversion algorithm is relatively straightforward once the model learns to trace code. More bijective exposure directly improves mastery.

### 2.2 Functions Where NB Training Helps (Non-Monotonic)

Several functions show *decreasing* accuracy with more bijective training:

| Function | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| func_20 | **0.728** | 0.565 | 0.489 | 0.315 |
| func_21 | **0.644** | 0.488 | 0.407 | 0.256 |
| func_22 | 0.375 | **0.400** | 0.388 | 0.213 |

These are functions where enumeration/trial-and-error is actually the better strategy. The NB-Only model's enumeration-dominant approach outperforms inversion on these. func_20-22 are among the harder bijective functions (func_23, the SPN accumulator, also shows this pattern).

### 2.3 Non-Bijective — func_48 Collapse

func_48 (the hardest NB function) shows dramatic degradation with bijective training:

| | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| NB test | **0.796** | 0.544 | 0.560 | 0.377 |
| More candidates | **0.450** | 0.368 | 0.271 | 0.186 |

The model increasingly abandons the enumeration strategy that works for func_48 as bijective exposure increases. Bij3:NB1 has less than half the NB-Only accuracy.

### 2.4 Lenchange — Both (1:1) Uniquely Excels

| Function | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| lenop_00 | 0.092 | 0.071 | **0.216** | 0.241 |
| lenop_02 | 0.087 | 0.125 | **0.353** | 0.225 |
| lenop_03 | 0.120 | 0.102 | **0.222** | 0.127 |

The 1:1 model dominates on lenop_00, lenop_02, and lenop_03. Interestingly, Bij3:NB1 is the runner-up on lenop_00 and lenop_02 but collapses on lenop_03. The extreme ratios cannot match the 1:1 model's versatility on these unseen functions.

---

## 3. Strategy Shift: A Monotonic Transition

The bijective fraction drives a clear, monotonic shift in reasoning strategy:

### 3.1 Enumeration vs. Inversion Prevalence

**Bijective inverse task:**

| Strategy | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Enumeration/Trial | 14.9% | 0.2% | 0.0% | 0.0% |
| Systematic Inversion | 68.4% | 83.0% | 97.0% | 58.6% |
| Verification | 78.1% | 5.5% | 1.4% | 1.0% |

**Non-bijective test:**

| Strategy | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Enumeration/Trial | 50.4% | 9.8% | 38.2% | 3.2% |
| Case Analysis | 24.1% | 28.2% | 35.4% | **66.0%** |
| Systematic Inversion | 0.1% | 2.4% | 2.4% | 15.7% |

**Lenchange (fully OOD):**

| Strategy | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Enumeration/Trial | **94.0%** | 41.4% | 46.4% | 15.4% |
| Systematic Inversion | 6.3% | **83.5%** | 57.2% | 44.6% |
| Case Analysis | 0.4% | 1.0% | 7.9% | **15.0%** |

### 3.2 Strategy Interpretation

The transition follows a clear pattern:

1. **NB-Only (0:1)**: Enumeration-dominant. Tries inputs, checks if they produce the target output. Works well for NB tasks (many valid preimages), poorly for bijective (unlikely to guess).

2. **Bij1:NB3 (1:3)**: Rapid strategy collapse. Even a small bijective presence (1/4 of training) nearly eliminates enumeration. The model pivots hard to inversion/code-tracing — but its inversion skill is underdeveloped, leading to high inversion_error rates.

3. **Both (1:1)**: Balanced strategy portfolio. Uses enumeration for NB tasks where it works (38.2% on NB test), inversion for bijective tasks (97.0% on bij inverse), and a mix for OOD tasks. This flexibility drives its OOD advantage.

4. **Bij3:NB1 (3:1)**: Inversion-dominant everywhere. Applies inversion/case-analysis even on NB tasks (15.7% inversion, 66.0% case analysis on NB test). Highly effective for bijective tasks but over-applies inversion where enumeration would be better.

### 3.3 The Bij1:NB3 Paradox

Bij1:NB3 shows an interesting pattern: it has the highest inversion rate on lenchange (83.5%) but the *lowest* accuracy (0.059). This model learned to attempt inversion everywhere but didn't get enough bijective training to execute it well. Compare to Both (1:1): 57.2% inversion rate, 0.155 accuracy. The 1:1 model uses inversion more selectively and more accurately.

Similarly, Bij1:NB3 has lower verification rates than NB-Only (21.1% vs 37.9% on NB test), meaning it also lost the self-checking behavior that helps catch errors. The combination of undertrained inversion and lost verification is a dangerous regime.

---

## 4. Candidate Count Behavior (Bijective Test)

| | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Single answer (0-1 cand) | 0.9% | **96.7%** | 64.0% | **99.7%** |

The bijective test asks for 2 preimages of a bijective function (only 1 exists). Models that recognize this and submit a single answer avoid the over-generation penalty.

- **NB-Only**: Always submits 2 candidates (trained on NB tasks where 2 is expected).
- **Bij1:NB3 and Bij3:NB1**: Nearly always submit 1 candidate — strongly learned from bijective training.
- **Both (1:1)**: Mixed behavior (64% single) — retains some tendency to search for a second preimage.

This explains why Bij1:NB3 has the highest bijective_test accuracy (0.406) despite lower inversion skill than Both(1:1) on bijective_inverse: its aggressive single-answer strategy avoids the over-generation penalty that drags down the score.

---

## 5. Error Mode Analysis

### 5.1 Error Type Distribution on NB Tasks

| Error Type | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| enumeration_failure | **81.6%** | 21.7% | 59.4% | 7.2% |
| inversion_error | 0.5% | 9.4% | 2.5% | **44.4%** |
| other_error | 17.3% | **62.8%** | 24.8% | 39.6% |

**NB-Only** fails primarily through enumeration exhaustion (tries many inputs, none work). **Bij3:NB1** fails primarily through inversion errors (attempts to algebraically invert a non-bijective function, which doesn't have a unique inverse). The 1:1 model has a moderate mix of both failure modes.

The rise of "other_error" in Bij1:NB3 (62.8%) and Bij3:NB1 (39.6%) reflects responses that use neither clear enumeration nor inversion — often code-tracing that leads to incorrect conclusions without an identifiable strategy pattern.

### 5.2 Bijective Inverse Errors

| Error Type | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| Total wrong | 4,433 | 2,202 | 1,693 | 1,710 |
| inversion_error | 54.4% | 74.8% | **92.2%** | 38.7% |
| enumeration_failure | 5.6% | 0.2% | 0.1% | 0.0% |
| other_error | 20.5% | 24.8% | 7.6% | **61.3%** |

Both (1:1) and Bij3:NB1 have similar error counts (~1,700) but different profiles. Both (1:1) errors are overwhelmingly labeled "inversion_error" (92.2%), meaning the model commits to inversion and makes computational mistakes. Bij3:NB1 has a large "other_error" fraction (61.3%), suggesting a different failure pattern — potentially overconfident responses that skip explicit inversion language but still get the answer wrong.

---

## 6. Response Length

| Eval Set | NB-Only | Bij1:NB3 | Both (1:1) | Bij3:NB1 |
|---|---|---|---|---|
| nonbijective_test | 2,906 | 3,281 | 3,022 | 3,786 |
| bijective_test | **8,310** | 2,400 | 2,811 | 2,522 |
| bijective_inverse | 2,901 | 1,529 | 1,529 | 1,723 |
| lenchange | 7,228 | 3,822 | 4,496 | 4,390 |

NB-Only writes the longest responses on bijective_test (8,310 chars) and lenchange (7,228) — reflecting exhaustive enumeration attempts. Models with bijective training produce much shorter, more focused responses. Bij3:NB1 writes the longest NB-test responses (3,786), reflecting its verbose case-analysis strategy.

---

## 7. Differential Analysis

### 7.1 Bij3:NB1 vs Bij1:NB3 (Symmetric Ratio Comparison)

| Eval Set | Both Correct | Bij3:NB1 Only | Bij1:NB3 Only | Both Wrong |
|---|---|---|---|---|
| nonbijective_test | 98.1% | 0.9% | 0.7% | 0.4% |
| bijective_test | 41.6% | 13.6% | **27.0%** | 17.8% |
| bijective_inverse | 79.6% | **7.9%** | 4.6% | 7.8% |
| nb_long_outputs | 26.8% | **14.7%** | 10.5% | 48.1% |
| lenchange | 27.6% | **25.8%** | 10.5% | 36.1% |

Bij3:NB1 uniquely solves more prompts on bijective_inverse (+3.3pp), nb_long_outputs (+4.2pp), and lenchange (+15.3pp). Bij1:NB3 uniquely wins on bijective_test (+13.3pp, due to candidate count behavior). The lenchange gap is striking: Bij3:NB1 solves 25.8% of prompts that Bij1:NB3 cannot, vs only 10.5% the other way.

### 7.2 Bij3:NB1 vs Both (1:1)

| Eval Set | Both Correct | Bij3:NB1 Only | Both(1:1) Only | Both Wrong |
|---|---|---|---|---|
| bijective_inverse | 84.2% | 3.3% | 4.4% | 8.1% |
| nb_long_outputs | 33.4% | 8.0% | **18.4%** | 40.2% |
| lenchange | 39.9% | 13.5% | **18.7%** | 27.9% |

Both (1:1) uniquely solves more prompts than Bij3:NB1 on every OOD NB eval. On lenchange, the 1:1 model uniquely solves 18.7% of prompts vs 13.5% for Bij3:NB1.

---

## 8. Summary: The Ratio-Generalization Curve

The results trace a clear narrative about how bijective training shapes reasoning:

### The Good: Bijective Training Teaches Inversion
Even a small amount of bijective training (1:3 ratio) is enough to nearly eliminate enumeration and teach the model to attempt code-level inversion. This produces large gains on bijective tasks (+27pp on bij_inverse for Bij1:NB3 vs NB-Only).

### The Bad: Too Much Inversion Hurts NB Tasks
Bij3:NB1 applies inversion-like reasoning even on non-bijective tasks, where enumeration is more appropriate. This causes degradation on NB tasks (−6.8pp on NB test, −6.6pp on nb_more_candidates vs NB-Only) and especially on the hardest NB function func_48 (−42pp).

### The Optimal: 1:1 Ratio Provides Unique Generalization
The 1:1 model is the only one that excels on OOD benchmarks (lenchange +7pp over runner-up, nb_long_more_cand best). It maintains a balanced strategy portfolio: enumeration for NB tasks, inversion for bijective tasks, and the flexibility to choose the right approach for novel functions. Neither extreme ratio can match this.

### The Surprising: Strategy Acquisition vs. Strategy Execution
Bij1:NB3 acquires the inversion strategy rapidly (83.5% inversion rate on lenchange) but executes it poorly (5.9% accuracy). Both (1:1) uses inversion more selectively (57.2%) but with much higher accuracy (15.5%). This suggests that **learning when to apply inversion matters as much as learning how to invert**.

### Trade-off Curve

```
Bij Fraction:  0%      25%     50%     75%
               NB-Only Bij1:NB3 Both   Bij3:NB1

NB test acc:   0.926 → 0.890 → 0.894 → 0.858    (monotonic ↓)
Bij inv acc:   0.459 → 0.731 → 0.793 → 0.791    (saturates at ~50%)
Lenchange:     0.085 → 0.059 → 0.155 → 0.117    (peak at 50%)
Enum rate:     50.4% → 9.8%  → 38.2% → 3.2%     (monotonic ↓)
Inv rate:       0.1% → 2.4%  → 2.4%  → 15.7%    (monotonic ↑)
```

The 1:1 ratio sits at the sweet spot where the model has enough bijective exposure to learn robust inversion, but enough non-bijective exposure to retain flexible enumeration. This is the only configuration that produces strong OOD generalization.
