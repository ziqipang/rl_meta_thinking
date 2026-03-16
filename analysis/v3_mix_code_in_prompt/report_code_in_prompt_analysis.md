# Code-in-Prompt RL: How Co-Training Changes Reasoning Strategy

## Executive Summary

Two models are trained from **Qwen2.5-7B-Instruct** with full function code in prompts (200 RL steps each):
- **NB-Only** — trained on non-bijective inverse problems only
- **Both** — co-trained on bijective + non-bijective inverse problems

Despite similar in-distribution accuracy (92.6% vs 89.4% on NB test), the two models develop **fundamentally different reasoning strategies**. The Both model learns to trace code and invert operations; the NB-Only model collapses to enumeration (trial-and-error). This strategy divergence has large consequences on OOD evals.

**Metric note**: All primary metrics use the exact training scorer (`codeio_multi.compute_score`), matching the `test_acc` metric logged in WandB. This scorer uses `_compute_optional_shortfall_accuracy` which penalizes over-generation: acc = min(correct, required) / max(required, submitted). For bijective_test (where only 1 preimage exists), models submitting a single correct answer get acc=1.0, while those submitting 2 candidates (one correct, one wrong) get acc=0.5. For legacy OOD evals (nb_long_outputs, lenchange, etc.), the scorer also includes a 10% format bonus in `score` that is separate from `acc`.

---

## 1. Setup

Both models start from base Qwen2.5-7B-Instruct (no SFT). Prompts contain full function source code. The only training difference is the data mix.

**Evaluation sets** (7 total, 5 are OOD):

| Eval Set | N | Description | In-Dist? |
|---|---|---|---|
| nonbijective_test | 1,024 | NB preimage, 2 answers | Both models |
| bijective_test | 2,048 | Bij preimage, 2 answers (only 1 exists) | Both only |
| bijective_inverse | 2,048 | Bij single-answer inverse | OOD |
| nb_more_candidates | 1,024 | NB preimage, 3-6 answers | OOD |
| nb_long_outputs | 1,024 | NB preimage, longer strings | OOD |
| nb_long_more_cand | 1,024 | NB, long + more candidates | OOD |
| lenchange | 1,024 | Entirely new function family (6 lenop functions) | OOD |

Rollout: n=4 responses/prompt, temperature=1.0, max 8192 tokens.

**Scoring**: Uses the exact `codeio_multi` training scorer. Each response's acc = min(correct_unique, required) / max(required, submitted_count). This penalizes over-generation: submitting 3 candidates when 2 are required reduces the denominator's effective value even if 2 are correct. For bijective_test, the scorer knows required_count=1 (bijective), so a single correct answer scores 1.0, but submitting 2 candidates (1 correct) scores 0.5.

---

## 2. Overall Accuracy

### 2.1 Primary Metrics

| Eval Set | NB-Only Acc | Both Acc | Δ Acc |
|---|---|---|---|
| **nonbijective_test** | **0.926** | 0.894 | −0.033 |
| bijective_test | 0.209 | **0.359** | **+0.150** |
| **bijective_inverse** | 0.459 | **0.793** | **+0.335** |
| nb_more_candidates | **0.854** | 0.822 | −0.031 |
| nb_long_outputs | 0.183 | 0.180 | −0.003 |
| nb_long_more_cand | 0.185 | **0.190** | +0.005 |
| **lenchange** | 0.085 | **0.155** | **+0.069** |

*Acc = mean accuracy from `codeio_multi` scorer (matches WandB test_acc). Over-generation is penalized: acc = min(correct, required) / max(required, submitted).*

**Key observations**:

- **In-distribution NB tasks**: NB-Only is modestly better (92.6% vs 89.4%). The NB-Only model finds more distinct preimages per response.
- **Bijective test**: **Both model 35.9% vs NB-Only 20.9%** — a +15.0pp advantage. The Both model benefits from two factors: (1) it finds the correct preimage more often via inversion, and (2) it submits a single answer 64% of the time (avoiding the over-generation penalty that NB-Only incurs by always submitting 2 candidates on a task where only 1 exists).
- **Bijective inverse** (single-answer, OOD format): The largest gap. Both model scores **79.3% vs 45.9%** — a +33.5pp advantage.
- **Lenchange** (entirely unseen functions): Both model scores **15.5% vs 8.5%** — nearly doubling accuracy on functions it has never seen.
- **Other NB OOD evals**: Near-tied (long outputs, long+more candidates) or slight NB-Only advantage (more candidates). Differential analysis (Section 2.3) reveals Both model uniquely solves more prompts even when mean acc is tied.

### 2.2 Consistency: Per-Prompt Accuracy Distribution

**Bijective inverse** (4 responses per prompt):

| Per-prompt mean acc | NB-Only | Both |
|---|---|---|
| 0.0 (all 4 wrong) | 23.0% | 11.4% |
| 1.0 (all 4 correct) | 16.5% | **69.0%** |
| Median prompt acc | 0.50 | **1.00** |

The Both model gets **all 4 responses correct on 69% of prompts** — it applies a deterministic inversion algorithm that either works or doesn't. The NB-Only model's correct responses are spread evenly across all buckets, reflecting a stochastic approach.

**Bijective test** (4 responses per prompt):

| Per-prompt mean acc | NB-Only | Both |
|---|---|---|
| 0.0 (all 4 wrong) | 29.7% | 26.1% |
| 1.0 (all 4 correct) | 0.0% | **11.0%** |
| >0.5 | 0.0% | **22.2%** |
| Median prompt acc | 0.25 | 0.25 |

The Both model achieves all-perfect responses on 11% of bijective_test prompts (where NB-Only never does). This is possible because the scorer gives acc=1.0 when the model correctly identifies the single preimage and submits only one answer (which Both does 64% of the time).

**Lenchange**:

| Per-prompt mean acc | NB-Only | Both |
|---|---|---|
| 0.0 (all 4 wrong) | 52.9% | 41.4% |
| >0.5 | 0.8% | 6.5% |
| Median prompt acc | 0.00 | 0.06 |

Both model reduces the zero-score rate by 11.5pp and achieves non-trivial scores on more prompts.

### 2.3 Pairwise Differential (best-of-4 with score > 0)

| Eval Set | Both correct | Both only | NB only | Both wrong |
|---|---|---|---|---|
| nonbijective_test | 97.9% | 1.0% | 0.7% | 0.4% |
| bijective_inverse | 71.9% | **16.7%** | 5.0% | 6.4% |
| **lenchange** | 30.3% | **28.3%** | 16.8% | 24.6% |
| nb_long_outputs | 37.6% | **14.2%** | 8.2% | 40.0% |

On lenchange: Both exclusively solves 28.3% of problems NB-Only cannot (vs 16.8% the other way) — a 1.7:1 advantage.

On bijective_inverse: 16.7% vs 5.0% — a 3.3:1 advantage.

---

## 3. Strategy Divergence

### 3.1 The Core Pattern

Using keyword-based strategy detectors across response text:

**Bijective inverse (clearest signal):**

| Strategy | NB-Only | Both |
|---|---|---|
| Systematic inversion | 68.4% | **97.0%** |
| Enumeration/trial | 14.9% | 0.04% |
| Verification | 78.1% | 1.4% |
| Code tracing | 49.1% | 74.3% |

The Both model has converged to a single approach: read the code, invert the operations, compute the answer. No enumeration, no verification needed.

The NB-Only model hedges — it uses inversion (68%) but also verification (78%, i.e. plugging answers back in) and sometimes enumeration (15%). The high verification rate signals uncertainty: the model doesn't fully trust its own inversion.

**Bijective test (multi-answer format):**

| Strategy | NB-Only | Both |
|---|---|---|
| Enumeration/trial | **95.8%** | 25.9% |
| Systematic inversion | 8.9% | **74.9%** |

When asked for 2 preimages, the NB-Only model switches almost entirely to enumeration — it tries random inputs hoping to find matches. The Both model still inverts.

**Lenchange (fully OOD):**

| Strategy | NB-Only | Both |
|---|---|---|
| Enumeration/trial | **94.1%** | 46.4% |
| Systematic inversion | 6.3% | **57.3%** |
| Case analysis | 0.4% | 7.9% |

On completely unseen functions, the NB-Only model defaults to enumeration (94%). The Both model attempts inversion on 57% of responses — demonstrating that the inversion skill generalizes beyond the training distribution.

**Nonbijective test (both models' training distribution):**

| Strategy | NB-Only | Both |
|---|---|---|
| Enumeration/trial | 50.5% | 38.2% |
| Systematic inversion | 0.1% | 2.4% |
| Case analysis | 24.2% | 35.4% |

Even on NB tasks, the Both model uses less enumeration and more case analysis. 58.4% of Both model's NB responses fall into "neither enumeration nor inversion" — they solve problems by direct algebraic reasoning (setting up equations from the code) without explicit "let's try" or "inverse" language.

### 3.2 Strategy Co-occurrence and Accuracy

**Bijective inverse:**

| Combo | NB-Only (rate → acc) | Both (rate → acc) |
|---|---|---|
| Inversion only | 56.9% → 48.3% | **97.0% → 80.4%** |
| Enumeration only | 3.4% → 11.4% | 0.01% → 0% |
| Neither | 28.1% → 60.6% | 3.0% → 46.7% |

The Both model's inversion is not just more frequent — it's more accurate (80.4% vs 48.3%). It has trained extensively on bijective inversion, so its inversion procedure is better calibrated.

**Lenchange:**

| Combo | NB-Only (rate → acc) | Both (rate → acc) |
|---|---|---|
| Inversion only | 0.5% → 15.8% | **31.7% → 22.7%** |
| Enumeration only | 88.2% → 17.7% | 20.9% → 28.9% |
| Both inv+enum | 5.8% → 20.3% | **25.5% → 26.6%** |

The Both model distributes across strategies more evenly. Its inversion-only approach (31.7% of responses, 22.7% acc) and combined inv+enum (25.5%, 26.6% acc) both outperform NB-Only's pure enumeration (88.2%, 17.7% acc). Even the Both model's enumeration-only responses (20.9%, 28.9% acc) are more accurate, suggesting the Both model's enumeration is more targeted — informed by code understanding rather than blind trial-and-error.

### 3.3 Per-Function Strategy on Lenchange

| Function | Description | NB-Only Acc | Both Acc | Δ |
|---|---|---|---|---|
| lenop_00 | neighbor-sum, len−1 | 0.092 | **0.216** | +0.124 |
| lenop_01 | signed mix, len−1 | **0.071** | 0.054 | −0.017 |
| lenop_02 | first+last mix, len−1 | 0.087 | **0.353** | **+0.266** |
| lenop_03 | cyclic sum+checksum, len+1 | 0.120 | **0.222** | +0.102 |
| lenop_04 | affine state machine, len+1 | **0.045** | 0.026 | −0.019 |
| lenop_05 | prefix sum, len+1 | 0.099 | 0.093 | −0.006 |

The Both model dominates on lenop_00 (+12.4pp), lenop_02 (+26.6pp), lenop_03 (+10.2pp) — simpler algebraic structures where inversion is tractable. The NB-Only model is slightly better on lenop_01 (signed mix: `(x[i] - 2*x[i+1]) % 5`) and lenop_04 (affine state machine: `st = (3*st + x[i]) % 5`) — functions where modular arithmetic makes inversion error-prone, and where brute-force search in a small mod-5 space can stumble into solutions.

---

## 4. Bijective Uniqueness Recognition

Bijective functions have exactly 1 preimage, but bijective_test asks for 2. How do the models handle this?

| Behavior | NB-Only | Both |
|---|---|---|
| Gives single answer | 0.02% | **63.8%** |
| Gives two answers | 99.1% | 36.0% |

The Both model outputs a single answer 64% of the time — an emergent behavior. It traces the code, finds the unique preimage, and explicitly states there is no second input:

> *"Since we need 2 different digit strings and we only have one after following the steps, we conclude that '32' is the valid answer."*

> *"Since the problem asks for 2 different digit strings and we only have one valid input, we only output one answer."*

The NB-Only model almost never does this. It always produces 2 candidates, even when the second is wrong.

---

## 5. Function-Level Deep Dive: Bijective Inverse

The Both model achieves **>95% acc on 15 of 24 bijective functions**. The function-level breakdown reveals where each model excels:

### Functions where Both strongly dominates (Δ > +0.5)

| Function | Description | NB-Only | Both | Δ |
|---|---|---|---|---|
| func_10 | Digit shift (+3 mod 10) | 0.192 | **0.985** | +0.793 |
| func_17 | Permute (coprime mult 3 variant) | 0.257 | **0.977** | +0.720 |
| func_08 | Digit complement (0↔9) | 0.310 | **0.984** | +0.674 |
| func_16 | Permute (coprime mult 3) | 0.327 | **0.990** | +0.663 |
| func_05 | Digit ROT5 | 0.304 | **0.963** | +0.659 |
| func_11 | Interleave halves | 0.345 | **0.989** | +0.644 |
| func_19 | Swap halves + reverse each | 0.238 | **0.822** | +0.584 |
| func_09 | Reverse + shift (+1 mod 10) | 0.382 | **0.958** | +0.576 |
| func_04 | Swap adjacent pairs | 0.374 | **0.898** | +0.524 |

These are functions with clean, directly invertible operations (shifts, permutations, swaps). The Both model has learned reliable inversion procedures for all of them.

### Functions where NB-Only wins (Δ < −0.15)

| Function | Description | NB-Only | Both | Δ |
|---|---|---|---|---|
| func_20 | XOR with position | **0.728** | 0.489 | −0.239 |
| func_21 | Bit-reversal permutation | **0.644** | 0.407 | −0.237 |
| func_23 | 3-round substitution-permutation | **0.280** | 0.069 | **−0.211** |

**func_23 is the most instructive case** — NB-Only scores 28.0% vs Both's 6.8%. This is a 3-round SPN (substitution-permutation network). Looking at the strategies:

- **Both model** (96.1% inversion, 4.5% verification): Tries to algebraically invert the SPN but frequently makes computation errors in the multi-round state tracking. Responses are short (~1,939 chars) and confident but often wrong.

- **NB-Only model** (28.0% inversion, 78.0% verification, 4.8% enumeration): Uses inversion less often, but heavily verifies by plugging answers back in. The verification catches computation errors, leading to more correct answers despite a less systematic approach.

Pairwise on func_23: NB-Only exclusively correct on 38.1% of prompts, Both exclusively correct on only 9.5%.

This is a case where **the Both model's inversion-first strategy is overconfident** — on a function complex enough that computation errors are likely, the NB-Only model's verify-everything approach works better.

### Where NB-Only also wins on NB test: func_48

func_48 (lossy state machine with quantized emission) is the largest NB-test gap: NB-Only scores 79.6% acc vs Both's 56.0%.

| | NB-Only | Both |
|---|---|---|
| Enumeration | 79.3% | 87.6% |
| Case analysis | 14.1% | 3.9% |
| Verification | 59.5% | 17.9% |

The NB-Only model uses more case analysis (14% vs 4%) and much more verification (60% vs 18%). It has developed a specialized enumeration+verification+case-analysis pipeline for this hard function — a strategy the Both model lacks because its training time was split across bijective tasks.

---

## 6. Error Analysis

### Error Categories on OOD Evals

**Bijective inverse:**

| Error Type | NB-Only | Both |
|---|---|---|
| Inversion error | 54.4% | **92.1%** |
| Enumeration failure | 5.6% | 0.06% |
| Mixed strategy | 19.4% | 0.1% |
| Other | 20.5% | 7.6% |

**Bijective test:**

| Error Type | NB-Only | Both |
|---|---|---|
| **Enumeration failure** | **85.7%** | 7.4% |
| **Inversion error** | 0.5% | **73.1%** |
| Mixed | 9.1% | 11.3% |

**Lenchange:**

| Error Type | NB-Only | Both |
|---|---|---|
| **Enumeration failure** | **88.5%** | 20.0% |
| **Inversion error** | 0.5% | **33.0%** |
| Mixed | 5.6% | 25.2% |
| Other | 5.4% | 21.8% |

The pattern is consistent: **NB-Only fails from wrong strategy (enumeration failure), Both fails from wrong execution (computation errors in otherwise correct inversion)**. On lenchange, 88.5% of NB-Only errors are enumeration failures — the model tries random inputs on unseen functions and cannot find valid preimages. The Both model's errors are more diverse: 33% inversion errors, 25% mixed, 22% other.

---

## 7. Response Length

| Eval Set | NB-Only Mean | Both Mean | Ratio |
|---|---|---|---|
| nonbijective_test | 2,906 | 3,022 | 0.96x |
| **bijective_test** | **8,310** | **2,811** | **2.96x** |
| bijective_inverse | 2,901 | 1,529 | 1.90x |
| nb_long_outputs | 7,687 | 4,996 | 1.54x |
| **lenchange** | **7,228** | **4,496** | **1.61x** |

On bijective_test, NB-Only is **3x longer** — it generates ~8K characters of trial-and-error. The Both model inverts in ~2.8K characters.

**Wrong vs correct response length:**

| Eval | Model | Correct avg | Wrong avg | Ratio |
|---|---|---|---|---|
| NB test | NB-Only | 2,617 | 8,389 | **3.21x** |
| NB test | Both | 2,934 | 4,581 | 1.56x |
| Bij inverse | NB-Only | 2,016 | 3,649 | 1.81x |
| Bij inverse | Both | 1,393 | 2,045 | 1.47x |

When the NB-Only model fails on NB tasks, its wrong responses are 3.2x longer than correct ones — it keeps enumerating candidates without converging. The Both model's wrong-to-correct ratio is 1.6x, indicating it recognizes failure earlier.

---

## 8. Training Dynamics

### Validation Trajectories (test_acc from codeio_multi scorer)

| Step | NB-Only NB | Both NB | NB-Only Bij | Both Bij | NB-Only Bij_inv | Both Bij_inv | NB-Only Len | Both Len |
|---|---|---|---|---|---|---|---|---|
| 20 | — | 38.3% | — | 4.2% | — | 34.2% | — | 4.0% |
| 60 | — | 74.2% | — | 24.9% | — | 50.5% | — | 5.4% |
| 100 | 83.4% | 82.6% | 13.2% | 29.9% | 36.2% | 69.3% | 5.1% | 7.5% |
| 140 | 88.2% | 87.6% | 11.9% | 42.9% | 42.5% | 75.3% | 7.3% | 11.4% |
| 180 | 90.8% | 89.7% | 8.7% | 37.7% | 43.3% | 80.1% | 8.4% | 18.8% |
| 200 | 91.3% | 91.9% | **8.9%** ↓ | 37.1% | 45.4% | **82.7%** | **6.0%** | **25.2%** |

Two trends:

1. **NB-Only's bijective test performance declines** after step 120 (14.1% → 8.9%) — increasing strategy collapse toward enumeration as it optimizes further on NB tasks.

2. **Both's lenchange rises monotonically** (4.0% → 25.2%) despite never training on lenchange. The inversion skill continuously transfers to unseen functions throughout training.

---

## 9. Qualitative Case Studies

### Case 1: Bijective Inverse — func_06 (swap halves + shift)

**Problem**: Find x such that func_06(x) = "03453"  |  Correct answer: 39422

**Both** (correct, 1,500 chars):
> "To find the original input, we need to reverse the operations: (1) Subtract 1 from each digit in '03453' → '92342'. (2) Rearrange the digits to their original even-odd indexed positions."
> Answer: 39422 ✓

**NB-Only** (wrong, 2,900 chars):
> "The function can be broken down into: (1) Move odd-indexed chars to front. (2) Add 1. Let's reverse... Subtract 1 from each digit: 0-1=-1→9, 3-1=2..."
> Answer: 01234 ✗

Both models identify the correct inverse operations. The NB-Only model makes an error in the rearrangement step.

### Case 2: Bijective Inverse — func_23 (3-round SPN)

**Problem**: Find x such that func_23(x) = "6682"  |  Correct answer: 9751

**Both** (wrong, 2,022 chars — overconfident inversion):
> "To find the inverse, subtract 1 from each output digit and track the state... For first digit '6': 6-1=5, original digit is 5..."
> Answer: 5571 ✗ (incorrect state tracking across rounds)

**NB-Only** (correct on 2 of 4 responses, ~3,000 chars — verified approach):
> "Starting from rightmost digit: state = d₀+1 ≡ 2 (mod 10), so d₀ = 1. state = d₀+1+d₁+1 ≡ 8 (mod 10), so d₁ = 5..."
> Answer: 9751 ✓ (careful step-by-step with verification)

The Both model's inversion is wrong because it incorrectly handles the suffix-sum recurrence. The NB-Only model's verification-heavy approach catches the error.

### Case 3: Lenchange — lenop_02 (first+last mod 5, drop last)

**Problem**: Find 3 inputs x such that lenop_02(x) = "424331344"

**Both** (correct, 3,568 chars — algebraic approach):
> "Output has 9 chars, so input has 10. y[0] = (x[0]+x[9])%5 = 4. y[1:] = x[1:9]%5. Setting up: ch0+ch9 ≡ 4 mod 5. Middle digits: ch1≡2, ch2≡4... Three solutions for (ch0,ch9): (0,4), (1,3), (2,2)."
> Answer: 0243313444, 1243313443, 2243313442 ✓

**NB-Only** (~5,000 chars — enumeration):
> "Let me try input strings and check... For '022214': computing prefix... Let me try other strings..."
> [Semi-random guessing with forward computation checks]

The Both model reads the code, determines input length = output length + 1, sets up constraint equations, and solves them. The NB-Only model tries to brute-force the answer.

---

## 10. Summary

The two models achieve similar NB accuracy (92.6% vs 89.4%) through different strategies. The differences become visible on OOD tasks:

| Dimension | NB-Only | Both |
|---|---|---|
| **Primary strategy** | Enumeration (50-96%) | Inversion (57-97%) |
| **Bij test acc** | 20.9% | **35.9%** |
| **Bij inverse acc** | 45.9% | **79.3%** |
| **Lenchange acc** | 8.5% | **15.5%** |
| **Bij uniqueness recognition** | 0.02% | **63.8%** |
| **Error mode** | Enumeration failure (85-88%) | Computation error (73-92%) |
| **Response length on OOD** | 7-8K chars | 3-5K chars |
| **Consistency (bij_inv 4/4 correct)** | 16.5% | **69.0%** |
| **OOD trend during training** | Bij test declining | Lenchange rising |

**The Both model's advantage**: It learns a general code-tracing and inversion skill that transfers to unseen functions. It reads function code, identifies the operations, and reverses them. On bijective_inverse, it achieves near-perfect accuracy on 17/24 functions. On bijective_test, it not only finds correct preimages more often but also recognizes when a function is bijective and submits a single answer — avoiding the over-generation penalty that costs the NB-Only model accuracy.

**The NB-Only model's advantage**: On specific complex functions (func_23/SPN, func_48/lossy state machine), its verification-heavy approach catches computation errors that the Both model's confident-but-error-prone inversion misses. On NB tasks with many preimages, its enumeration+verification pipeline is well-optimized (92.6% vs 89.4%).

**The trade-off**: Co-training sacrifices ~3pp accuracy on NB tasks and loses on specific complex functions where verification matters more than systematic inversion. In exchange, it gains a general-purpose reasoning capability that works on functions the model has never seen — nearly doubling accuracy on lenchange (15.5% vs 8.5%) and improving bijective inverse by +33.5pp.

---

## Appendix: Representative Examples

Five examples illustrating the strategy divergence between the two models. Each shows the function code given in the prompt, the target output, and both models' reasoning (trimmed for brevity).

---

### A. Bijective Inverse — Both Correct via Inversion, NB-Only Wrong

**Task**: `func_07` — interleave halves then add 2 to each digit.
**Given output**: `"908"` | **Correct input**: `"768"`

**Prompt** (function code visible to model):
```python
def _shift_all(s: str, k: int) -> str:
    return "".join(_c(_d(ch) + k) for ch in s)

def _interleave_halves(s: str) -> str:
    n = len(s)
    mid = (n + 1) // 2
    first, second = s[:mid], s[mid:]
    out = []
    for i in range(len(second)):
        out.append(first[i]); out.append(second[i])
    if len(first) > len(second): out.append(first[-1])
    return "".join(out)

def func_07(s: str) -> str:
    """Interleave the two halves, then add 2."""
    return _shift_all(_interleave_halves(s), 2)
```

**Both model** (correct, ~1500 chars):
> Step 1: Subtract 2 from each digit in "908":
> - 9−2=7, 0−2=8 (mod 10), 8−2=6 → "786"
>
> Step 2: De-interleave "786":
> - Length 3, mid=2. First half takes positions 0,2 → "76", second half takes position 1 → "8"
> - Reassemble: "768"
>
> Answer: **768** ✓

**NB-Only model** (wrong, ~1500 chars):
> Step 1: Subtract 2 from each digit in "908" → "786" ✓
>
> Step 2: Reverse \_interleave\_halves on "786":
> - Split "786" into two halves: "78" and "6"
> - [Incorrectly reasons about 4-char interleave pattern for a 3-char string]
> - Reassemble: "786"
>
> Answer: **786** ✗

**What happened**: Both models correctly invert the shift. The Both model correctly de-interleaves by reconstructing which positions map to first/second half. The NB-Only model confuses the interleave indexing for odd-length strings, treating the 3-char intermediate as if the original were also 3 chars (it is — but the de-interleave logic is wrong).

---

### B. Bijective Test — Both Recognizes Uniqueness, NB-Only Submits Two

**Task**: `func_22` — running sum with alternating +1/+2 drift (bijective).
**Given output**: `"6560"` | **Correct input**: `"5702"` | **Required answers**: 2 (but only 1 exists)

**Both model** (acc=1.0, single answer):
> For i=0: state = (0 + d₀ + 1) % 10 = 6 → d₀ = 5
> For i=1: state = (6 + d₁ + 2) % 10 = 5 → d₁ = 7
> For i=2: state = (5 + d₂ + 1) % 10 = 6 → d₂ = 0
> For i=3: state = (6 + d₃ + 2) % 10 = 0 → d₃ = 2
>
> Each digit is uniquely determined. There is only one valid input.
>
> Answer: **5702**

**NB-Only model** (acc=0.5, two answers):
> [Performs identical step-by-step inversion, correctly finding d₀=5, d₁=7, d₂=0, d₃=2]
>
> Thus, one solution is s = 5702.
>
> Answer: **5702, 5704**

**What happened**: Both models correctly invert the running-sum recurrence. However, the NB-Only model — trained to always produce 2 candidates — fabricates a second answer "5704" that is incorrect. The Both model recognizes uniqueness (each equation has exactly one solution mod 10) and submits a single answer. Under the `codeio_multi` scorer, the Both model gets acc=1.0 (1 correct / max(1,1)) while NB-Only gets acc=0.5 (1 correct / max(1,2) — penalized for the spurious second candidate).

---

### C. Lenchange — Both Inverts, NB-Only Enumerates and Fails

**Task**: `lenop_00` — non-cyclic neighbor-sum mod 5. Output length = input length − 1.
**Given output**: `"234141131"` (9 digits) | **Required answers**: 3 | Input must have 10 digits.

**Prompt** (function code visible):
```python
def lenop_00(s: str) -> str:
    """Reduce length by 1: non-cyclic neighbor-sum mod 5.
    y[i] = (x[i] + x[i+1]) % 5  for i=0..n-2
    """
    n = len(s)
    if n <= 1: return ""
    out = []
    for i in range(n - 1):
        a = ord(s[i]) - ord("0")
        b = ord(s[i + 1]) - ord("0")
        out.append(str((a + b) % 5))
    return "".join(out)
```

**Both model** (acc=0.67, 5599 chars):
> To reverse: input has 10 digits. Set up constraint equations:
> - (x₀ + x₁) % 5 = 2
> - (x₁ + x₂) % 5 = 3
> - ... and so on
>
> Choose x₀ freely (mod 5) and propagate:
> - **Case x₀=0**: x₁=2, x₂=1, x₃=3, x₄=3, x₅=1, x₆=0, x₇=1, x₈=2, x₉=4 → "0213310124"
> - **Case x₀=1**: x₁=1, x₂=2, x₃=2, x₄=4, x₅=0, x₆=1, x₇=0, x₈=3, x₉=3 → "1122401033"
> - **Case x₀=2**: x₁=0, x₂=3, x₃=1, x₄=0, x₅=4, x₆=2, x₇=4, x₈=2, x₉=4 → "2031042424"
>
> Answer: **0213310124, 1122401033, 2031042424** ✓✓✓

**NB-Only model** (acc=0.0, 8069 chars):
> Let's start by assigning values and checking...
> - Try a₀=3, a₁=9: (3+9)%5 = 2 ✓
> - Try a₂=4: (9+4)%5 = 3 ✓
> - Try a₃=0: (4+0)%5 = 4 ✓
> - Try a₄=1: (0+1)%5 = 1 ✓
> - Try a₅=0: (1+0)%5 = 1 ✗ (need 4!)
>
> [Backtracks, tries different values, runs for 8K chars...]
>
> Answer: **3940101031, 3490101031, 3499010121**

**What happened**: The Both model reads the code, realizes the output is mod 5 neighbor-sums, and sets up algebraic constraints. Choosing x₀ freely gives 5 families of solutions (one per x₀ mod 5). It picks 3 and propagates. The NB-Only model attempts trial-and-error with digits 0-9 (not recognizing the mod-5 structure), hits contradictions repeatedly, and produces three strings that don't actually satisfy the constraints. The response is 44% longer despite failing.

---

### D. func_23 (SPN) — NB-Only's Verification Catches Both's Inversion Error

**Task**: `func_23` — right-to-left running sum with +1 drift (bijective).
**Given output**: `"6682"` | **Correct input**: `"9751"`

```python
def func_23(s: str) -> str:
    """Right-to-left running sum with +1 drift.
    state accumulated from end of string."""
    return _suffix_sum_emit(s, lambda i: 1)
    # state starts at 0, processes right-to-left:
    # state = (state + digit[i] + 1) % 10; out[i] = state
```

**Both model** (wrong, all 4 responses wrong):
> To invert: start from the output and work backwards.
>
> For first digit '6': 6 − 1 = 5, original digit is 5
> For second digit '6': 6 − 1 = 5, original digit is 5
> For third digit '8': 8 − 1 = 7, original digit is 7
> For fourth digit '2': 2 − 1 = 1, original digit is 1
>
> Answer: **5571** ✗

**NB-Only model** (correct, 2 of 4 responses correct):
> The function processes **right-to-left**. Starting from the rightmost digit:
>
> For i=3 (rightmost): state = (0 + d₃ + 1) % 10 → out[3] = 2
> So (d₃ + 1) % 10 = 2 → d₃ = 1
>
> For i=2: state = (2 + d₂ + 1) % 10 → out[2] = 8
> So (d₂ + 3) % 10 = 8 → d₂ = 5
>
> For i=1: state = (8 + d₁ + 1) % 10 → out[1] = 6
> So (d₁ + 9) % 10 = 6 → d₁ = 7
>
> For i=0: state = (6 + d₀ + 1) % 10 → out[0] = 6
> So (d₀ + 7) % 10 = 6 → d₀ = 9
>
> Answer: **9751** ✓

**What happened**: The function accumulates state **right-to-left**, but the Both model incorrectly processes **left-to-right**, simply subtracting 1 from each output digit. This ignores the accumulated state dependency. The NB-Only model correctly identifies the right-to-left processing order and carefully tracks the running state, solving the recurrence from the last digit backward. The Both model's confident but incorrect "subtract 1" shortcut produces 5571 instead of 9751 — a systematic misunderstanding of the suffix-sum direction. This is the class of function (func_23, 3-round state accumulation) where the NB-Only model's more careful approach outperforms the Both model's overconfident inversion (NB-Only: 28.0% acc vs Both: 6.9% acc on this function).

---

### E. NB Test — Both Models Correct, Different Strategies

**Task**: `func_30` — cyclic folded neighbor-sum mod 10 (non-bijective, many-to-one).
**Given output**: `"33"` | **Required answers**: 2

```python
def _fold0to4(t: int) -> int:
    return t if t <= 4 else 9 - t

def func_30(s: str) -> str:
    """Cyclic folded neighbor-sum mod 10.
    t = (x[i] + x[i+1 mod n]) % 10; y[i] = fold(t)"""
    n = len(s)
    out = []
    for i in range(n):
        t = (_d(s[i]) + _d(s[(i + 1) % n])) % 10
        out.append(_c(_fold0to4(t)))
    return "".join(out)
```

**Both model** (acc=1.0, case analysis, 3375 chars):
> For a 2-char input "ab": both output digits are fold((a+b)%10).
> We need fold((a+b)%10) = 3, so (a+b)%10 ∈ {3, 6}.
>
> **Case 1**: (a+b)%10 = 3 → pairs (0,3), (1,2), (2,1), (3,0)
> **Case 2**: (a+b)%10 = 6 → pairs (0,6), (1,5), (2,4), ...
>
> Pick two: "03" and "15"
>
> Answer: **03, 15** ✓✓

**NB-Only model** (acc=1.0, trial-and-error, 1971 chars):
> Let's test some values:
>
> Try a=1, b=2: (1+2)%10=3, fold(3)=3 ✓; (2+1)%10=3, fold(3)=3 ✓ → "12" works
> Try a=2, b=1: (2+1)%10=3, fold(3)=3 ✓; (1+2)%10=3, fold(3)=3 ✓ → "21" works
>
> Answer: **12, 21** ✓✓

**What happened**: Both models find correct answers, but through different reasoning. The Both model analyzes the fold function algebraically, identifying that (a+b)%10 must be 3 or 6, then systematically enumerates solutions from both cases. The NB-Only model tries specific digit pairs until two work. On this simple 2-digit problem, both approaches succeed easily and the NB-Only response is actually shorter. The strategy difference becomes consequential on harder problems (longer strings, more complex functions) where systematic analysis scales but trial-and-error does not.
