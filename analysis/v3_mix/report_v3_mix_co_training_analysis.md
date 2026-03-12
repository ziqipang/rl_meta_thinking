# Co-training on Atomic Skills Alleviates Reasoning Hacking: A v3_mix Analysis

## Executive Summary

This report analyzes rollout data from three models trained on the v3_mix pipeline:
1. **SFT Baseline** — supervised fine-tuned on both bijective (v3) and non-bijective (v2) functions
2. **NB-Only RL** — RL-trained on non-bijective inverse problems only
3. **Both RL** — RL co-trained on both bijective and non-bijective inverse problems

The central finding is that **co-training on bijective inversion fundamentally changes the model's reasoning strategy**. The NB-Only model develops an enumeration-dominant approach ("reasoning hacking") that works well on non-bijective tasks but fails on bijective ones. The Both model develops genuine inversion skills that transfer across task types, at a minimal cost to non-bijective performance.

---

## 1. Experimental Setup

### Task Structure
All models start from the same SFT checkpoint trained on forward problems with full function code (stage 1). During RL (stage 2), prompts show only `def main_solution(x): return func_XX(x)` with no implementation code. The model must recall function semantics from SFT.

### Function Families
- **Bijective (func_00–func_23)**: Each function has exactly one valid preimage per output. Includes swap-halves, shift-mod, running-sum, and scramble operations on digit strings.
- **Non-bijective (func_24–func_48)**: Multiple valid preimages per output. Includes neighbor-sum, max-min-difference, fold, and affine-mix operations.

### Evaluation Sets
| Eval Set | Type | N | Description |
|---|---|---|---|
| nonbijective_test | NB preimage (2 answers) | 1,024 | Standard NB inverse |
| bijective_test | Bij preimage (2 answers) | 2,048 | Standard bij inverse (only 1 valid) |
| bijective_inverse | Bij single-answer | 2,048 | Standard bij inverse (1 answer) |
| nb_more_candidates | NB preimage (3-6 answers) | 1,024 | Harder NB inverse |
| nb_long_outputs | NB preimage, long strings | 1,024 | Longer I/O strings |
| nb_long_more_cand | NB, long + more candidates | 1,024 | Combined difficulty |
| lenchange | Length-changing NB | 1,024 | Functions that change string length |

### Rollout Configuration
- n=4 responses per prompt, temperature=1.0
- 8 GPUs, vLLM generation with max 8192 response tokens

---

## 2. Quantitative Results

### 2.1 Overall Accuracy

| Eval Set | SFT Base P@1 | NB-Only RL P@1 | Both RL P@1 | Δ (Both − NB) |
|---|---|---|---|---|
| **nonbijective_test** | 0.413 | **0.873** | 0.863 | −0.010 |
| **bijective_test** | 0.014 | 0.147 | **0.336** | **+0.189** |
| **bijective_inverse** | 0.002 | 0.114 | **0.252** | **+0.138** |
| nb_more_candidates | 0.516 | **0.861** | 0.851 | −0.010 |
| nb_long_outputs | 0.002 | 0.008 | **0.018** | +0.010 |
| nb_long_more_cand | 0.013 | 0.024 | **0.060** | +0.036 |
| lenchange | 0.000 | 0.000 | 0.000 | 0.000 |

**Key observation**: Co-training costs ~1% on NB tasks but gains 14–19% on bijective tasks. This is a highly favorable trade-off.

### 2.2 Per-Response Strict Accuracy (score = 1.0)

| Eval Set | NB-Only RL | Both RL |
|---|---|---|
| nonbijective_test | **0.693** | 0.554 |
| bijective_inverse | 0.046 | **0.123** |
| nb_more_candidates | **0.397** | 0.252 |

The NB-Only model has higher strict accuracy on NB tasks (69.3% vs 55.4%). This suggests NB-Only is more specialized at finding multiple preimages on NB functions, while Both RL trades some NB specialization for broader generalization.

### 2.3 Pairwise Differential Analysis

| Eval Set | Both correct | Both-RL only | NB-Only only | Both wrong |
|---|---|---|---|---|
| nonbijective_test | 82.6% | 3.7% | 4.7% | 9.0% |
| **bijective_test** | 12.6% | **21.0%** | 2.1% | 64.3% |
| **bijective_inverse** | 6.5% | **18.8%** | 4.9% | 69.8% |
| nb_more_candidates | 81.5% | 3.5% | 4.6% | 10.4% |

On bijective tasks, Both-RL exclusively solves 21% of problems that NB-Only cannot, while NB-Only exclusively solves only 2.1%. The asymmetry is striking: **10:1 advantage for co-training on out-of-distribution bijective tasks**.

---

## 3. Reasoning Strategy Analysis

### 3.1 Strategy Prevalence on Bijective Test

This is the most revealing analysis. On bijective_test (where enumeration should fail):

| Strategy | NB-Only RL | Both RL |
|---|---|---|
| **Enumeration/Trial** | **92.3%** | 12.5% |
| **Systematic Inversion** | 20.1% | **74.1%** |
| Verification | 25.7% | 42.2% |

The NB-Only model overwhelmingly relies on enumeration (92.3%), while the Both model primarily uses systematic inversion (74.1%). This is the clearest signature of "reasoning hacking" — the NB-Only model has learned a strategy that happens to work on its training distribution (non-bijective functions with multiple preimages) but fails on bijective functions.

### 3.2 Strategy Prevalence on Non-bijective Test

| Strategy | NB-Only RL | Both RL |
|---|---|---|
| Enumeration/Trial | **82.0%** | 15.1% |
| Systematic Inversion | 2.4% | 17.9% |
| Verification | 15.7% | 16.6% |

Even on NB tasks, the models differ dramatically. NB-Only uses enumeration 82% of the time; Both RL uses it only 15%. Yet Both RL still achieves 86.3% pass@1 — it succeeds through a different mix of strategies including more verification and digit manipulation.

### 3.3 Strategy Co-occurrence on Bijective Test

| Category | NB-Only RL | Acc | Both RL | Acc |
|---|---|---|---|---|
| Inversion only | 1.5% | 16.7% | **66.0%** | **20.5%** |
| Enumeration only | **78.8%** | 5.7% | 2.5% | 10.0% |
| Both | 14.5% | 0.0% | 5.5% | 13.6% |
| Neither | 5.2% | 0.0% | 26.0% | 11.5% |

When the NB-Only model does attempt inversion (1.5% of the time), it achieves 16.7% accuracy — comparable to Both RL's 20.5%. The problem is not that NB-Only *can't* do inversion; it's that RL training on NB tasks alone doesn't reinforce inversion as the preferred strategy.

### 3.4 Strategy Prevalence on Bijective Inverse (Single Answer)

| Strategy | NB-Only RL | Both RL |
|---|---|---|
| Enumeration/Trial | **29.9%** | 1.3% |
| Systematic Inversion | 37.9% | **47.1%** |

On single-answer bijective inverse, the difference persists but is less extreme. NB-Only still uses enumeration 30% of the time (vs 1.3% for Both). The Both model's inversion-focused approach achieves 12.3% strict accuracy vs NB-Only's 4.6%.

---

## 4. Error Analysis

### 4.1 Error Categorization on Bijective Test

| Error Type | NB-Only RL | Both RL |
|---|---|---|
| **Enumeration failure** | **69.6%** | 3.5% |
| **Computation error** | 1.5% | **61.2%** |
| Mixed strategy error | 18.7% | 8.1% |
| Format error | 4.8% | 7.7% |
| Other | 5.5% | 19.6% |

**This is the most important finding.** When the NB-Only model fails on bijective tasks, 70% of failures are from enumeration (trying random inputs and not finding the answer). When the Both model fails, 61% are computation errors (using the correct inversion strategy but making arithmetic mistakes).

The distinction is: **NB-Only fails because of wrong strategy; Both RL fails because of execution errors.** Computation errors can be reduced with more training or better prompting; strategy errors require fundamentally different learning signals.

### 4.2 Error Categorization on Non-bijective Test

| Error Type | NB-Only RL | Both RL |
|---|---|---|
| Enumeration failure | **76.9%** | 9.4% |
| Computation error | 0.0% | 20.7% |
| Format error | 9.6% | 33.3% |
| Other | 9.2% | 31.2% |

Even on NB tasks, NB-Only's errors are dominated by enumeration failure (77%). The Both model's errors are more diverse — computation errors, format errors, and other issues. The Both model's higher format error rate (33%) suggests it sometimes fails to produce well-formed answers rather than failing at reasoning.

### 4.3 Error Categorization on Bijective Inverse

| Error Type | NB-Only RL | Both RL |
|---|---|---|
| Enumeration failure | **13.4%** | 0.5% |
| Computation error | 21.3% | **37.3%** |

Both models fail primarily due to computation errors on the inverse task, but NB-Only still has a non-trivial enumeration failure rate (13.4%).

---

## 5. Response Length and Efficiency

| Eval Set | NB-Only RL (mean chars) | Both RL (mean chars) | Ratio |
|---|---|---|---|
| nonbijective_test | 6,688 | 7,298 | 1.09x |
| **bijective_test** | **10,732** | **6,595** | **0.61x** |
| bijective_inverse | 3,817 | 2,050 | 0.54x |
| nb_more_candidates | 8,905 | 9,089 | 1.02x |

On bijective tasks, NB-Only produces responses that are **1.6x longer** than Both RL. This is because enumeration requires trying many candidates, while systematic inversion is more direct. The NB-Only model spends most of its response length computing forward functions on candidate inputs and observing whether they match the target output.

**Correctness vs. length**: For NB-Only RL on NB tasks, wrong responses are 2.1x longer than correct ones (11,318 vs 5,382 chars). This suggests the model generates increasingly long outputs when enumeration fails, exploring more candidates without converging.

---

## 6. Bijective Uniqueness Recognition

Bijective functions have exactly one valid preimage, but the prompt asks for 2 answers. Ideally, the model should recognize this and output 1 answer + "None".

| Metric | NB-Only RL | Both RL |
|---|---|---|
| Two different candidates | **95.1%** | 91.5% |
| Recognizes uniqueness | 2.9% | 4.1% |
| Says "None" | 0.5% | 0.4% |

Neither model strongly recognizes bijective uniqueness. Both predominantly give 2 different candidates. The critical difference is not in format but in accuracy: Both RL's candidates are more likely to contain the correct preimage (20.1% correct vs 8.6%).

---

## 7. Qualitative Case Studies

### Case 1: Bijective Inversion — func_19 (swap-halves + shift-mod3)

**Problem**: Find x such that func_19(x) = "980"

**Both RL** (correct):
> "To find the input, we need to reverse the operations:
> 1. Subtract the pattern 1, 2, 3 from each digit.
> 2. Swap back the halves.
> Starting with '980': Subtracting pattern → 867 → Swap halves → 768"

The Both model immediately identifies the inverse operations and applies them systematically.

**NB-Only RL** (wrong):
> "So, func_19('980') returns '121'. We need to find the input that returns '980'.
> Let's test other input strings:
> For '123': ... gives '435'. Not '980'.
> For '981': ... gives '209'. Not '980'.
> For '960': ..."

The NB-Only model computes the forward function on the output, then starts blindly testing inputs. This enumeration approach will almost never find the unique preimage of a bijective function.

### Case 2: Running-Sum Function — func_22 (alternating +1/+2 drift)

**Problem**: Find x such that func_22(x) = "245"

**Both RL** (correct):
> "We need the outputs to be '2', '4', and '5' respectively. So we can set up:
> 1. For first output '2': (0 + d(x[0]) + 1) % 10 = 2, so d(x[0]) = 1
> 2. For second output '4': (2 + d(x[1]) + 2) % 10 = 4, so d(x[1]) = 0
> 3. For third output '5': (4 + d(x[2]) + 1) % 10 = 5, so d(x[2]) = 0"

The Both model sets up algebraic equations from the function's recurrence relation and solves them step by step.

**NB-Only RL** (wrong):
> "Let's try another input, '236': ... gives '385'. Not '245'.
> Let's try '245' again but with a different processing..."

Again, the NB-Only model falls into trial-and-error enumeration.

### Case 3: Non-Bijective — func_26 (absolute differences)

On NB tasks, both models use similar approaches:

**Both RL**: "Let's consider the input '01': abs(0-1)=1, abs(1-0)=1 → '11'. Not '00'. Let's try '00': abs(0-0)=0, abs(0-0)=0 → '00'. Valid."

**NB-Only RL**: "For '10': abs(1-0)=1 → '1'. Not '00'. Let's try '01': ... Let's try '00': ..."

Both models use forward computation + search, which is the appropriate strategy for NB functions (no clean algebraic inverse exists). The critical difference is that Both RL *also* knows inversion for bijective functions, while NB-Only applies the same enumeration strategy everywhere.

---

## 8. Mechanistic Interpretation

### Why Does Co-Training Work?

The results strongly support the hypothesis that **bijective inversion is structurally incompatible with enumeration**:

1. **Non-bijective functions** have multiple valid preimages. Enumeration (trying random inputs) has a reasonable probability of finding at least one valid preimage. RL rewards this strategy.

2. **Bijective functions** have exactly one valid preimage. Enumeration has near-zero probability of finding it. The only reliable approach is to understand the function's structure and invert it step by step.

3. **NB-Only training** reinforces enumeration because it works on the training distribution. The model never encounters a signal that penalizes this strategy.

4. **Co-training** introduces bijective tasks where enumeration consistently fails. The model must develop inversion skills to earn rewards on bijective tasks. These skills then transfer beneficially to NB tasks as well (explaining why Both RL still achieves 86% on NB despite using less enumeration).

### The "Reasoning Hacking" Signature

The NB-Only model exhibits classic reasoning hacking:
- **Strategy collapse**: 92% of bijective responses use enumeration
- **Length inflation**: 1.6x longer responses on bijective tasks (more guessing iterations)
- **Wrong error mode**: 70% of failures are enumeration failures (wrong strategy) vs 62% computation errors in Both RL (right strategy, wrong execution)
- **Low efficiency**: Correct responses in NB-Only take 5.4K chars; wrong responses take 11.3K chars (2.1x ratio), suggesting the model generates increasingly desperate guesses

### Transfer of Skills

The Both model's inversion skills don't just help on bijective tasks — they also change how it approaches NB tasks:
- Both RL uses verification 16.6% of the time on NB tasks (vs 15.7% for NB-Only)
- Both RL uses pattern matching 9.8% (vs 3.0% for NB-Only)
- The Both model's "neither enumeration nor inversion" category has 80.5% accuracy on NB tasks — suggesting it has learned a more diverse and robust strategy repertoire

---

## 9. Limitations

1. **Uniqueness recognition is still poor**: Both models rarely recognize that bijective functions have unique preimages (~4%). Future work could explore explicit prompting or reward shaping.

2. **Long-output and lenchange tasks remain unsolved**: Both models score near 0% on these harder evaluation sets, indicating that the current training signal doesn't cover these regimes.

3. **Both RL has higher format error rates** (33% on NB errors vs 10% for NB-Only), suggesting that the diverse training signal may occasionally confuse the model's output formatting.

4. **Sample size**: The qualitative analysis is based on 100 sampled problems per eval set. Broader sampling could reveal additional patterns.

---

## 10. Summary of Key Findings

| Finding | Evidence |
|---|---|
| Co-training preserves NB accuracy | 86.3% vs 87.3% (−1%) |
| Co-training dramatically improves bijective accuracy | 33.6% vs 14.7% (+19%) |
| NB-Only develops enumeration dominance | 92% enumeration on bijective test |
| Both RL develops genuine inversion | 74% inversion on bijective test |
| Error modes are fundamentally different | NB-Only: 70% strategy errors; Both: 62% execution errors |
| Enumeration is structurally incompatible with bijective tasks | NB-Only tries 36+ candidates without finding the unique preimage |
| Co-training forces development of transferable inversion skills | Both RL achieves high NB accuracy with less enumeration |
| Response length reflects strategy efficiency | NB-Only 1.6x longer on bijective tasks |

**Bottom line**: Co-training on bijective inversion is a structural intervention that prevents reasoning hacking by introducing a task where the spurious strategy (enumeration) categorically fails. This forces the model to develop genuine reasoning skills that generalize across task types.
