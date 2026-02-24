#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from typing import Any
from functools import lru_cache


_FUNC_RE = re.compile(r"\breturn\s+(func_\d+)\s*\(")
_FUNC_FALLBACK_RE = re.compile(r"\bfunc_\d+\b")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)

_CODEIO_FUNC_ENV: dict[str, object] | None = None
try:  # best-effort: only needed for inverse/preimage evaluation
    from projects.datasets import create_string_data as _create_string_data  # type: ignore

    _CODEIO_FUNC_ENV = {
        name: getattr(_create_string_data, name)
        for name in dir(_create_string_data)
        if re.fullmatch(r"(func_\d{2}|nb_\d{2})", name) is not None
    }

    # NB-v2 datasets render functions as `func_{xx}` in prompts/reference code. At runtime
    # we must provide those `func_XX` names pointing at the NB-v2 `nb_XX` implementations.
    _num_bijective = int(getattr(_create_string_data, "NUM_BIJECTIVE_FUNCS", 24))
    _num_nb_v2 = int(getattr(_create_string_data, "NUM_NB_V2_FUNCS", 25))
    for _i in range(_num_nb_v2):
        _nb_name = f"nb_{_i:02d}"
        _func_name = f"func_{(_num_bijective + _i):02d}"
        if hasattr(_create_string_data, _nb_name):
            _CODEIO_FUNC_ENV[_func_name] = getattr(_create_string_data, _nb_name)
    for _helper_name in [
        "_digits",
        "_undigits",
        "scr_reverse",
        "scr_rotate1",
        "scr_swappairs",
        "scr_oddeven",
        "scr_multperm3",
        "mix_A_neighbor_sum_affine",
        "mix_B_triple_weight_affine",
        "mix_C_runsum_affine",
        "mix_D_nonlinear_mod5_embed",
    ]:
        if hasattr(_create_string_data, _helper_name):
            _CODEIO_FUNC_ENV[_helper_name] = getattr(_create_string_data, _helper_name)
except Exception:
    _CODEIO_FUNC_ENV = None


@dataclass(frozen=True)
class RowGroup:
    example_index: int
    data_source: str
    prompt_content: str
    reward_info: str
    func_name: str | None
    func_code: str
    responses: list[str]
    inverse_exec_info: list[str] | None


@dataclass(frozen=True)
class RolloutSummary:
    acc_mean: float | None
    avg_score: float | None
    scored_response_rows: int
    error_count: int
    note: str | None = None


def _extract_prompt_content(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts: list[str] = []
        for msg in prompt:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg.get("content") or ""))
        return "\n\n".join(parts).strip()
    if isinstance(prompt, dict):
        return str(prompt.get("content") or "").strip()
    if prompt is None:
        return ""
    return str(prompt).strip()


def _extract_func_name(prompt_content: str) -> str | None:
    m = _FUNC_RE.search(prompt_content)
    if m:
        return m.group(1)
    m2 = _FUNC_FALLBACK_RE.search(prompt_content)
    if m2:
        return m2.group(0)
    return None


def _load_funcs_source_map(funcs_path: Path) -> dict[str, str]:
    src = funcs_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(funcs_path))
    lines = src.splitlines(keepends=True)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (node.name.startswith("func_") or node.name.startswith("nb_")):
            if getattr(node, "lineno", None) is None or getattr(node, "end_lineno", None) is None:
                continue
            start = node.lineno - 1
            end = node.end_lineno
            out[node.name] = "".join(lines[start:end]).rstrip()

    try:
        num_bijective = int(getattr(_create_string_data, "NUM_BIJECTIVE_FUNCS", 24))
        num_nb_v2 = int(getattr(_create_string_data, "NUM_NB_V2_FUNCS", 25))
        for i in range(num_nb_v2):
            nb_name = f"nb_{i:02d}"
            func_name = f"func_{(num_bijective + i):02d}"
            nb_src = out.get(nb_name)
            if nb_src:
                out[func_name] = f"# NOTE: {func_name} is an NB-v2 alias for {nb_name}\n\n{nb_src}"
    except Exception:
        pass
    return out


def _format_reward_info(reward_model: Any) -> str:
    if not isinstance(reward_model, dict):
        return ""

    ground_truth = reward_model.get("ground_truth")

    def _fmt_ref(ref_input: Any, ref_output: Any) -> str:
        ref_input_s = json.dumps(ref_input, ensure_ascii=False, sort_keys=True)
        ref_output_s = json.dumps(ref_output, ensure_ascii=False, sort_keys=True)
        return f"ref_input: {ref_input_s}\nref_output: {ref_output_s}"

    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            # Some datasets store a plain string answer.
            return f"ground_truth: {ground_truth}"

        if isinstance(gt, dict) and ("ref_input" in gt or "ref_output" in gt):
            return _fmt_ref(gt.get("ref_input"), gt.get("ref_output"))
        return f"ground_truth: {json.dumps(gt, ensure_ascii=False)}"

    if isinstance(ground_truth, dict):
        if "ref_input" in ground_truth or "ref_output" in ground_truth:
            return _fmt_ref(ground_truth.get("ref_input"), ground_truth.get("ref_output"))
        return f"ground_truth: {json.dumps(ground_truth, ensure_ascii=False)}"

    if isinstance(ground_truth, list):
        # Common RLHF format: list of acceptable targets
        gt_s = json.dumps(ground_truth, ensure_ascii=False)
        return f"ground_truth: {gt_s}"

    if ground_truth is None:
        return ""

    return f"ground_truth: {ground_truth}"


def _json_dumps_best_effort(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _coerce_score_acc(result: Any) -> tuple[float | None, float | None]:
    if result is None:
        return None, None

    if isinstance(result, (int, float, bool)):
        v = float(result)
        return v, v

    if isinstance(result, (list, tuple)):
        if not result:
            return None, None
        if len(result) == 1:
            try:
                v = float(result[0])
            except Exception:
                return None, None
            return v, v
        try:
            return float(result[0]), float(result[1])
        except Exception:
            return None, None

    if isinstance(result, dict):
        score_raw: Any | None = None
        for key in ("score", "reward", "rm_score", "sequence_score"):
            if key in result:
                score_raw = result.get(key)
                break

        acc_raw: Any | None = None
        for key in ("acc", "accuracy"):
            if key in result:
                acc_raw = result.get(key)
                break

        score = None
        acc = None
        if score_raw is not None:
            try:
                score = float(score_raw)
            except Exception:
                score = None
        if acc_raw is not None:
            try:
                acc = float(acc_raw)
            except Exception:
                acc = None

        if score is None and acc is not None:
            score = acc
        if acc is None and score is not None:
            acc = score
        return score, acc

    return None, None


def _parse_ground_truth(reward_model: Any) -> dict[str, Any] | None:
    if not isinstance(reward_model, dict):
        return None
    ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            return None
        return gt if isinstance(gt, dict) else None
    if isinstance(ground_truth, dict):
        return ground_truth
    return None


def _is_inverse_task(data_source: str) -> bool:
    # Our RL datasets use:
    # - codeio-*-backward-... (unique inverse; bijective only)
    # - codeio-nb-*-preimage-... (non-bijective backward; any valid preimage)
    return "-backward-" in data_source or "-preimage-" in data_source


def _is_codeio_multi(data_source: str) -> bool:
    return "codeio-nb-multi-" in data_source or "codeio-multi-" in data_source


def _extract_model_answer_text(response: str) -> str | None:
    m = _ANSWER_RE.search(response)
    if not m:
        return None
    inner = m.group(1)
    inner = inner.replace('"', "")
    inner = re.sub(r"\s+", "", inner)
    return inner


@lru_cache(maxsize=256)
def _build_ref_fn(ref_code: str, funcname: str) -> object:
    env: dict[str, object] = {"__builtins__": __builtins__}
    if _CODEIO_FUNC_ENV is not None:
        env.update(_CODEIO_FUNC_ENV)
    exec(ref_code, env, env)
    return env[funcname]


def _is_close(pred: Any, target: Any, tol: float = 0.001) -> bool:
    if isinstance(pred, dict) and isinstance(target, dict):
        if pred.keys() != target.keys():
            return False
        return all(_is_close(pred[k], target[k], tol) for k in pred)

    if isinstance(pred, list) and isinstance(target, list):
        if len(pred) != len(target):
            return False
        return all(_is_close(p, t, tol) for p, t in zip(pred, target))

    if isinstance(pred, (int, float)) and isinstance(target, (int, float)):
        try:
            if isinstance(pred, float) or isinstance(target, float):
                if pred != pred or target != target:  # NaN
                    return False
                if pred in (float("inf"), float("-inf")) or target in (float("inf"), float("-inf")):
                    return False
                return abs(pred - target) <= tol * abs(target) and int(pred) == int(target)
            return pred == target
        except Exception:
            return False

    return pred == target


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _inverse_exec_block(gt: dict[str, Any], response: str, data_source: str) -> str:
    answer_text = _extract_model_answer_text(response)
    if answer_text is None:
        return "answer: <missing <answer>...</answer>>"

    ref_output = gt.get("ref_output")
    ref_code = gt.get("ref_code")
    funcname = gt.get("funcname", "main_solution")
    if not isinstance(ref_code, str) or not ref_code.strip():
        return f"answer: {json.dumps(answer_text)}\nerror: missing ref_code"

    is_close_fn = _is_close
    build_ref_fn = _build_ref_fn
    try:
        from verl.utils.reward_score import codeio as _reward_codeio  # type: ignore

        is_close_fn = getattr(_reward_codeio, "is_close", _is_close)
        build_ref_fn = getattr(_reward_codeio, "_build_codeio_ref_fn", _build_ref_fn)
    except Exception:
        pass

    if _CODEIO_FUNC_ENV is None and build_ref_fn is _build_ref_fn:
        return (
            f"answer: {json.dumps(answer_text)}\n"
            "error: could not import projects.datasets.create_string_data (func env missing)"
        )

    try:
        ref_fn = build_ref_fn(str(ref_code), str(funcname))
        if not callable(ref_fn):
            return f"answer: {json.dumps(answer_text)}\nerror: {funcname!r} is not callable after exec(ref_code)"
    except Exception as e:
        return f"answer: {json.dumps(answer_text)}\nerror: build ref_fn failed: {e!r}"

    if _is_codeio_multi(data_source) and "preimage" in data_source:
        raw_candidates = re.findall(r"\d+", answer_text)
        if not raw_candidates:
            return f"answer: {json.dumps(answer_text)}\nerror: no digit candidates in <answer>..."

        unique_candidates = _dedupe_keep_order(raw_candidates)[:5]

        try:
            num_answers = int(gt.get("num_answers", 2))
        except Exception:
            num_answers = 2
        if num_answers <= 0:
            num_answers = 2

        correct: list[str] = []
        per_cand: list[str] = []
        for cand in unique_candidates:
            try:
                execed_output = ref_fn(str(cand))
                ok = bool(is_close_fn(execed_output, ref_output))
                if ok:
                    correct.append(cand)
                per_cand.append(
                    f"cand: {json.dumps(cand)}\n"
                    f"exec_output: {json.dumps(execed_output, ensure_ascii=False)}\n"
                    f"match: {ok}"
                )
            except Exception as e:
                per_cand.append(f"cand: {json.dumps(cand)}\nerror: exec/run failed: {e!r}")

        score = min(len(correct), num_answers) / float(num_answers)
        match = len(correct) >= num_answers

        header = (
            f"answer: {json.dumps(answer_text)}\n"
            f"candidates: {json.dumps(unique_candidates, ensure_ascii=False)}\n"
            f"correct: {json.dumps(correct, ensure_ascii=False)}\n"
            f"target_output: {json.dumps(ref_output, ensure_ascii=False)}\n"
            f"num_answers: {num_answers}\n"
            f"score: {score:.4f}\n"
            f"match: {match}"
        )
        return header + "\n\n" + "\n\n".join(per_cand)

    pred_x = answer_text
    try:
        execed_output = ref_fn(str(pred_x))
    except Exception as e:
        return f"pred_x: {json.dumps(pred_x)}\nerror: exec/run failed: {e!r}"

    match = bool(is_close_fn(execed_output, ref_output))
    return (
        f"pred_x: {json.dumps(pred_x)}\n"
        f"exec_output: {json.dumps(execed_output, ensure_ascii=False)}\n"
        f"target_output: {json.dumps(ref_output, ensure_ascii=False)}\n"
        f"match: {match}"
    )


def _tqdm(iterable: Any, *, total: int | None = None, desc: str | None = None) -> Any:
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _load_dataset(parquet_path: Path) -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: datasets. Install dependencies (e.g. `pip install -r requirements.txt`)."
        ) from e

    return load_dataset("parquet", data_files=str(parquet_path))["train"]


def _build_groups(
    dataset: Any,
    funcs_source: dict[str, str],
    funcs_label: str,
    start: int,
    limit: int | None,
    max_responses: int | None,
) -> list[RowGroup]:
    total = len(dataset)
    end = total if limit is None else min(total, start + limit)
    groups: list[RowGroup] = []

    for i in _tqdm(range(start, end), total=max(0, end - start), desc="Rendering"):
        row = dataset[int(i)]

        data_source = str(row.get("data_source") or "")
        prompt_content = _extract_prompt_content(row.get("prompt"))
        func_name = _extract_func_name(prompt_content)
        func_code = funcs_source.get(func_name or "", "")
        if not func_code:
            func_code = f"# Could not locate {func_name or 'function'} in {funcs_label}"

        reward_model = row.get("reward_model")
        reward_info = _format_reward_info(reward_model)

        responses_raw = row.get("responses") or []
        if isinstance(responses_raw, list):
            responses = [str(r) for r in responses_raw]
        else:
            responses = [str(responses_raw)]

        if max_responses is not None:
            responses = responses[:max_responses]

        if not responses:
            responses = [""]

        inverse_exec_info: list[str] | None = None
        if _is_inverse_task(data_source):
            gt = _parse_ground_truth(reward_model)
            if gt is None:
                inverse_exec_info = ["error: could not parse reward_model.ground_truth"] * len(responses)
            else:
                inverse_exec_info = [_inverse_exec_block(gt, r, data_source) for r in responses]

        groups.append(
            RowGroup(
                example_index=i,
                data_source=data_source,
                prompt_content=prompt_content,
                reward_info=reward_info,
                func_name=func_name,
                func_code=func_code,
                responses=responses,
                inverse_exec_info=inverse_exec_info,
            )
        )

    return groups


def _cell_pre(text: str) -> str:
    return f"<pre class='cell-pre'>{html_escape(text)}</pre>"


def _compute_summary(
    dataset: Any,
    *,
    start: int,
    end: int,
    max_responses: int | None,
) -> RolloutSummary:
    try:
        from verl.utils.reward_score import _default_compute_score as compute_score  # type: ignore
    except Exception as e:
        return RolloutSummary(
            acc_mean=None,
            avg_score=None,
            scored_response_rows=0,
            error_count=0,
            note=f"metrics unavailable: {e!r}",
        )

    score_sum = 0.0
    acc_sum = 0.0
    scored = 0
    errors = 0

    for i in _tqdm(range(start, end), total=max(0, end - start), desc="Scoring"):
        row = dataset[int(i)]
        data_source = str(row.get("data_source") or "")
        reward_model = row.get("reward_model") or {}
        if not isinstance(reward_model, dict):
            reward_model = {}
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, (dict, list)):
            ground_truth = _json_dumps_best_effort(ground_truth)

        responses_raw = row.get("responses") or []
        if isinstance(responses_raw, list):
            responses = [str(r) for r in responses_raw]
        else:
            responses = [str(responses_raw)]

        if max_responses is not None:
            responses = responses[:max_responses]

        if not responses:
            continue

        for resp in responses:
            if ground_truth is None:
                errors += 1
                continue
            try:
                result = compute_score(data_source, resp, ground_truth, extra_info=row.get("extra_info"))
            except Exception:
                errors += 1
                continue

            score, acc = _coerce_score_acc(result)
            if score is None and acc is None:
                errors += 1
                continue
            if score is None:
                score = acc
            if acc is None:
                acc = score
            if score is None or acc is None:
                errors += 1
                continue

            score_sum += float(score)
            acc_sum += float(acc)
            scored += 1

    if scored == 0:
        return RolloutSummary(
            acc_mean=None,
            avg_score=None,
            scored_response_rows=0,
            error_count=errors,
            note="no scorable responses found",
        )

    return RolloutSummary(
        acc_mean=acc_sum / float(scored),
        avg_score=score_sum / float(scored),
        scored_response_rows=scored,
        error_count=errors,
    )


def _render_html(groups: list[RowGroup], title: str, summary: RolloutSummary | None) -> str:
    rows_html: list[str] = []
    total_response_rows = sum(len(g.responses) for g in groups)

    for g in groups:
        rowspan = len(g.responses)
        idx_td = f"<td rowspan='{rowspan}' class='idx'>{g.example_index}</td>"
        prompt_td = f"<td rowspan='{rowspan}' class='prompt'>{_cell_pre(g.prompt_content)}</td>"
        reward_td = f"<td rowspan='{rowspan}' class='reward'>{_cell_pre(g.reward_info)}</td>"
        func_title = g.func_name or ""
        func_cell = f"{func_title}\n\n{g.func_code}".strip()
        func_td = f"<td rowspan='{rowspan}' class='func'>{_cell_pre(func_cell)}</td>"
        ds_td = f"<td rowspan='{rowspan}' class='ds'>{_cell_pre(g.data_source)}</td>"

        for j, resp in enumerate(g.responses):
            resp_td = f"<td class='resp'>{_cell_pre(resp)}</td>"
            inv = ""
            if g.inverse_exec_info is not None:
                inv = g.inverse_exec_info[j] if j < len(g.inverse_exec_info) else ""
            inv_td = f"<td class='inv'>{_cell_pre(inv)}</td>"
            if j == 0:
                rows_html.append(f"<tr>{idx_td}{ds_td}{prompt_td}{reward_td}{func_td}{resp_td}{inv_td}</tr>")
            else:
                rows_html.append(f"<tr>{resp_td}{inv_td}</tr>")

    styles = """
    :root { color-scheme: light dark; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
    h1 { font-size: 18px; margin: 0 0 8px 0; }
    .meta { opacity: 0.8; margin: 0 0 12px 0; font-size: 12px; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    th, td { border: 1px solid rgba(127,127,127,0.35); vertical-align: top; padding: 8px; }
    th { position: sticky; top: 0; background: Canvas; z-index: 1; text-align: left; }
    td.idx { width: 70px; text-align: right; font-variant-numeric: tabular-nums; }
    td.ds { width: 14%; }
    td.prompt { width: 26%; }
    td.reward { width: 16%; }
    td.func { width: 25%; }
    td.resp { width: 19%; }
    td.inv { width: 20%; }
    pre.cell-pre { margin: 0; white-space: pre-wrap; word-break: break-word; max-height: 340px; overflow: auto; }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_escape(title)}</title>
  <style>{styles}</style>
</head>
<body>
  <h1>{html_escape(title)}</h1>
  <div class="meta">{html_escape(_render_meta(groups=groups, total_response_rows=total_response_rows, summary=summary))}</div>
  <table>
    <thead>
      <tr>
        <th>idx</th>
        <th>data_source</th>
        <th>prompt</th>
        <th>reward info</th>
        <th>func code</th>
        <th>response</th>
        <th>inverse exec</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""


def _render_meta(*, groups: list[RowGroup], total_response_rows: int, summary: RolloutSummary | None) -> str:
    parts: list[str] = []
    if summary is not None:
        if summary.acc_mean is None:
            parts.append("acc: N/A")
        else:
            parts.append(f"acc: {summary.acc_mean:.4f}")

        if summary.avg_score is None:
            parts.append("avg_score: N/A")
        else:
            parts.append(f"avg_score: {summary.avg_score:.4f}")

    parts.append(f"examples: {len(groups)}")
    parts.append(f"response rows: {total_response_rows}")

    if summary is not None:
        if summary.scored_response_rows:
            parts.append(f"scored rows: {summary.scored_response_rows}")
        if summary.error_count:
            parts.append(f"metric errors: {summary.error_count}")
        if summary.note:
            parts.append(summary.note)

    return " · ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert rollout parquet to an inspectable HTML table.")
    parser.add_argument("parquet", type=Path, help="Path to rollout parquet (e.g. data/.../rollout.parquet)")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: alongside parquet as rollout.html).",
    )
    parser.add_argument(
        "--funcs",
        type=Path,
        default=(Path(__file__).resolve().parent / "datasets" / "create_string_data.py"),
        help=(
            "Path to a python file containing func_XX definitions "
            "(default: projects/datasets/create_string_data.py)."
        ),
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0).")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to render (default: all).")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Alias for --limit (e.g. --n 100).",
    )
    parser.add_argument(
        "--max-responses",
        type=int,
        default=None,
        help="Max responses per example (default: all).",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compute acc/avg_score over rendered rows and show them in the HTML (can be slow).",
    )
    args = parser.parse_args()

    parquet_path: Path = args.parquet
    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}")

    out_path: Path = args.out or parquet_path.with_suffix(".html")
    funcs_path: Path = args.funcs
    if not funcs_path.exists():
        raise SystemExit(f"Funcs file not found: {funcs_path}")

    funcs_source = _load_funcs_source_map(funcs_path)
    dataset = _load_dataset(parquet_path)
    if args.n is not None and args.limit is not None:
        raise SystemExit("Use only one of --n or --limit.")
    limit = args.limit if args.n is None else args.n
    groups = _build_groups(
        dataset=dataset,
        funcs_source=funcs_source,
        funcs_label=str(funcs_path),
        start=max(0, args.start),
        limit=limit,
        max_responses=args.max_responses,
    )
    summary = None
    if args.metrics:
        total = len(dataset)
        start = max(0, args.start)
        end = total if limit is None else min(total, start + limit)
        summary = _compute_summary(dataset, start=start, end=end, max_responses=args.max_responses)
    title = f"Rollout viewer: {parquet_path}"
    out_path.write_text(_render_html(groups, title=title, summary=summary), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
