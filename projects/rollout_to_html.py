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
        if re.fullmatch(r"func_\d{2}", name) is not None
    }
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
        if isinstance(node, ast.FunctionDef) and node.name.startswith("func_"):
            if getattr(node, "lineno", None) is None or getattr(node, "end_lineno", None) is None:
                continue
            start = node.lineno - 1
            end = node.end_lineno
            out[node.name] = "".join(lines[start:end]).rstrip()
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


def _inverse_exec_block(gt: dict[str, Any], response: str) -> str:
    pred_x = _extract_model_answer_text(response)
    if pred_x is None:
        return "pred_x: <missing <answer>...</answer>>"

    ref_output = gt.get("ref_output")
    ref_code = gt.get("ref_code")
    funcname = gt.get("funcname", "main_solution")
    if not isinstance(ref_code, str) or not ref_code.strip():
        return f"pred_x: {json.dumps(pred_x)}\nerror: missing ref_code"

    if _CODEIO_FUNC_ENV is None:
        return f"pred_x: {json.dumps(pred_x)}\nerror: could not import projects.datasets.create_string_data (func_XX env)"

    try:
        ref_fn = _build_ref_fn(ref_code, str(funcname))
        if not callable(ref_fn):
            return f"pred_x: {json.dumps(pred_x)}\nerror: {funcname!r} is not callable after exec(ref_code)"
        execed_output = ref_fn(str(pred_x))
    except Exception as e:
        return f"pred_x: {json.dumps(pred_x)}\nerror: exec/run failed: {e!r}"

    match = execed_output == ref_output
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
                inverse_exec_info = [_inverse_exec_block(gt, r) for r in responses]

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


def _render_html(groups: list[RowGroup], title: str) -> str:
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
  <div class="meta">examples: {len(groups)} · response rows: {total_response_rows}</div>
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
    title = f"Rollout viewer: {parquet_path}"
    out_path.write_text(_render_html(groups, title=title), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
