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


_FUNC_RE = re.compile(r"\breturn\s+(func_\d+)\s*\(")
_FUNC_FALLBACK_RE = re.compile(r"\bfunc_\d+\b")


@dataclass(frozen=True)
class RowGroup:
    example_index: int
    prompt_content: str
    reward_info: str
    func_name: str | None
    func_code: str
    responses: list[str]


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
    start: int,
    limit: int | None,
    max_responses: int | None,
) -> list[RowGroup]:
    total = len(dataset)
    end = total if limit is None else min(total, start + limit)
    groups: list[RowGroup] = []

    for i in _tqdm(range(start, end), total=max(0, end - start), desc="Rendering"):
        row = dataset[int(i)]

        prompt_content = _extract_prompt_content(row.get("prompt"))
        func_name = _extract_func_name(prompt_content)
        func_code = funcs_source.get(func_name or "", "")
        if not func_code:
            func_code = f"# Could not locate {func_name or 'function'} in funcs.py"

        reward_info = _format_reward_info(row.get("reward_model"))

        responses_raw = row.get("responses") or []
        if isinstance(responses_raw, list):
            responses = [str(r) for r in responses_raw]
        else:
            responses = [str(responses_raw)]

        if max_responses is not None:
            responses = responses[:max_responses]

        if not responses:
            responses = [""]

        groups.append(
            RowGroup(
                example_index=i,
                prompt_content=prompt_content,
                reward_info=reward_info,
                func_name=func_name,
                func_code=func_code,
                responses=responses,
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

        for j, resp in enumerate(g.responses):
            resp_td = f"<td class='resp'>{_cell_pre(resp)}</td>"
            if j == 0:
                rows_html.append(f"<tr>{idx_td}{prompt_td}{reward_td}{func_td}{resp_td}</tr>")
            else:
                rows_html.append(f"<tr>{resp_td}</tr>")

    styles = """
    :root { color-scheme: light dark; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
    h1 { font-size: 18px; margin: 0 0 8px 0; }
    .meta { opacity: 0.8; margin: 0 0 12px 0; font-size: 12px; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    th, td { border: 1px solid rgba(127,127,127,0.35); vertical-align: top; padding: 8px; }
    th { position: sticky; top: 0; background: Canvas; z-index: 1; text-align: left; }
    td.idx { width: 70px; text-align: right; font-variant-numeric: tabular-nums; }
    td.prompt { width: 33%; }
    td.reward { width: 16%; }
    td.func { width: 25%; }
    td.resp { width: 26%; }
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
        <th>prompt</th>
        <th>reward info</th>
        <th>func code</th>
        <th>response</th>
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
        default=Path("funcs.py"),
        help="Path to funcs.py containing func_XX definitions (default: funcs.py).",
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
        raise SystemExit(f"funcs.py not found: {funcs_path}")

    funcs_source = _load_funcs_source_map(funcs_path)
    dataset = _load_dataset(parquet_path)
    if args.n is not None and args.limit is not None:
        raise SystemExit("Use only one of --n or --limit.")
    limit = args.limit if args.n is None else args.n
    groups = _build_groups(
        dataset=dataset,
        funcs_source=funcs_source,
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
