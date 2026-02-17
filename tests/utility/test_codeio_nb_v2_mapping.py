import json


def test_codeio_preimage_func_aliases_nb_v2():
    from projects.datasets import create_string_data as create_string_data
    from verl.utils.reward_score.codeio import compute_score

    x = "012345"
    target = create_string_data.nb_00(x)

    # NB-v2 prompts/reference code still use `func_{xx}` identifiers.
    # `func_24` should resolve to `nb_00` at stage-2 runtime.
    ref_code = "def main_solution(x):\n    return func_24(x)\n"
    gt = {
        "ref_input": {"x": x},
        "ref_output": target,
        "ref_code": ref_code,
        "funcname": "main_solution",
    }

    score, _acc = compute_score(
        f"<answer>{x}</answer>",
        json.dumps(gt),
        task="codeio-nb-preimage-incomplete-depth1",
    )
    assert score == 1

