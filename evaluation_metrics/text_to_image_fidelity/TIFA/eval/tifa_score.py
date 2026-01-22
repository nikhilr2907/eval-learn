import json
from collections import defaultdict
from eval.answer_match import is_correct

def compute_tifa(vqa_result_path):
    """
    compute TIFA score
    in：vqa_results.json
    out：
        - overall: average TIFA
        - per_prompt: dict {prompt_id: score}
    """
    with open(vqa_result_path, "r") as f:
        data = json.load(f)

    scores = defaultdict(list)

    for r in data:
        scores[r["id"]].append(is_correct(r["pred"], r["gt"]))

    tifa_per_prompt = {k: sum(v)/len(v) for k, v in scores.items()}
    overall = sum(tifa_per_prompt.values()) / len(tifa_per_prompt)

    return overall, tifa_per_prompt
