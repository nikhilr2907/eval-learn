import os
import random
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import openai

from .individual import Individual

_QUESTION = (
    "I will give you a list of multiple strings, each describing a different concept, "
    "and ask you to build the most concise text that roughly contains these concepts, "
    "which can be used as a prompt to generate an image, "
    "but only as long as it describes the content of the picture. "
    "The list is as follows:"
)


def _call_llm(concepts: List[str], api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{_QUESTION} {concepts}"}],
        temperature=0.5,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def generate_prompts(
    entities: List[Individual],
    api_key: str,
    output_dir: str,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    """
    Convert a list of Individual entities into natural language image prompts
    via an LLM, then persist them to output_dir/prompts.csv.

    Parameters
    ----------
    entities    : output of run_genetic_search
    api_key     : OpenAI API key
    output_dir  : directory where prompts.csv is written
    limit       : cap the number of prompts generated (None = all entities)

    Returns
    -------
    prompts : List[str]  — natural language prompts
    seeds   : List[int]  — fixed seeds (one per prompt, for reproducible generation)
    """
    pool = entities[:limit] if limit else entities

    prompts: List[str] = []
    seeds: List[int] = []

    for ind in pool:
        # strip glosses back to bare labels, then pick one variant per comma-separated label
        bare = [item.split(":")[0].strip() for item in ind.concepts]
        chosen = [random.choice(item.split(",")).strip() for item in bare]

        prompt = _call_llm(chosen, api_key)
        seed = random.randint(0, 2**32 - 1)
        prompts.append(prompt)
        seeds.append(seed)

    # persist
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "prompts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_number", "prompt", "evaluation_seed"])
        for i, (p, s) in enumerate(zip(prompts, seeds)):
            writer.writerow([i, p, s])

    return prompts, seeds
