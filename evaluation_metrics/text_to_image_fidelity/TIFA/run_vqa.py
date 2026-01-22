import json
import os
from tqdm import tqdm
from data.load_tifa import load_tifa_v1
from models.vqa import VQAModel

# Set Hugging Face cache to a directory with space
os.environ['HF_HUB_CACHE'] = '/data2/zh3225/hf_cache'

script_dir = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(script_dir, "data", "tifa_v1.0_text_inputs.json")
QA_PATH   = os.path.join(script_dir, "data", "tifa_v1.0_question_answers.json")
IMG_DIR   = os.path.join(script_dir, "images", "sd15")
OUT_PATH  = os.path.join(script_dir, "vqa_results.json")

data = load_tifa_v1(TEXT_PATH, QA_PATH)
vqa = VQAModel()

results = []

for item in tqdm(data[:10]):# 10 samples, for demo
    img_path = os.path.join(IMG_DIR, f"{item['id']}.png")

    if not os.path.exists(img_path):
        continue

    for qa in item["qas"]:
        pred = vqa.answer(img_path, qa["question"])

        results.append({
            "id": item["id"],
            "question": qa["question"],
            "gt": qa["answer"],
            "pred": pred
        })

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
