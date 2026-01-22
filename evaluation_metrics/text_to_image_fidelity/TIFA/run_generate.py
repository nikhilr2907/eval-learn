import os
from tqdm import tqdm
from data.load_tifa import load_tifa_v1
from models.generate import ImageGenerator

TEXT_PATH = "data/tifa_v1.0_text_inputs.json"
QA_PATH   = "data/tifa_v1.0_question_answers.json"
OUT_DIR   = "images/sd15"

os.makedirs(OUT_DIR, exist_ok=True)

data = load_tifa_v1(TEXT_PATH, QA_PATH)
generator = ImageGenerator()

for item in tqdm(data[:10]): # Sample 10 prompts, for demo
    img_path = os.path.join(OUT_DIR, f"{item['id']}.png")

    if os.path.exists(img_path):
        continue

    image = generator.generate(item["prompt"])
    image.save(img_path)
