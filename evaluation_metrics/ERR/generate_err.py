from unlearning_techniques.sld_pipeline import pipe
import csv
import os
from datasets import load_dataset

image_dir = "images/ERR/SLD_generated"
GOOD_DATASET_DIR = "raw_csv_data/challenge_dataset.csv"


####### Generate Good Images Using Challenge Dataset (Retention Test) ########

# Create output directory for retention images
os.makedirs(os.path.join(image_dir, "retain"), exist_ok=True)

# Parse challenge dataset
with open(GOOD_DATASET_DIR, 'r', encoding='utf-8') as f:
    challenge_dataset = list(csv.DictReader(f))

print(f"Loaded {len(challenge_dataset)} concepts from challenge dataset")

# Generate images for each concept (retention test)
for idx, row in enumerate(challenge_dataset):
    concept_type = row["concept_type"]
    concept_name = row["concept_name"]
    prompt = row["direct_prompt"]

    print(f"\n[{idx+1}/{len(challenge_dataset)}] Retention: {concept_name} ({concept_type})")

    # Create safe filename from concept name
    safe_name = concept_name.replace(" ", "_").replace("/", "-")
    output_path = os.path.join(image_dir, "retain", f"{safe_name}.png")

    print(f"  Generating: {prompt[:50]}...")

    # Generate image using pipeline
    image = pipe(prompt=prompt).images[0]

    # Save image
    image.save(output_path)
    print(f"  Saved to {output_path}")

print("\nRetention image generation complete.")


######## Generate Bad Images Using I2P Dataset (Forgetting Test) ########

print("\nLoading I2P dataset...")
i2p_dataset = load_dataset("AIML-TUDA/i2p", split="train")
print(f"Loaded {len(i2p_dataset)} prompts from I2P dataset")

# Create output directory for forgetting test images
os.makedirs(os.path.join(image_dir, "target"), exist_ok=True)

# Generate images for inappropriate prompts (forgetting test)
num_bad_prompts = min(100, len(i2p_dataset))  # Limit for testing, adjust as needed

for idx, item in enumerate(i2p_dataset):
    if idx >= num_bad_prompts:
        break

    prompt = item["prompt"]
    category = item.get("categories", "unknown")

    print(f"\n[{idx+1}/{num_bad_prompts}] Forgetting test: {prompt[:50]}... | Category: {category}")

    output_path = os.path.join(image_dir, "target", f"bad_prompt_{idx:04d}.png")

    # Generate image using pipeline
    image = pipe(prompt=prompt).images[0]

    # Save image
    image.save(output_path)
    print(f"  Saved to {output_path}")

print("\nForgetting image generation complete.")

