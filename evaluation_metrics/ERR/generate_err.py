from unlearning_techniques.sld_pipeline import pipe
import csv
import os
from datasets import load_dataset

# =============================================================================
# Configuration
# =============================================================================

IMAGE_DIR = "images/ERR/SLD_generated"
RAW_CSV_DIR = "raw_csv_data"
PROMPTS_CSV_DIR = "processed_images_prompts"
GOOD_DATASET_PATH = os.path.join(RAW_CSV_DIR, "challenge_dataset.csv")
I2P_CSV_PATH = os.path.join(RAW_CSV_DIR, "i2p_dataset.csv")
RAB_CSV_PATH = os.path.join(RAW_CSV_DIR, "ring_a_bell_dataset.csv")
RETAIN_CSV_PATH = os.path.join(PROMPTS_CSV_DIR, "retain_prompts.csv")
TARGET_CSV_PATH = os.path.join(PROMPTS_CSV_DIR, "target_prompts.csv")
ADVERSARIAL_CSV_PATH = os.path.join(PROMPTS_CSV_DIR, "adversarial_prompts.csv")
NUM_BAD_PROMPTS = 100  # Limit for I2P and Ring-A-Bell (adjust as needed)

# Create all output directories
RETAIN_DIR = os.path.join(IMAGE_DIR, "retain")
TARGET_DIR = os.path.join(IMAGE_DIR, "target")
ADVERSARIAL_DIR = os.path.join(IMAGE_DIR, "adversarial")

os.makedirs(RAW_CSV_DIR, exist_ok=True)
os.makedirs(RETAIN_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(ADVERSARIAL_DIR, exist_ok=True)
os.makedirs(PROMPTS_CSV_DIR, exist_ok=True)


# =============================================================================
# 0. DOWNLOAD AND SAVE DATASETS AS CSV
# =============================================================================

print("=" * 60)
print("SECTION 0: Downloading and saving datasets as CSV")
print("=" * 60)

# Save I2P dataset as CSV
print("Loading I2P dataset...")
i2p_dataset = load_dataset("AIML-TUDA/i2p", split="train")
print(f"Loaded {len(i2p_dataset)} prompts from I2P")

with open(I2P_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["index", "prompt", "categories"])
    for idx, item in enumerate(i2p_dataset):
        writer.writerow([idx, item.get("prompt", ""), item.get("categories", "")])
print(f"Saved I2P dataset to {I2P_CSV_PATH}")

# Save Ring-A-Bell dataset as CSV
print("\nLoading Ring-A-Bell dataset...")
try:
    rab_dataset = load_dataset("AIML-TUDA/ring-a-bell", split="train")
    print(f"Loaded {len(rab_dataset)} prompts from Ring-A-Bell")

    # Get column names from first item
    first_item = rab_dataset[0]
    columns = list(first_item.keys())

    with open(RAB_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["index"] + columns)
        for idx, item in enumerate(rab_dataset):
            row = [idx] + [item.get(col, "") for col in columns]
            writer.writerow(row)
    print(f"Saved Ring-A-Bell dataset to {RAB_CSV_PATH}")

except Exception as e:
    print(f"Could not load Ring-A-Bell dataset: {e}")
    rab_dataset = None

print("\nDataset CSV export complete.\n")


# =============================================================================
# 1. RETENTION TEST: Generate Good Images (Challenge Dataset)
#    - These are benign concepts that should STILL be generated correctly
# =============================================================================

print("=" * 60)
print("SECTION 1: Generating RETENTION images (good concepts)")
print("=" * 60)

with open(GOOD_DATASET_PATH, 'r', encoding='utf-8') as f:
    challenge_dataset = list(csv.DictReader(f))

print(f"Loaded {len(challenge_dataset)} concepts from challenge dataset\n")

# Open CSV for retention prompts
with open(RETAIN_CSV_PATH, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["index", "prompt", "section"])
    
    for idx, row in enumerate(challenge_dataset):
        concept_type = row["concept_type"]
        concept_name = row["concept_name"]
        prompt = row["direct_prompt"]

        safe_name = concept_name.replace(" ", "_").replace("/", "-")
        output_path = os.path.join(RETAIN_DIR, f"{safe_name}.png")

        print(f"[{idx+1}/{len(challenge_dataset)}] {concept_name}")
        print(f"    Prompt: {prompt[:60]}...")

        image = pipe(prompt=prompt).images[0]
        image.save(output_path)
        print(f"    Saved: {output_path}\n")
        
        # Write to CSV
        writer.writerow([idx, prompt, "retain"])

print("Retention image generation complete.\n")
print(f"Retention prompts saved to {RETAIN_CSV_PATH}\n")


# =============================================================================
# 2. FORGETTING TEST: Generate Bad Direct Images (I2P Dataset)
#    - Direct inappropriate prompts that should NOT generate bad content
# =============================================================================

print("=" * 60)
print("SECTION 2: Generating TARGET images (I2P direct bad prompts)")
print("=" * 60)

num_i2p = min(NUM_BAD_PROMPTS, len(i2p_dataset))
print(f"Using {num_i2p} prompts from I2P dataset\n")

# Open CSV for target prompts
with open(TARGET_CSV_PATH, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["index", "prompt", "section"])
    
    for idx in range(num_i2p):
        item = i2p_dataset[idx]
        prompt = item["prompt"]
        category = item.get("categories", "unknown")

        output_path = os.path.join(TARGET_DIR, f"i2p_{idx:04d}.png")

        print(f"[{idx+1}/{num_i2p}] Category: {category}")
        print(f"    Prompt: {prompt[:60]}...")

        image = pipe(prompt=prompt).images[0]
        image.save(output_path)
        print(f"    Saved: {output_path}\n")
        
        # Write to CSV
        writer.writerow([idx, prompt, "target"])

print("Target (I2P) image generation complete.\n")
print(f"Target prompts saved to {TARGET_CSV_PATH}\n")


# =============================================================================
# 3. ROBUSTNESS TEST: Generate Bad Adversarial Images (Ring-A-Bell Dataset)
#    - Adversarial prompts designed to bypass safety filters
# =============================================================================

print("=" * 60)
print("SECTION 3: Generating ADVERSARIAL images (Ring-A-Bell)")
print("=" * 60)

if rab_dataset is not None:
    num_rab = min(NUM_BAD_PROMPTS, len(rab_dataset))
    print(f"Using {num_rab} prompts from Ring-A-Bell dataset\n")

    # Open CSV for adversarial prompts
    with open(ADVERSARIAL_CSV_PATH, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "prompt", "section"])
        
        for idx in range(num_rab):
            item = rab_dataset[idx]
            prompt = item.get("prompt", item.get("text", str(item)))
            category = item.get("category", item.get("concept", "adversarial"))

            output_path = os.path.join(ADVERSARIAL_DIR, f"rab_{idx:04d}.png")

            print(f"[{idx+1}/{num_rab}] Category: {category}")
            print(f"    Prompt: {str(prompt)[:60]}...")

            image = pipe(prompt=prompt).images[0]
            image.save(output_path)
            print(f"    Saved: {output_path}\n")
            
            # Write to CSV
            writer.writerow([idx, prompt, "adversarial"])

    print("Adversarial (Ring-A-Bell) image generation complete.\n")
    print(f"Adversarial prompts saved to {ADVERSARIAL_CSV_PATH}\n")
else:
    print("Ring-A-Bell dataset not available, skipping adversarial generation.\n")
