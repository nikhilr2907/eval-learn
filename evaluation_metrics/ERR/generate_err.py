# evaluation_metrics/ERR/generate_err_data.py
"""
Script to generate the ERR benchmark dataset by:
1. Downloading datasets (I2P, Ring-A-Bell, Challenge)
2. Creating prompt CSV files for target/retain/adversarial categories
3. Saving to data/ERR/prompt_data/
"""

import csv
import os
from datasets import load_dataset


class ERRDataGenerator:
    """
    Generates the ERR benchmark dataset files.
    Creates CSV files with prompts for the three ERR categories.
    """
    
    def __init__(self, base_dir: str = None, num_bad_prompts: int = 100):
        """
        Initialize ERR Data Generator.
        
        Args:
            base_dir: Base directory of the project
            num_bad_prompts: Number of bad prompts to use from I2P and Ring-A-Bell
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.base_dir = base_dir
        self.num_bad_prompts = num_bad_prompts
        
        # Setup directories
        self.raw_csv_dir = os.path.join(base_dir, "data", "ERR", "raw_csv_data")
        self.prompt_data_dir = os.path.join(base_dir, "data", "ERR", "prompt_data")
        
        # Dataset paths
        self.good_dataset_path = os.path.join(self.raw_csv_dir, "challenge_dataset.csv")
        self.i2p_csv_path = os.path.join(self.raw_csv_dir, "i2p_dataset.csv")
        self.rab_csv_path = os.path.join(self.raw_csv_dir, "ring_a_bell_dataset.csv")
        
        # Output paths
        self.retain_csv_path = os.path.join(self.prompt_data_dir, "retain_prompts.csv")
        self.target_csv_path = os.path.join(self.prompt_data_dir, "target_prompts.csv")
        self.adversarial_csv_path = os.path.join(self.prompt_data_dir, "adversarial_prompts.csv")
        
        # Create directories
        os.makedirs(self.raw_csv_dir, exist_ok=True)
        os.makedirs(self.prompt_data_dir, exist_ok=True)
        
        print("ERR Data Generator initialized")
        print(f"  Base directory: {base_dir}")
        print(f"  Output directory: {self.prompt_data_dir}")
    
    def download_and_save_datasets(self):
        """Download datasets from HuggingFace and save as CSV."""
        print("\n" + "=" * 80)
        print("STEP 1: Downloading and saving datasets as CSV")
        print("=" * 80)
        
        # Download I2P dataset
        print("\nDownloading I2P dataset...")
        i2p_dataset = load_dataset("AIML-TUDA/i2p", split="train")
        print(f"  Loaded {len(i2p_dataset)} prompts from I2P")
        
        with open(self.i2p_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "prompt", "categories"])
            for idx, item in enumerate(i2p_dataset):
                writer.writerow([idx, item.get("prompt", ""), item.get("categories", "")])
        print(f"  Saved to {self.i2p_csv_path}")
        
        # Download Ring-A-Bell dataset
        print("\nDownloading Ring-A-Bell dataset...")
        try:
            rab_dataset = load_dataset("AIML-TUDA/ring-a-bell", split="train")
            print(f"  Loaded {len(rab_dataset)} prompts from Ring-A-Bell")
            
            first_item = rab_dataset[0]
            columns = list(first_item.keys())
            
            with open(self.rab_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["index"] + columns)
                for idx, item in enumerate(rab_dataset):
                    row = [idx] + [item.get(col, "") for col in columns]
                    writer.writerow(row)
            print(f"  Saved to {self.rab_csv_path}")
            return i2p_dataset, rab_dataset
            
        except Exception as e:
            print(f"  Warning: Could not load Ring-A-Bell dataset: {e}")
            return i2p_dataset, None
    
    def generate_retention_prompts(self):
        """
        Generate retention prompts from challenge dataset.
        These are benign concepts that should STILL be generated correctly.
        """
        print("\n" + "=" * 80)
        print("STEP 2: Generating RETENTION prompts (benign concepts)")
        print("=" * 80)
        
        if not os.path.exists(self.good_dataset_path):
            print(f"Error: Challenge dataset not found at {self.good_dataset_path}")
            return []
        
        with open(self.good_dataset_path, 'r', encoding='utf-8') as f:
            challenge_dataset = list(csv.DictReader(f))
        
        print(f"  Loaded {len(challenge_dataset)} concepts from challenge dataset")
        
        prompts = []
        with open(self.retain_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["index", "prompt", "section", "concept_name"])
            
            for idx, row in enumerate(challenge_dataset):
                concept_name = row["concept_name"]
                prompt = row["direct_prompt"]
                
                writer.writerow([idx, prompt, "retain", concept_name])
                prompts.append(prompt)
        
        print(f"  Generated {len(prompts)} retention prompts")
        print(f"  Saved to {self.retain_csv_path}")
        return prompts
    
    def generate_target_prompts(self, i2p_dataset):
        """
        Generate target (forgetting) prompts from I2P dataset.
        These are direct inappropriate prompts that should NOT generate bad content.
        """
        print("\n" + "=" * 80)
        print("STEP 3: Generating TARGET prompts (I2P direct bad prompts)")
        print("=" * 80)
        
        num_i2p = min(self.num_bad_prompts, len(i2p_dataset))
        print(f"  Using {num_i2p} prompts from I2P dataset")
        
        prompts = []
        with open(self.target_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["index", "prompt", "section", "category"])
            
            for idx in range(num_i2p):
                item = i2p_dataset[idx]
                prompt = item["prompt"]
                category = item.get("categories", "unknown")
                
                writer.writerow([idx, prompt, "target", category])
                prompts.append(prompt)
        
        print(f"  Generated {len(prompts)} target prompts")
        print(f"  Saved to {self.target_csv_path}")
        return prompts
    
    def generate_adversarial_prompts(self, rab_dataset):
        """
        Generate adversarial prompts from Ring-A-Bell dataset.
        These are adversarial prompts designed to bypass safety filters.
        """
        print("\n" + "=" * 80)
        print("STEP 4: Generating ADVERSARIAL prompts (Ring-A-Bell)")
        print("=" * 80)
        
        if rab_dataset is None:
            print("  Ring-A-Bell dataset not available, skipping adversarial generation")
            return []
        
        num_rab = min(self.num_bad_prompts, len(rab_dataset))
        print(f"  Using {num_rab} prompts from Ring-A-Bell dataset")
        
        prompts = []
        with open(self.adversarial_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["index", "prompt", "section", "category"])
            
            for idx in range(num_rab):
                item = rab_dataset[idx]
                prompt = item.get("prompt", item.get("text", str(item)))
                category = item.get("category", item.get("concept", "adversarial"))
                
                writer.writerow([idx, prompt, "adversarial", category])
                prompts.append(prompt)
        
        print(f"  Generated {len(prompts)} adversarial prompts")
        print(f"  Saved to {self.adversarial_csv_path}")
        return prompts
    
    def generate_all(self):
        """Generate all ERR benchmark data files."""
        print("=" * 80)
        print("ERR BENCHMARK DATA GENERATION")
        print("=" * 80)
        
        # Step 1: Download datasets
        i2p_dataset, rab_dataset = self.download_and_save_datasets()
        
        # Step 2-4: Generate prompts
        retain_prompts = self.generate_retention_prompts()
        target_prompts = self.generate_target_prompts(i2p_dataset)
        adversarial_prompts = self.generate_adversarial_prompts(rab_dataset)
        
        print("\n" + "=" * 80)
        print("DATA GENERATION COMPLETE")
        print("=" * 80)
        print(f"\nTotal prompts generated:")
        print(f"  Retention:    {len(retain_prompts)}")
        print(f"  Target:       {len(target_prompts)}")
        print(f"  Adversarial:  {len(adversarial_prompts)}")
        print(f"  TOTAL:        {len(retain_prompts) + len(target_prompts) + len(adversarial_prompts)}")
        print(f"\nFiles saved to: {self.prompt_data_dir}")


def main():
    """Entry point for data generation script."""
    generator = ERRDataGenerator(num_bad_prompts=100)
    generator.generate_all()


if __name__ == "__main__":
    main()