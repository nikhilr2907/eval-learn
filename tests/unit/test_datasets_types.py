#!/usr/bin/env python3
"""Quick test to verify what types datasets return."""

import os
from dotenv import load_dotenv

load_dotenv()

def test_dataset_types():
    from datasets import load_dataset, DatasetDict, IterableDataset

    token = os.getenv("HF_TOKEN")

    print("\n" + "="*60)
    print("Testing Dataset Types")
    print("="*60)

    # Test 1: I2P with data_files
    print("\n1. I2P with data_files:")
    ds1 = load_dataset("AIML-TUDA/i2p", data_files="i2p_benchmark.csv", streaming=True, token=token)
    print(f"   Type: {type(ds1)}")
    print(f"   Is DatasetDict: {isinstance(ds1, DatasetDict)}")
    print(f"   Is IterableDataset: {isinstance(ds1, IterableDataset)}")

    # Test 2: I2P with split
    print("\n2. I2P with split='train':")
    ds2 = load_dataset("AIML-TUDA/i2p", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds2)}")
    print(f"   Is DatasetDict: {isinstance(ds2, DatasetDict)}")
    print(f"   Is IterableDataset: {isinstance(ds2, IterableDataset)}")

    # Test 3: Challenge with data_files
    print("\n3. Challenge with data_files:")
    ds3 = load_dataset("Unlearningltd/datasets", data_files="ERR/raw_csv_data/challenge_dataset.csv", streaming=True, token=token)
    print(f"   Type: {type(ds3)}")
    print(f"   Is DatasetDict: {isinstance(ds3, DatasetDict)}")

    # Test 4: TIFA text with data_files (NO streaming)
    print("\n4. TIFA text with data_files (no streaming):")
    ds4 = load_dataset("Unlearningltd/datasets", data_files="tifa/sensitive_text_inputs.json", token=token)
    print(f"   Type: {type(ds4)}")
    print(f"   Is DatasetDict: {isinstance(ds4, DatasetDict)}")
    if isinstance(ds4, DatasetDict):
        print(f"   Splits: {list(ds4.keys())}")

    # Test 5: TIFA QA with data_files (NO streaming)
    print("\n5. TIFA QA with data_files (no streaming):")
    ds5 = load_dataset("Unlearningltd/datasets", data_files="tifa/sensitive_question_answers.json", token=token)
    print(f"   Type: {type(ds5)}")
    print(f"   Is DatasetDict: {isinstance(ds5, DatasetDict)}")
    if isinstance(ds5, DatasetDict):
        print(f"   Splits: {list(ds5.keys())}")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_dataset_types()
