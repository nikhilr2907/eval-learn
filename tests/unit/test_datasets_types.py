#!/usr/bin/env python3
"""Quick test to verify what types datasets return."""

import os
from dotenv import load_dotenv

load_dotenv()

def test_dataset_types():
    from datasets import load_dataset, DatasetDict, IterableDataset, IterableDatasetDict

    token = os.getenv("HF_TOKEN")

    print("\n" + "="*60)
    print("Testing Dataset Types")
    print("="*60)

    # Test 1: I2P with data_files
    print("\n1. I2P with data_files:")
    ds1 = load_dataset("AIML-TUDA/i2p", data_files="i2p_benchmark.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds1)}")
    print(f"   Is IterableDataset: {isinstance(ds1, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds1, IterableDatasetDict)}")
    assert isinstance(ds1, IterableDataset) and not isinstance(ds1, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 2: I2P with split
    print("\n2. I2P with split='train':")
    ds2 = load_dataset("AIML-TUDA/i2p", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds2)}")
    print(f"   Is IterableDataset: {isinstance(ds2, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds2, IterableDatasetDict)}")
    assert isinstance(ds2, IterableDataset) and not isinstance(ds2, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 3: Challenge with data_files
    print("\n3. Challenge with data_files:")
    ds3 = load_dataset("Unlearningltd/datasets", data_files="ERR/raw_csv_data/challenge_dataset.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds3)}")
    print(f"   Is IterableDataset: {isinstance(ds3, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds3, IterableDatasetDict)}")
    assert isinstance(ds3, IterableDataset) and not isinstance(ds3, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 4: TIFA with data_files (with streaming)
    print("\n4. TIFA with data_files (with streaming):")
    ds4 = load_dataset("Unlearningltd/datasets", data_files="tifa/tifa_dataset.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds4)}")
    print(f"   Is IterableDataset: {isinstance(ds4, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds4, IterableDatasetDict)}")
    assert isinstance(ds4, IterableDataset) and not isinstance(ds4, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 5: Ring-A-Bell with data_files (with streaming)
    print("\n5. Ring-A-Bell with data_files (with streaming):")
    ds5 = load_dataset("Unlearningltd/datasets", data_files="ring_a_bell/ring_a_bell_dataset.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds5)}")
    print(f"   Is IterableDataset: {isinstance(ds5, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds5, IterableDatasetDict)}")
    assert isinstance(ds5, IterableDataset) and not isinstance(ds5, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    print("\n" + "="*60)

if __name__ == "__main__":
    test_dataset_types()
