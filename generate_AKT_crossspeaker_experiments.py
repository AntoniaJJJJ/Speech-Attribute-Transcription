"""
Generate 20 cross-speaker AKT experiments after phonemization

This script:
  1. Loads the phonemized AKT dataset (DatasetDict with 'train' and 'test')
  2. Uses speaker IDs to make 20 experiments:
        - 10 SSD IDs (fixed, from 'test')
        - 10 fixed non-SSD IDs (from 'train')
        - 16 rotating non-SSD IDs (unique per experiment)
  3. Saves each experiment as a DatasetDict under:
        /srv/scratch/z5369417/AKT_crossspeaker_experiments/AKT_exp{i}/dataset
  4. Creates a modified YAML config for each experiment
  5. Exports an Excel file summarizing the splits
"""

import os
import random
import pandas as pd
import yaml
from datasets import load_from_disk, DatasetDict

# === USER PATHS ===
BASE_PATH = "/srv/scratch/z5369417"
SOURCE_DATA = f"{BASE_PATH}/outputs/phonemization_AKT"
ORIGINAL_YAML = f"{BASE_PATH}/Speech-Attribute-Transcription/configs/exp14.yaml"
SAVE_ROOT = f"{BASE_PATH}/AKT_crossspeaker_experiments"
os.makedirs(SAVE_ROOT, exist_ok=True)

N_EXPERIMENTS = 20
FIXED_SSD_N = 10
FIXED_NONSSD_N = 10
ROTATING_NONSSD_N = 16

# === 1. Load dataset ===
dataset = load_from_disk(SOURCE_DATA)
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# Identify unique speaker IDs
ssd_ids = sorted(test_df["speaker_id"].unique())          # 10 SSD
nonssd_ids = sorted(train_df["speaker_id"].unique())      # 338 non-SSD
print(f"Total SSD IDs: {len(ssd_ids)}, non-SSD IDs: {len(nonssd_ids)}")

# === 2. Select fixed and rotating IDs ===
random.seed(42)
fixed_nonssd = random.sample(nonssd_ids, FIXED_NONSSD_N)
remaining_nonssd = [i for i in nonssd_ids if i not in fixed_nonssd]
print(f"Fixed non-SSD IDs: {fixed_nonssd}")

# Split remaining 328 non-SSD into 20 non-overlapping groups of 16
random.shuffle(remaining_nonssd)
rotating_groups = [remaining_nonssd[i*ROTATING_NONSSD_N:(i+1)*ROTATING_NONSSD_N] for i in range(N_EXPERIMENTS)]

# === 3. Generate experiments ===
records = []
from datasets import concatenate_datasets

for i in range(N_EXPERIMENTS):
    exp_name = f"AKT_exp{i+1}"
    exp_dir = os.path.join(SAVE_ROOT, exp_name)
    dataset_dir = os.path.join(exp_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Test set = 10 SSD + 10 fixed nonSSD + 16 rotating nonSSD
    test_ids = ssd_ids + fixed_nonssd + rotating_groups[i]
    train_ids = [id for id in nonssd_ids if id not in test_ids]

    # Filter datasets
    ds_train = dataset["train"].filter(lambda x: x["speaker_id"] in train_ids)
    ds_test_nonssd = dataset["train"].filter(lambda x: x["speaker_id"] in (fixed_nonssd + rotating_groups[i]))
    ds_test_ssd = dataset["test"]  # 10 SSD speakers
    ds_test = concatenate_datasets([ds_test_nonssd, ds_test_ssd])

    # Save DatasetDict
    exp_dataset = DatasetDict({"train": ds_train, "test": ds_test})
    exp_dataset.save_to_disk(dataset_dir)

    # === 4. Modify YAML for this experiment ===
    with open(ORIGINAL_YAML, "r") as f:
        yaml_config = yaml.safe_load(f)

    yaml_config["datasets"]["data_path"] = dataset_dir
    yaml_config["output"]["working_dir"] = os.path.join(exp_dir, "results")

    yaml_out = os.path.join(exp_dir, f"exp14_exp{i+1}.yaml")
    with open(yaml_out, "w") as f:
        yaml.dump(yaml_config, f, sort_keys=False)

    records.append({
        "Experiment": exp_name,
        "Train_IDs": ", ".join(map(str, train_ids)),
        "Test_IDs": ", ".join(map(str, test_ids))
    })

    print(f"Saved {exp_name}: {len(train_ids)} train, {len(test_ids)} test IDs")

# === 5. Save Excel summary ===
summary_path = os.path.join(SAVE_ROOT, "experiment_splits_summary.xlsx")
pd.DataFrame(records).to_excel(summary_path, index=False)
print(f"\nAll experiments created successfully")
print(f"Summary file saved: {summary_path}")
