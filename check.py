import pandas as pd
from datasets import load_from_disk

# === Config ===
dataset_path = "/srv/scratch/z5369417/outputs/phonemization_speechocean_exp11_4/"
mapping_csv_path = "/srv/scratch/z5369417/Speech-Attribute-Transcription/data/Phoneme2att_camb_att_noDiph.csv"

# === Load dataset ===
ds = load_from_disk(dataset_path)

# === Load valid phonemes ===
mapping_df = pd.read_csv(mapping_csv_path)
valid_phonemes = set(mapping_df["phoneme"].str.lower().tolist())

# === Collect all unique phonemes from the dataset ===
all_phonemes = set()
for split in ["train", "test"]:
    for sample in ds[split]:
        if sample["actual_spoken_phonemes"]:
            phonemes = sample["actual_spoken_phonemes"].strip().split()
            all_phonemes.update(phonemes)

# === Check for missing phonemes ===
missing_phonemes = sorted(ph for ph in all_phonemes if ph not in valid_phonemes)

# === Output ===
if missing_phonemes:
    print("🚫 Missing phonemes found (not in mapping):")
    for p in missing_phonemes:
        print(" -", p)
else:
    print("✅ All phonemes are valid and present in the mapping.")