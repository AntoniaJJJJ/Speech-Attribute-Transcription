import os
import pandas as pd
from datasets import load_from_disk

# === Paths ===
MAP_FILE = "/srv/scratch/z5369417/Speech-Attribute-Transcription/data/Phoneme2att_camb_att_Diph_v1.csv"
INPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped/"
OUTPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped_ipa/"

# === Only these columns are checked ===
PHONEME_COLUMNS = ["phoneme_unsw", "actual_spoken_phonemes"]

def load_valid_phoneme_set(map_file):
    """Load valid ARPA phonemes from the mapper CSV."""
    df = pd.read_csv(map_file)
    df["Phoneme_arpa"] = df["Phoneme_arpa"].astype(str).str.strip().str.lower()
    valid_set = set(df["Phoneme_arpa"])
    print(f"Loaded {len(valid_set)} valid phonemes.")
    return valid_set

def sample_has_only_valid_phonemes(example, valid_set):
    """Return True if all phonemes in phoneme_unsw and actual_spoken_phonemes are valid."""
    for col in PHONEME_COLUMNS:
        if col not in example:
            continue
        tokens = str(example[col]).strip().split()
        if any(p.lower() not in valid_set for p in tokens):
            return False
    return True

def process_dataset(input_path, output_path, valid_set):
    print(f"Loading dataset from: {input_path}")
    ds = load_from_disk(input_path)

    for split in ds:
        print(f"Filtering split: {split}")
        original_len = len(ds[split])
        ds[split] = ds[split].filter(
            lambda ex: sample_has_only_valid_phonemes(ex, valid_set),
            desc="Filtering invalid phoneme samples"
        )
        print(f"→ {len(ds[split])} / {original_len} samples kept in '{split}'")

    print(f"Saving filtered dataset to: {output_path}")
    ds.save_to_disk(output_path)

if __name__ == "__main__":
    valid_phonemes = load_valid_phoneme_set(MAP_FILE)
    process_dataset(INPUT_DATASET, OUTPUT_DATASET, valid_phonemes)
