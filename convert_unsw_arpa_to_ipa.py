import os
import pandas as pd
from datasets import load_from_disk

# === Paths ===
MAP_FILE = "/srv/scratch/z5369417/Speech-Attribute-Transcription/data/Phoneme2att_camb_att_Diph_v1.csv"
INPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped/"
OUTPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped_ipa/"

# === Only check these two ===
PHONEME_CHECK_COLUMNS = ["phoneme_unsw", "actual_spoken_phonemes"]

# === Convert these columns (same as before) ===
PHONEME_COLUMNS = ["text", "phoneme_unsw", "actual_spoken_phonemes", "aligned_phonemes"]

def load_arpa_to_ipa_map(map_file):
    """Load ARPA→IPA mapping from the CSV file."""
    df = pd.read_csv(map_file)
    df["Phoneme_arpa"] = df["Phoneme_arpa"].astype(str).str.strip().str.lower()
    df["Phoneme_ipa"] = df["Phoneme_ipa"].astype(str).str.strip()
    mapping = dict(zip(df["Phoneme_arpa"], df["Phoneme_ipa"]))
    print(f"Loaded {len(mapping)} ARPA→IPA mappings.")
    return mapping, set(mapping.keys())

def convert_sequence(seq, mapping):
    """Convert a space-separated or list-based ARPA phoneme sequence into IPA."""
    if not seq:
        return seq
    if isinstance(seq, list):
        seq = " ".join(str(x) for x in seq)
    tokens = seq.strip().split()
    return " ".join(mapping.get(t.lower(), t) for t in tokens)

def sample_has_only_valid_phonemes(example, valid_set):
    """Check only phoneme_unsw & actual_spoken_phonemes."""
    for col in PHONEME_CHECK_COLUMNS:
        if col not in example:
            continue
        tokens = str(example[col]).strip().split()
        if any(p.lower() not in valid_set for p in tokens):
            return False
    return True

def process_dataset(input_path, output_path, mapping, valid_set):
    print(f"Loading dataset from: {input_path}")
    ds = load_from_disk(input_path)

    # Filter invalid samples first
    for split in ds:
        print(f"\nFiltering split: {split}")
        before = len(ds[split])
        ds[split] = ds[split].filter(
            lambda ex: sample_has_only_valid_phonemes(ex, valid_set),
            desc="Removing samples with invalid phonemes"
        )
        after = len(ds[split])
        print(f"→ Kept {after}/{before} samples")

    # Now convert remaining
    print("\nStarting ARPA → IPA conversion...")
    first_split = next(iter(ds.keys()))
    existing_cols = ds[first_split].column_names
    cols_to_convert = [c for c in PHONEME_COLUMNS if c in existing_cols]

    def convert_batch(batch):
        for col in cols_to_convert:
            batch[col] = [convert_sequence(seq, mapping) for seq in batch[col]]
        return batch

    for split in ds:
        print(f"Converting split: {split}")
        ds[split] = ds[split].map(convert_batch, batched=True, desc="Converting ARPA→IPA")

    print(f"\nSaving filtered + converted dataset to: {output_path}")
    ds.save_to_disk(output_path)

if __name__ == "__main__":
    mapping, valid_set = load_arpa_to_ipa_map(MAP_FILE)
    process_dataset(INPUT_DATASET, OUTPUT_DATASET, mapping, valid_set)
