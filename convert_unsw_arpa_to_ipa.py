import os
import pandas as pd
from datasets import load_from_disk

# === Paths ===
MAP_FILE = "/srv/scratch/z5369417/Speech-Attribute-Transcription/data/Phoneme2att_camb_att_Diph_v1.csv"
INPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped/"
OUTPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped_ipa/"

# === Columns containing phoneme strings ===
PHONEME_COLUMNS = ["text", "phoneme_unsw", "actual_spoken_phonemes", "aligned_phonemes"]

def load_arpa_to_ipa_map(map_file):
    """Load ARPA→IPA mapping from the CSV file."""
    df = pd.read_csv(map_file)
    df["Phoneme_arpa"] = df["Phoneme_arpa"].astype(str).str.strip().str.lower()
    df["Phoneme_ipa"] = df["Phoneme_ipa"].astype(str).str.strip()
    mapping = dict(zip(df["Phoneme_arpa"], df["Phoneme_ipa"]))
    print(f"Loaded {len(mapping)} ARPA→IPA mappings.")
    return mapping

def convert_sequence(seq, mapping):
    """Convert a space-separated or list-based ARPA phoneme sequence into IPA."""
    if not seq:
        return seq
    if isinstance(seq, list):
        seq = " ".join(str(x) for x in seq)
    tokens = seq.strip().split()
    return " ".join(mapping.get(t.lower(), t) for t in tokens)

def process_dataset(input_path, output_path, mapping):
    print(f"Loading dataset from: {input_path}")
    ds = load_from_disk(input_path)

    # Identify existing columns that need conversion
    first_split = next(iter(ds.keys()))
    existing_cols = ds[first_split].column_names
    cols_to_convert = [c for c in PHONEME_COLUMNS if c in existing_cols]
    print(f"Columns to convert: {cols_to_convert}")

    def convert_batch(batch):
        for col in cols_to_convert:
            batch[col] = [convert_sequence(seq, mapping) for seq in batch[col]]
        return batch

    for split in ds:
        print(f"Processing split: {split}")
        ds[split] = ds[split].map(convert_batch, batched=True, desc="Converting ARPA→IPA")

    print(f"Saving IPA-converted dataset to: {output_path}")
    ds.save_to_disk(output_path)

if __name__ == "__main__":
    mapping = load_arpa_to_ipa_map(MAP_FILE)
    process_dataset(INPUT_DATASET, OUTPUT_DATASET, mapping)
