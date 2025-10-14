import os
import pandas as pd
from datasets import load_from_disk

# Paths
MAP_FILE = "/srv/scratch/z5369417/Speech-Attribute-Transcription/data/Phoneme2att_camb_att_Diph_v1.csv"
INPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped/"
OUTPUT_DATASET = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped_ipa/"

# Columns
PHONEME_COLUMN = "phoneme_unsw"
NEW_COLUMN = "phoneme_unsw_ipa"

def load_arpa_to_ipa_map(map_file):
    """Load the ARPA→IPA conversion map from the CSV."""
    df = pd.read_csv(map_file)
    df["Phoneme_arpa"] = df["Phoneme_arpa"].astype(str).str.strip().str.lower()
    df["Phoneme_ipa"] = df["Phoneme_ipa"].astype(str).str.strip()
    mapping = dict(zip(df["Phoneme_arpa"], df["Phoneme_ipa"]))
    print(f"Loaded {len(mapping)} ARPA→IPA mappings.")
    return mapping

def convert_sequence(seq, mapping):
    """Convert an ARPAbet phoneme sequence (string or list) into IPA."""
    if isinstance(seq, str):
        tokens = seq.strip().split()
    elif isinstance(seq, list):
        tokens = seq
    else:
        return seq
    return " ".join(mapping.get(t.lower(), t) for t in tokens)

def process_dataset(input_path, output_path, mapping, column=PHONEME_COLUMN, new_col=NEW_COLUMN):
    print(f"Loading dataset from: {input_path}")
    ds = load_from_disk(input_path)

    def convert_batch(batch):
        batch[new_col] = [convert_sequence(seq, mapping) for seq in batch[column]]
        return batch

    for split in ds:
        print(f"Processing split: {split}")
        ds[split] = ds[split].map(convert_batch, batched=True, desc=f"Converting {column} to IPA")

    print(f"Saving converted dataset to: {output_path}")
    ds.save_to_disk(output_path)

if __name__ == "__main__":
    mapping = load_arpa_to_ipa_map(MAP_FILE)
    process_dataset(INPUT_DATASET, OUTPUT_DATASET, mapping)
