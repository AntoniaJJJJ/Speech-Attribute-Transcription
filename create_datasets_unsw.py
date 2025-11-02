"""
Author: Antonia Jian  
Date (Last Modified): 23/09/2025  
Description:  
This script processes the "UNSW Final Deliverables" dataset located under:
    /srv/scratch/z5369417/UNSW_final_deliverables/CAAP_2023-04-27/

The dataset includes:
- A spreadsheet (`dataset_spreadsheet.xlsx`) containing:
    • `audio_file`: filename
    • `word`: spoken word
    • `word_phonemes`: canonical phoneme sequence (used as input to train model)
    • `recording_phonemes`: actual spoken phonemes (for evaluation/MDD)
    • `aligned_phonemes`: aligned canonical > spoken pairs (for diagnostics)
    • `age`, `gender`, `speech_status`: speaker metadata

- Corresponding audio files under:
    • `wavs/non_disordered/` → used as the **train** split
    • `wavs/disordered/`     → used as the **test** split

This script:
- Loads the spreadsheet and matches `.wav` files to their metadata entries.
- Reads and attaches audio arrays using Hugging Face's `Audio` feature.
- Adds all metadata columns to each example.
- Combines all samples (both non-disordered and disordered) into a **single split** for easier downstream processing.
- Saves the dataset as a Hugging Face `DatasetDict` with a single `"test"` split, ensuring compatibility with evaluation and transcription pipelines.

Output format:
DatasetDict({
    "test": Dataset({
        features: ["audio", "text", "phoneme_unsw", "actual_spoken_phonemes", "aligned_phonemes", "word", "age", "gender", "speech_status"],
        ...
    })
})
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio
import numpy as np
import re

# === PATH SETUP ===
base_dir = "/srv/scratch/z5369417/UNSW_final_deliverables/CAAP_2023-04-27"
wav_dir = os.path.join(base_dir, "wavs")
spreadsheet_path = os.path.join(base_dir, "dataset_spreadsheet.xlsx")
output_dir = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped"  # <- wrapped as DatasetDict

# === Load the spreadsheet ===
df = pd.read_excel(spreadsheet_path)

# Normalize all filenames
df["audio_file"] = df["audio_file"].str.strip()

# Add full wav paths
df["wav_path"] = df["audio_file"].apply(lambda f: 
    os.path.join(wav_dir, "non_disordered", f) 
    if os.path.exists(os.path.join(wav_dir, "non_disordered", f))
    else os.path.join(wav_dir, "disordered", f)
)

# Filter rows with missing audio files
df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)

# Rename columns to match previous data preprocessing
df.rename(columns={
    "word_phonemes": "phoneme_unsw",                      # Canonical
    "recording_phonemes": "actual_spoken_phonemes",      # Spoken
    "word": "word",                                       # Word (unchanged)
    "aligned_phonemes": "aligned_phonemes",              # For diagnosis
    "speech_status": "speech_status",                    # 1 = disordered
    "age": "age",
    "gender": "gender",
}, inplace=True)

# Store results
data = []

# === Load each row into dict with audio data ===  
for _, row in df.iterrows():

    # Clean spoken phonemes: remove symbols like ?, *, ., etc.
    cleaned_spoken = " ".join([
        re.sub(r"[^\wɑɒæʌəɛɜɪiːɔʃθðŋʒʊuː]", "", p)
        for p in str(row["actual_spoken_phonemes"]).split()
    ])

    data.append({
        "audio": row["wav_path"],
        "text": row["phoneme_unsw"],  #  Canonical transcription
        "phoneme_unsw": row["phoneme_unsw"],
        "actual_spoken_phonemes": row["actual_spoken_phonemes"],
        "aligned_phonemes": row["aligned_phonemes"],
        "word": row["word"],
        "age": row["age"],
        "gender": row["gender"],
        "speech_status": row["speech_status"]
    })

# === Convert to HuggingFace Datasets ===
dataset = Dataset.from_list(data)

# Cast audio column
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Wrap in DatasetDict with 'test' split
dataset_dict = DatasetDict({"test": dataset})

# Save to disk
os.makedirs(output_dir, exist_ok=True)
dataset_dict.save_to_disk(output_dir)

print(f"UNSW dataset saved to: {output_dir}")
print(dataset_dict)
