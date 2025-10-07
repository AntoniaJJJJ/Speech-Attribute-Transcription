"""
Author: Antonia Jian
Date: 07/10/2025
Description:
This script identifies and counts words in the AKT Task 3 dataset that are missing
from the Australian English phoneme mapping (AusKidTalk_transcription.xlsx).

It counts frequency and saves results to a CSV file.
"""

import os
import re
import pandas as pd
from collections import defaultdict, Counter
from datasets import load_from_disk

# === Load phoneme mapping (same logic as phonemization script) ===
def load_phoneme_mapping(file_path):
    df = pd.read_excel(file_path, sheet_name='transcription')
    phoneme_dict = defaultdict(list)
    for _, row in df.iterrows():
        word = str(row['word']).strip().lower()
        phoneme_dict[word] = row['transcription'].strip()
    return phoneme_dict


# === Extract all words from dataset ===
def extract_words_with_counts(dataset):
    word_counter = Counter()
    for split_name, split_data in dataset.items():
        print(f"🔹 Scanning split: {split_name} ({len(split_data)} rows)")
        for text in split_data['text']:
            words = re.sub(r"[^\w\s']", '', str(text).lower()).split()
            word_counter.update(words)
    return word_counter


# === Compare with phoneme mapping ===
def find_unknown_words_with_counts(word_counter, mapping):
    known_words = set(mapping.keys())
    unknown_counts = {}
    for word, count in word_counter.items():
        if word not in known_words:
            unknown_counts[word] = count
    return unknown_counts


# === Save results as CSV ===
def save_unknown_words_to_csv(unknown_counts, output_csv):
    df = pd.DataFrame(list(unknown_counts.items()), columns=["Word", "Frequency"])
    df.to_csv(output_csv, index=False)
    print(f" Saved {len(unknown_counts)} unknown words with frequencies to {output_csv}")


# === Main ===
if __name__ == "__main__":
    akt_dataset_path = "/srv/scratch/z5369417/created_dataset_AKT_task3"
    phoneme_mapping_file = "/srv/scratch/z5369417/AKT_data_processing/AusKidTalk_transcription.xlsx"
    output_csv = "/srv/scratch/z5369417/unknown_words_AKT_task3_counts.csv"

    dataset = load_from_disk(akt_dataset_path)
    mapping = load_phoneme_mapping(phoneme_mapping_file)

    word_counter = extract_words_with_counts(dataset)
    unknown_counts = find_unknown_words_with_counts(word_counter, mapping)
    save_unknown_words_to_csv(unknown_counts, output_csv)