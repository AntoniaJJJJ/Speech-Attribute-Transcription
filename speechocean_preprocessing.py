"""
Author: Antonia Jian
Date(Last modified): 10/03/2025
Description: 
This script format SpeechOcean762 to match the CU dataset
- convert phonemes into sequences
- assign labels: 1 (correct pronunciation) or 0 (mispronounced, including heavy accent)
- Extract actual spoken phonemes from the "mispronunciations" block
- Filters only children aged 11 and below
- Handles special cases (`*`, `<unk>`, `<del>`)
- Generates phoneme- and sentence-level statistics

SpeechOcean762 dataset contains two subsets:
- Train set (ds["train"]): 2500 samples
- Test set (ds["test"]): 2500 samples

Only phonemes with an accuracy score lower than 0.5 have an explicit "mispronunciations" block

Each entry contains:
- Sentence-level scores: accuracy, completeness, fluency, prosodic, total
- Text transcription: text
- Word-level details:
    words: List of word annotations, each containing:
        Phonemes (phones) and their pronunciation accuracy (phones-accuracy)
        Stress information (stress)
        Mispronunciations list (mispronunciations, empty if no errors)
- Speaker metadata: speaker, gender, age
- Audio file: Stored as audio["array"] with sampling_rate = 16000

"""

import torch
import re
import pandas as pd
from datasets import DatasetDict, load_dataset

# Function to remove stress markers from vowels (AH0 -> AH, AH1 -> AH)
def remove_stress(phoneme):
    return re.sub(r'([a-z]+)[0-2]$', r'\1', phoneme)  # Removes final digit if it exists

# Statistics Dictionary
stats = {
    "train": {
        "total_sentences": 0,
        "mispronounced_sentences": 0,
        "correct_sentences": 0,
        "removed_sentences": 0,  # Due to special cases

        "total_phonemes": 0,
        "correct_phonemes": 0,
        "mispronounced_phonemes": 0,

        "phonemes_removed_star": 0,  # `*` cases (distortion)
        "phonemes_removed_unk": 0,   # `<unk>` cases
        "phonemes_removed_ar": 0,    # `ar` cases
        "phonemes_removed_ir": 0,    # `ir` cases

        "deletion": 0,
        "substitution": 0,
        "insertion": 0,
        "distortion": 0
    },
    "test": {
        "total_sentences": 0,
        "mispronounced_sentences": 0,
        "correct_sentences": 0,
        "removed_sentences": 0,  # Due to special cases

        "total_phonemes": 0,
        "correct_phonemes": 0,
        "mispronounced_phonemes": 0,

        "phonemes_removed_star": 0,  # `*` cases (distortion)
        "phonemes_removed_unk": 0,   # `<unk>` cases
        "phonemes_removed_ar": 0,    # `ar` cases
        "phonemes_removed_ir": 0,    # `ir` cases

        "deletion": 0,
        "substitution": 0,
        "insertion": 0,
        "distortion": 0
    },
    "total": {}  # Will be computed later
}

# Function to preprocess each sample
def preprocess_sample(sample, split):
    global stats
    # Extract phonemes and their correctness labels
    phonemes = []   # Canonical phonemes
    spoken_phonemes = []  # Actual spoken phonemes
    labels = []     # Correct = 1, Mispronounced = 0

    has_mispronunciation = False
    removed = False  # Flag for sentence removal
    
    for word in sample["words"]:
        for i, phone in enumerate(word["phones"]):
            clean_phone = remove_stress(phone.lower())  # Convert to lowercase and remove stress
            spoken_phone = clean_phone  # Default: assume spoken phoneme is the same
            accuracy = word["phones-accuracy"][i]

            # Handle mispronunciations
            if "mispronunciations" in word and word["mispronunciations"]:
                for misp in word["mispronunciations"]:
                    if misp["index"] == i:
                        spoken_phone = misp["pronounced-phone"].lower()

                        # **Remove if 'ar' or 'ir' found**
                        if spoken_phone in ["ar", "ir"]:
                            if spoken_phone == "ar":
                                stats[split]["phonemes_removed_ar"] += 1
                            if spoken_phone == "ir":
                                stats[split]["phonemes_removed_ir"] += 1
                            removed = True  # Mark for removal
                            continue

                        # **If `<unk>`, remove**
                        if spoken_phone == "<unk>":
                            stats[split]["phonemes_removed_unk"] += 1
                            removed = True
                            continue
                        
                        # **Track Mispronunciation Type**
                        if spoken_phone == "<del>":
                            stats[split]["deletion"] += 1
                            has_mispronunciation = True  # Mark sentence as mispronounced
                            continue  # Skip adding this phoneme
                        
                        """ # **If *, ambiguous pronounciation, remove **
                        if "*" in spoken_phone:  
                            stats[split]["phonemes_removed_star"] += 1
                            stats[split]["distortion"] += 1
                            removed = True
                            continue  # Remove this phoneme """

                        # **If *, ambiguous pronunciation: strip *, count as distortion, but keep phoneme**
                        if "*" in spoken_phone:
                            spoken_phone = spoken_phone.replace("*", "")
                            stats[split]["phonemes_removed_star"] += 1
                            stats[split]["distortion"] += 1
                            has_mispronunciation = True  # Mark sentence as mispronounced
                        
                        # **Substitution Case**
                        if spoken_phone != clean_phone:
                            has_mispronunciation = True
                            stats[split]["substitution"] += 1
            
            # Assign labels  
            label = 1 if accuracy >= 0.5 else 0
            if label == 0:
                stats[split]["mispronounced_phonemes"] += 1
            else:
                stats[split]["correct_phonemes"] += 1

            phonemes.append(clean_phone)
            spoken_phonemes.append(spoken_phone)
            labels.append(label)
           
            # Update statistics  
            stats[split]["total_phonemes"] += 1
    # **Ensure `<del>` is fully removed before saving**
    spoken_phonemes = [ph for ph in spoken_phonemes if ph != "<del>"]
    
    # **Update Sentence-Level Statistics**
    stats[split]["total_sentences"] += 1
    if removed:
        stats[split]["removed_sentences"] += 1
        # Remove this sentence
        return {
        "phoneme_speechocean": "",  # Empty string instead of missing key
        "actual_spoken_phonemes": "",
        "labels": [],
        "text": "",
        "audio": {"array": [], "sampling_rate": 16000},  # Empty array, default sampling rate
        "age": -1  # Use -1 as a placeholder for filtered samples
        }
    
    if has_mispronunciation:
        stats[split]["mispronounced_sentences"] += 1
    else:
        stats[split]["correct_sentences"] += 1

    # Prepare structured output
    return {
        "phoneme_speechocean": " ".join(phonemes),  # Canonical phoneme sequence
        "actual_spoken_phonemes": " ".join(spoken_phonemes),  # Actual spoken phoneme sequence
        "labels": torch.tensor(labels, dtype=torch.long).tolist(),  # Convert labels to tensor
        "text": sample["text"],
        "audio": sample["audio"],
        "age": sample["age"]
    }

# Load the dataset
ds = load_dataset("mispeech/speechocean762")

# Filter only speakers aged 11 and below
ds_filtered = ds.filter(lambda x: x["age"] <= 11)

# Apply preprocessing separately to train and test splits
ds_preprocessed = DatasetDict({
    "train": ds_filtered["train"].map(lambda x: preprocess_sample(x, "train")),
    "test": ds_filtered["test"].map(lambda x: preprocess_sample(x, "test"))
})

# Remove uncertain entries (deleted samples)
ds_preprocessed["train"] = ds_preprocessed["train"].filter(lambda x: x["age"] != -1)
ds_preprocessed["test"] = ds_preprocessed["test"].filter(lambda x: x["age"] != -1)

# Save the dataset
ds_preprocessed.save_to_disk("/srv/scratch/z5369417/outputs/phonemization_speechocean/")

# Save Statistics as Excel
# Compute Total Statistics
stats["total"] = {key: stats["train"][key] + stats["test"][key] for key in stats["train"]}

df_stats = pd.DataFrame(stats).T
df_stats.to_excel("/srv/scratch/z5369417/outputs/phonemization_speechocean/dataset_statistics.xlsx")