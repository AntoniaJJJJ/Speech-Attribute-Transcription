"""
Author: Antonia Jian
Date(Last modified): 10/03/2025
Description: 
This script format SpeechOcean762 to match the CU dataset
- convert phonemes into sequences
- assign labels: 1 (correct pronunciation) or 0 (mispronounced, including heavy accent)
- Extract actual spoken phonemes from the "mispronunciations" block
- Filters only children aged 11 and below

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
from datasets import DatasetDict, load_dataset

# Function to remove stress markers from vowels (AH0 -> AH, AH1 -> AH)
def remove_stress(phoneme):
    return re.sub(r'([a-z]+)[0-2]$', r'\1', phoneme)  # Removes final digit if it exists

# Function to preprocess each sample
def preprocess_sample(sample):
    # Extract phonemes and their correctness labels
    phonemes = []   # Canonical phonemes
    spoken_phonemes = []  # Actual spoken phonemes
    labels = []     # Correct = 1, Mispronounced = 0
    
    for word in sample["words"]:
        for i, phone in enumerate(word["phones"]):
            clean_phone = remove_stress(phone.lower())  # Convert to lowercase and remove stress
            spoken_phone = clean_phone  # Default: assume spoken phoneme is the same

            # Handle mispronunciations
            if "mispronunciations" in word and word["mispronunciations"]:
                for misp in word["mispronunciations"]:
                    if misp["index"] == i:
                        spoken_phone = misp["pronounced-phone"].lower()
            
            phonemes.append(clean_phone)
            spoken_phonemes.append(spoken_phone)
           
             # Assign labels
            accuracy = word["phones-accuracy"][i]
            label = 1 if accuracy >= 0.5 else 0  # 1 = Correct, 0 = Mispronounced
            labels.append(label)

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

# Apply preprocessing to both train and test sets
ds_preprocessed = ds.map(preprocess_sample)

# Filter only speakers aged 11 and below
ds_preprocessed = ds_preprocessed.filter(lambda x: x["age"] <= 11)

# Save the dataset
ds_preprocessed.save_to_disk("/srv/scratch/z5369417/outputs/phonemization_speechocean/")