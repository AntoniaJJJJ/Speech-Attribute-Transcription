"""
Author: Antonia Jian
Date(Last modified): 10/03/2025
Description: 
This script format SpeechOcean762 to match the CU dataset
- convert phonemes into sequences
- assign label: 0 (correct pronunciation) or 1 (mispronounced, including heavy accent)

SpeechOcean762 dataset contains two subsets:
-Train set (ds["train"]): 2500 samples
- Test set (ds["test"]): 2500 samples

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
from datasets import DatasetDict, load_dataset

# Function to preprocess each sample
def preprocess_sample(sample):
    # Extract phonemes and their correctness labels
    phonemes = []
    labels = []
    
    for word in sample["words"]:
        for i, phone in enumerate(word["phones"]):
            phonemes.append(phone.lower())
            # Convert accuracy score to mispronunciation label (1 = error, 0 = correct)
            labels.append(0 if word["phones-accuracy"][i] >= 2.0 else 1)

    # Prepare structured output
    return {
        "phoneme_speechocean": " ".join(phonemes),  # Convert phoneme list to a string
        "labels": torch.tensor(labels, dtype=torch.long),  # Convert labels to tensor
        "text": sample["text"],
        "audio": sample["audio"]["array"],  # Keep raw audio
        "sampling_rate": sample["audio"]["sampling_rate"],
    }

# Load the dataset
ds = load_dataset("mispeech/speechocean762")

# Apply preprocessing to both train and test sets
ds_preprocessed = ds.map(preprocess_sample)

# Save the dataset
ds_preprocessed.save_to_disk("/srv/scratch/z5369417/outputs/phonemization_speechocean/")