import torch
import re
import pandas as pd
from datasets import DatasetDict, load_dataset


def find_insertions(data):
    """
    Identifies potential insertion annotations within the dataset, including cases
    where canonical-phone is empty.
    
    Args:
        data (list): A list of dataset entries. Each entry is expected to be a dictionary
                     with a 'words' key, which is a list of word-level annotations.
    
    Returns:
        List of dictionaries containing the sample and details of the identified insertion.
    """
    insertion_candidates = []

    for sample in data:
        for word in sample.get("words", []):
            mispronunciations = word.get("mispronunciations", [])
            
            # Check each mispronunciation entry
            for misp in mispronunciations:
                # Consider it an insertion if canonical-phone is missing, empty, or just a space
                canonical_phone = misp.get("canonical-phone", "").strip()
                if not canonical_phone:
                    insertion_candidates.append({
                        "sample_text": sample.get("text", ""),
                        "pronounced_phone": misp.get("pronounced-phone", ""),
                        "word_level_data": word
                    })

    return insertion_candidates


# Load the dataset
ds = load_dataset("mispeech/speechocean762")

# Filter only speakers aged 11 and below
ds_filtered = ds.filter(lambda x: x["age"] <= 11)

# Find insertion cases
insertions = find_insertions(ds_filtered)

# Print results
for ins in insertions:
    print("Insertion Candidate Found:")
    print(f"  Sample Text: {ins['sample_text']}")
    print(f"  Pronounced Phone (Potential Insertion): {ins['pronounced_phone']}")