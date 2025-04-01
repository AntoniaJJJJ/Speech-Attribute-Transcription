import torch
import re
import pandas as pd
from datasets import DatasetDict, load_dataset


def find_insertions(data):
    """
    Finds entries where the pronounced phoneme sequence is longer than the canonical phoneme sequence,
    which could indicate insertion errors.

    Args:
        data: A list of dataset entries. Each entry should be a dictionary with keys such as
              "phones" for the canonical phoneme sequence and "mispronunciations" for the
              pronounced phoneme annotations.

    Returns:
        A list of entries where insertions might exist, along with the identified extra phonemes.
    """
    insertion_entries = []

    for sample in data:
        for word in sample.get("words", []):
            canonical_phones = word.get("phones", [])
            mispronunciations = word.get("mispronunciations", [])

            # Build the pronounced phoneme sequence
            pronounced_phones = list(canonical_phones)  # Start with canonical
            for misp in mispronunciations:
                idx = misp["index"]
                if idx < len(pronounced_phones):
                    # Replace the phoneme at the indicated index
                    pronounced_phones[idx] = misp["pronounced-phone"].lower()
                else:
                    # If index is out of bounds, append (potential insertion)
                    pronounced_phones.append(misp["pronounced-phone"].lower())

            # Check if the pronounced sequence is longer than the canonical sequence
            if len(pronounced_phones) > len(canonical_phones):
                # Identify the extra phonemes
                extra_phones = pronounced_phones[len(canonical_phones):]
                insertion_entries.append({
                    "sample": sample,
                    "canonical": canonical_phones,
                    "pronounced": pronounced_phones,
                    "inserted_phones": extra_phones
                })

    return insertion_entries


# Load the dataset
ds = load_dataset("mispeech/speechocean762")

# Filter only speakers aged 11 and below
ds_filtered = ds.filter(lambda x: x["age"] <= 11)

# Find insertion cases
insertions = find_insertions(ds_filtered)

# Print results
for entry in insertions:
    print("Insertion found:")
    print(f"  Canonical Phones: {entry['canonical']}")
    print(f"  Pronounced Phones: {entry['pronounced']}")
    print(f"  Inserted Phones: {entry['inserted_phones']}")