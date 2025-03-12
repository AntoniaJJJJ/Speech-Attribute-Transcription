import argparse
from datasets import load_from_disk
import pandas as pd
from collections import defaultdict
import ace_tools as tools

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Mispronunciation Detection Script")
parser.add_argument("--data_path", type=str, required=True, help="Path to the results dataset (e.g., results_speechocean_test.db)")
args = parser.parse_args()

# Load the evaluation results dataset
ds_results = load_from_disk(args.data_path)

# Store mispronunciations
mispronunciations = []

# Iterate through test samples
for sample in ds_results["test"]:
    text = sample["text"]
    phonemes = sample["phoneme_speechocean"].split()
    pred_attributes = sample["pred_str"]
    target_attributes = sample["target_text"]

    # Track mispronounced phonemes
    for i, (phoneme, pred, target) in enumerate(zip(phonemes, pred_attributes, target_attributes)):
        if pred != target:  # Mismatch detected
            mispronunciations.append({
                "text": text,
                "phoneme": phoneme,
                "predicted": pred,
                "expected": target
            })

# Convert to a DataFrame for analysis
df_mispronunciations = pd.DataFrame(mispronunciations)

# Count mispronunciation occurrences per phoneme
phoneme_errors = df_mispronunciations["phoneme"].value_counts().reset_index()
phoneme_errors.columns = ["Phoneme", "Mispronunciation Count"]

# Display results
tools.display_dataframe_to_user(name="Mispronunciations", dataframe=df_mispronunciations)
tools.display_dataframe_to_user(name="Phoneme Mispronunciation Counts", dataframe=phoneme_errors)