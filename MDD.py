import argparse
from datasets import load_from_disk
import pandas as pd
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Mispronunciation Detection Script")
parser.add_argument("--data_path", type=str, required=True, help="Path to the results dataset (e.g., results_speechocean_test.db)")
args = parser.parse_args()

# Load the evaluation results dataset
ds_results = load_from_disk(args.data_path)

# Determine the output directory (same as data_path directory)
output_dir = os.path.dirname(args.data_path)

# Store mispronunciations & metrics
mispronunciations = []
metrics = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

# Iterate through test samples
for sample in ds_results["test"]:
    text = sample["text"]
    phonemes = sample["phoneme_speechocean"].split()
    pred_attributes = sample["pred_str"]
    target_attributes = sample["target_text"]
    mispronunciation_labels = sample["labels"]  # 0 = correct, 1 = mispronounced

    # Track mispronounced phonemes and calculate TP, FP, TN, FN
    for i, (phoneme, pred, target, label) in enumerate(zip(phonemes, pred_attributes, target_attributes, mispronunciation_labels)):
        model_detected_mispronunciation = (pred != target)  # If predicted attributes mismatch expected attributes
        ground_truth_mispronunciation = (label == 1)  # 1 = phoneme is actually mispronounced

        if model_detected_mispronunciation and ground_truth_mispronunciation:
            metrics["TP"] += 1  # True Positive (correctly detected mispronunciation)
        elif model_detected_mispronunciation and not ground_truth_mispronunciation:
            metrics["FP"] += 1  # False Positive (incorrectly flagged a correct phoneme)
        elif not model_detected_mispronunciation and not ground_truth_mispronunciation:
            metrics["TN"] += 1  # True Negative (correct phoneme correctly classified)
        elif not model_detected_mispronunciation and ground_truth_mispronunciation:
            metrics["FN"] += 1  # False Negative (missed a mispronunciation)

        # Store mispronunciations for analysis
        if model_detected_mispronunciation:
            mispronunciations.append({
                "text": text,
                "phoneme": phoneme,
                "predicted": pred,
                "expected": target,
                "ground_truth_label": "Mispronounced" if label == 1 else "Correct"
            })

# Convert to a DataFrame for analysis
df_mispronunciations = pd.DataFrame(mispronunciations)

# Count mispronunciation occurrences per phoneme
phoneme_errors = df_mispronunciations["phoneme"].value_counts().reset_index()
phoneme_errors.columns = ["Phoneme", "Mispronunciation Count"]

# Define output file paths
mispronunciations_file = os.path.join(output_dir, "mispronunciations.csv")
phoneme_errors_file = os.path.join(output_dir, "phoneme_mispronunciation_counts.csv")
metrics_file = os.path.join(output_dir, "mdd_metrics.csv")

# Save results to CSV
df_mispronunciations.to_csv(mispronunciations_file, index=False)
phoneme_errors.to_csv(phoneme_errors_file, index=False)

# Save TP, FP, TN, FN metrics
df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv(metrics_file, index=False)