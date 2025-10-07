"""
Analyse UNSW transcriber results by demographic groups
------------------------------------------------------
This script:
1. Loads transcriber dataset (phoneme_unsw + pred_phoneme + demographics)
2. Computes per-sample WER using the same metric as transcriber.py
3. Groups WER by gender, speech_status (SSD vs non-SSD), and age group (in months)
4. Reads the saved confusion-matrix CSV and aggregates confusion counts
5. Saves:
   - sample-level WERs → sample_WER_detail.csv
   - grouped WER stats → grouped_WER_summary.csv
   - demographic-weighted confusion matrices → confusion_summary_by_group.csv
"""

import os
import pandas as pd
import evaluate
import re

# === USER PATHS ===
dataset_path = "/srv/scratch/z5369417/outputs/transcriber_result/cu_model_exp11_test_on_unsw"
confusion_matrix_file = "/srv/scratch/z5369417/outputs/transcriber_result/cu_model_exp11_test_on_unsw_no_diph_cm.csv"    # produced by transcriber.py
output_dir = "/srv/scratch/z5369417/outputs/transcriber_result/analysis_cu_model_exp11_test_on_unsw"
os.makedirs(output_dir, exist_ok=True)

# === 1. Load dataset ===
print("Loading transcriber dataset ...")
data = []
from datasets import load_from_disk
dataset = load_from_disk(dataset_path)
for example in dataset:
    data.append(example)
df = pd.DataFrame(data)

# === 2. Prepare text columns ===
df = df.dropna(subset=["phoneme_unsw", "pred_phoneme"])
df["phoneme_unsw"] = df["phoneme_unsw"].astype(str).str.strip()
df["pred_phoneme"] = df["pred_phoneme"].astype(str).str.strip()

# === 3. Compute per-sample WER ===
print("Computing WER per sample ...")
wer_metric = evaluate.load("wer")
df["WER"] = df.apply(
    lambda row: wer_metric.compute(predictions=[row["pred_phoneme"]], references=[row["phoneme_unsw"]]),
    axis=1
)


# === 4. Convert age in months → integer age in years ===
df["age_year"] = (df["age"].astype(float) / 12).round(0).astype(int)



# === 5. Compute grouped WER statistics ===
print("Grouping by demographics ...")
group_stats = (
    df.groupby(["gender", "speech_status", "age_group"])
      .agg(mean_WER=("WER","mean"),
           std_WER=("WER","std"),
           count=("WER","count"))
      .reset_index()
)
group_stats.to_csv(os.path.join(output_dir, "grouped_WER_summary.csv"), index=False)

# === 6. Load and analyse confusion matrix ===
print("Analysing confusion matrix ...")
if os.path.exists(confusion_matrix_file):
    cm_df = pd.read_csv(confusion_matrix_file, index_col=0)
    cm_df.index.name = "Reference"
    cm_df.columns.name = "Predicted"

    # total confusions per group of interest
    top_confusions = (
        cm_df.stack().reset_index(name="count")
        .query("Reference != Predicted and count > 0")
        .sort_values("count", ascending=False)
    )

    # compute relative confusion (normalized)
    cm_normalized = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)
    cm_norm_top = (
        cm_normalized.stack().reset_index(name="ratio")
        .query("Reference != Predicted and ratio > 0")
        .sort_values("ratio", ascending=False)
    )

    top_confusions.to_csv(os.path.join(output_dir, "confusion_summary_by_group_raw.csv"), index=False)
    cm_norm_top.to_csv(os.path.join(output_dir, "confusion_summary_by_group_normalized.csv"), index=False)
else:
    print(" Confusion matrix file not found:", confusion_matrix_file)

# === 7. Save sample-level WERs ===
df_out = df[["word","phoneme_unsw","pred_phoneme","age","age_group","gender","speech_status","WER"]]
df_out.to_csv(os.path.join(output_dir, "sample_WER_detail.csv"), index=False)

print("\n Analysis complete.")
print(f"→ Sample-level WERs: {os.path.join(output_dir,'sample_WER_detail.csv')}")
print(f"→ Grouped summary:   {os.path.join(output_dir,'grouped_WER_summary.csv')}")
print(f"→ Confusion summaries (raw & normalized): {output_dir}")