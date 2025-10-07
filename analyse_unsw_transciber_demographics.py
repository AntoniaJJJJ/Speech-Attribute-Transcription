"""
Author: Antonia Jian
Date (Last Modified): 07/10/2025

Description:
Analyses the demographic performance of the CU model (Exp11) tested on the UNSW dataset.

This script loads the transcriber output:
    /srv/scratch/z5369417/outputs/transcriber_result/cu_model_exp11_test_on_unsw/

and groups the evaluation results by:
    - age (converted from months to years, binned into age groups)
    - gender
    - speech_status (0 = Non-SSD, 1 = SSD)

The script then summarizes performance metrics (WER) for each demographic group and saves the results as CSV files.

It does NOT re-compute WER or modify prediction logic — it simply aggregates results.

Output directory:
    /srv/scratch/z5369417/outputs/transcriber_result/analysis_cu_model_exp11_test_on_unsw
"""

import os
import pandas as pd
from datasets import load_from_disk

# === PATH SETUP ===
input_dataset_path = "/srv/scratch/z5369417/outputs/transcriber_result/cu_model_exp11_test_on_unsw"
output_dir = "/srv/scratch/z5369417/outputs/transcriber_result/analysis_cu_model_exp11_test_on_unsw"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATASET ===
print("Loading dataset from:", input_dataset_path)
dataset = load_from_disk(input_dataset_path)
if "test" in dataset:
    dataset = dataset["test"]

# Convert to DataFrame
df = pd.DataFrame(dataset)

# === BASIC SANITY CHECKS ===
print(f"Loaded {len(df)} samples.")
print("Columns:", list(df.columns))

# === AGE CONVERSION (months → years) ===
if "age" in df.columns:
    df["age_years"] = df["age"] / 12.0
    df["age_group"] = pd.cut(
        df["age_years"],
        bins=[0, 6, 8, 10, 12, 14, 20],
        labels=["≤6", "7–8", "9–10", "11–12", "13–14", "≥15"]
    )
else:
    df["age_years"] = None
    df["age_group"] = "Unknown"

# === MAP SPEECH STATUS ===
if "speech_status" in df.columns:
    df["speech_status"] = df["speech_status"].map({0: "Non-SSD", 1: "SSD"}).fillna("Unknown")

# === GROUP METRIC AGGREGATION ===
# We'll summarize string-level metrics if available (WER, CER, etc.), otherwise count samples.
metric_columns = [c for c in df.columns if c.lower().startswith(("wer", "cer", "aer"))]

if not metric_columns:
    print("⚠️  No explicit metric columns found (WER/CER/AER). Assuming sample-level metrics are not stored.")
    df["count"] = 1
    summary = (
        df.groupby(["gender", "age_group", "speech_status"])
        .agg(count=("count", "sum"))
        .reset_index()
    )
else:
    # Compute mean metric per demographic cell
    summaries = []
    for metric in metric_columns:
        temp = (
            df.groupby(["gender", "age_group", "speech_status"])
            .agg(
                mean_metric=(metric, "mean"),
                std_metric=(metric, "std"),
                sample_count=(metric, "count")
            )
            .reset_index()
        )
        temp["metric_name"] = metric
        summaries.append(temp)
    summary = pd.concat(summaries, ignore_index=True)

# === SAVE OUTPUTS ===
summary_csv = os.path.join(output_dir, "demographic_summary.csv")
df_csv = os.path.join(output_dir, "full_data_flat.csv")

df.to_csv(df_csv, index=False)
summary.to_csv(summary_csv, index=False)

print(f"\n✅ Analysis complete.")
print(f"  → Full flattened dataset saved to: {df_csv}")
print(f"  → Demographic summary saved to: {summary_csv}")

# === OPTIONAL VISUALIZATION ===
try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    if metric_columns:
        for metric in metric_columns:
            subset = summary[summary["metric_name"] == metric]
            plt.figure(figsize=(10, 5))
            sns.barplot(
                data=subset,
                x="age_group",
                y="mean_metric",
                hue="speech_status",
                errorbar=None
            )
            plt.title(f"{metric.upper()} by Age Group and Speech Status")
            plt.ylabel(f"Mean {metric.upper()}")
            plt.xlabel("Age Group (years)")
            plt.legend(title="Speech Status")
            plt.tight_layout()

            fig_path = os.path.join(output_dir, f"{metric}_by_age_gender_ssd.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"  → Figure saved: {fig_path}")

except Exception as e:
    print("⚠️ Visualization skipped due to missing seaborn/matplotlib:", e)