"""
Analyse training (attribute-level) results by demographic groups
---------------------------------------------------------------
This replicates the same AER logic as train.py but computes it
per demographic (gender, speech_status, age_year).
"""

import os
import pandas as pd
from datasets import load_from_disk
import evaluate

# === USER PATHS ===
results_path = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_5/results_unsw_test.db"
output_dir = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_5/analysis_AER_demographics_unsw"
os.makedirs(output_dir, exist_ok=True)

# === 1. Load dataset ===
dataset = load_from_disk(results_path)
df = pd.DataFrame(dataset)
print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

# === 2. Verify demographic fields ===
expected_cols = ["age", "gender", "speech_status"]
for col in expected_cols:
    if col not in df.columns:
        raise ValueError(f" Missing column '{col}' in dataset. Ensure evaluate_SA_model included demographics.")

# === 3. Convert age (months) → integer years ===
df["age_year"] = (df["age"].astype(float) / 12).round(0).astype(int)

# === 4. Load same AER metric used in train.py ===
metric = evaluate.load("wer")

# === 5. Compute per-sample AER ===
print("Computing AER per sample ...")
records = []
for _, row in df.iterrows():
    rec = {
        "speaker_id": row.get("speaker_id", None),
        "gender": row["gender"],
        "speech_status": row["speech_status"],
        "age_year": row["age_year"]
    }

    preds = row["pred_str"]
    refs = row["target_text"]

    # Ensure both are lists of equal length (each attr group)
    if isinstance(preds, list) and isinstance(refs, list):
        for i in range(len(preds)):
            rec[f"AER_group{i}"] = metric.compute(predictions=[preds[i]], references=[refs[i]])
    records.append(rec)

sample_df = pd.DataFrame(records)
sample_out = os.path.join(output_dir, "sample_AER_detail.csv")
sample_df.to_csv(sample_out, index=False)
print(f"Saved sample-level AERs → {sample_out}")

# === 6. Group by demographics ===
print("Grouping by gender  speech_status  age_year ...")
value_cols = [c for c in sample_df.columns if c.startswith("AER_group")]
group_summary = (
    sample_df.groupby(["gender", "speech_status", "age_year"])[value_cols]
    .mean()
    .reset_index()
)
group_out = os.path.join(output_dir, "grouped_AER_summary.csv")
group_summary.to_csv(group_out, index=False)
print(f"✔ Saved grouped summary → {group_out}")

print("\n AER demographic analysis complete.")
print(f"→ Sample-level file: {sample_out}")
print(f"→ Grouped summary:   {group_out}")