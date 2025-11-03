"""
============================================================
MDD Analytical Summary (SpeechOcean)
Author: Antonia Jian
Date: Oct 2025
============================================================

Purpose:
- Perform complete analysis of phoneme-level MDD results
  using aligned phoneme outputs and true attribute predictions.
- Compute TA/FR/FA/TR/CD/DE statistics:
    (1) Per phoneme
    (2) Per attribute
    (3) Per age group and gender
    (4) Attribute difference patterns for high-error phonemes

Inputs:
    - MDD sample details from aligned evaluation
    - Attribute predictions from results_speechocean_test.db
    - Phoneme→attribute mapping
    - SpeechOcean metadata (age, gender)

Outputs:
    - mdd_phoneme_summary.csv
    - mdd_attribute_summary.csv
    - mdd_age_summary.csv
    - mdd_gender_summary.csv
    - mdd_attribute_differences.csv
    - phoneme_error_rates.png
    - attribute_error_rates.png
    - demographic_age_trends.png
    - demographic_gender_trends.png
============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from collections import Counter
from datasets import load_from_disk
import Levenshtein
import jiwer.transforms as tr
import re
from datasets import load_dataset

# ==================== CONFIG ====================

RESULTS_DB = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_4/results_speechocean_test.db"
MDD_DETAIL = "/srv/scratch/z5369417/outputs/mdd_speechocean_phoneme_level_exp11/mdd_sample_detail.csv"
PHONEME2ATT = "data/p2att_en_us-arpa_noDiph.csv"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_speechocean_analysis_exp11"

os.makedirs(OUT_DIR, exist_ok=True)

# ==================== LOAD DATA ====================

df_mdd = pd.read_csv(MDD_DETAIL)
dataset = load_from_disk(RESULTS_DB)
df_pred = dataset["test"].to_pandas() if "test" in dataset else dataset.to_pandas()
df_p2a = pd.read_csv(PHONEME2ATT)


# ==================== STAGE 1: PHONEME-LEVEL SUMMARY ====================
from collections import Counter
import Levenshtein
import jiwer.transforms as tr

def align_lists(ref, hyp):
    vocab = list(set(ref + hyp))
    char_map = {p: chr(i + 33) for i, p in enumerate(vocab)}
    ref_str = "".join(char_map[p] for p in ref)
    hyp_str = "".join(char_map[p] for p in hyp)
    ops = Levenshtein.editops(ref_str, hyp_str)

    aligned = []
    r_i = h_i = 0
    for op, i1, i2 in ops:
        while r_i < i1 and h_i < i2:
            aligned.append((ref[r_i], hyp[h_i]))
            r_i += 1; h_i += 1
        if op == "insert":
            aligned.append(("", hyp[h_i])); h_i += 1
        elif op == "delete":
            aligned.append((ref[r_i], "")); r_i += 1
        else:
            aligned.append((ref[r_i], hyp[h_i]))
            r_i += 1; h_i += 1

    while r_i < len(ref) and h_i < len(hyp):
        aligned.append((ref[r_i], hyp[h_i])); r_i += 1; h_i += 1
    while r_i < len(ref):
        aligned.append((ref[r_i], "")); r_i += 1
    while h_i < len(hyp):
        aligned.append(("", hyp[h_i])); h_i += 1

    return aligned

all_records = []

for _, sample in df_mdd.iterrows():
    can = str(sample.get("canonical", "") or "").split()
    spo = str(sample.get("spoken", "") or "").split()
    pred = str(sample.get("predicted", "") or "").split()

    align_truth = align_lists(can, spo)
    align_pred  = align_lists(can, pred)

    for (can_ph, spo_ph), (_, pred_ph) in zip(align_truth, align_pred):
        record = {"canonical": can_ph, "predicted": pred_ph, "spoken": spo_ph,
                  "TA": 0, "FR": 0, "FA": 0, "TR": 0, "CD": 0, "DE": 0}

        if can_ph == "" and spo_ph != "":
            # INSERTION
            record["TR"] = 1
            record["CD" if pred_ph == spo_ph else "DE"] = 1

        elif can_ph != "" and spo_ph == "":
            # DELETION
            record["TR"] = 1
            record["CD" if pred_ph == "" else "DE"] = 1

        elif can_ph == spo_ph:
            # CORRECT PRONUNCIATION
            if pred_ph == can_ph:
                record["TA"] = 1
            else:
                record["FR"] = 1

        else:
            # MISPRONUNCIATION (SUBSTITUTION)
            if pred_ph == can_ph:
                record["FA"] = 1
            else:
                record["TR"] = 1
                record["CD" if pred_ph == spo_ph else "DE"] = 1

        record["text"] = sample.get("text", "")
        all_records.append(record)

df_phoneme_detail = pd.DataFrame(all_records)
df_phoneme_detail.to_csv(os.path.join(OUT_DIR, "mdd_phoneme_expanded.csv"), index=False)

def compute_phoneme_summary(df):
    group = df.groupby("canonical").agg({
        "TA": "sum", "FR": "sum", "FA": "sum",
        "TR": "sum", "CD": "sum", "DE": "sum"
    })
    group["Total"] = group.sum(axis=1)
    group["FAR"] = group["FA"] / (group["FA"] + group["TR"] + 1e-8)
    group["FRR"] = group["FR"] / (group["FR"] + group["TA"] + 1e-8)
    group["DER"] = group["DE"] / (group["CD"] + group["DE"] + 1e-8)
    return group.reset_index()

df_phoneme = compute_phoneme_summary(df_phoneme_detail)
df_phoneme.loc[df_phoneme["canonical"] == "", "canonical"] = "[INSERTION]"
df_phoneme.to_csv(os.path.join(OUT_DIR, "mdd_phoneme_summary.csv"), index=False)

plt.figure(figsize=(14,6))
x = np.arange(len(df_phoneme["canonical"]))
width = 0.25
plt.bar(x - width, df_phoneme["FAR"], width, label="FAR")
plt.bar(x,         df_phoneme["FRR"], width, label="FRR")
plt.bar(x + width, df_phoneme["DER"], width, label="DER")
plt.xticks(x, df_phoneme["canonical"], rotation=90)
plt.ylabel("Error Rate")
plt.title("Phoneme-Level Error Rates (SpeechOcean)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "phoneme_error_rates.png"))
plt.close()

# ==================== STAGE 2: ATTRIBUTE-LEVEL SUMMARY ====================

df_p2a.set_index(df_p2a.columns[0], inplace=True)
attr_summary = []

for att in df_p2a.columns:
    pos_phonemes = df_p2a.index[df_p2a[att] == 1].tolist()
    subset = df_phoneme[df_phoneme["canonical"].isin(pos_phonemes)]
    if len(subset) == 0:
        continue
    attr_summary.append({
        "Attribute": att,
        "n_phonemes": len(pos_phonemes),
        "Mean_FAR": subset["FAR"].mean(),
        "Mean_FRR": subset["FRR"].mean(),
        "Mean_DER": subset["DER"].mean()
    })

df_attr = pd.DataFrame(attr_summary).sort_values("Mean_DER", ascending=False)
df_attr.to_csv(os.path.join(OUT_DIR, "mdd_attribute_summary.csv"), index=False)

plt.figure(figsize=(14,6))
x = np.arange(len(df_attr["Attribute"]))
width = 0.25
plt.bar(x - width, df_attr["Mean_FAR"], width, label="FAR")
plt.bar(x,         df_attr["Mean_FRR"], width, label="FRR")
plt.bar(x + width, df_attr["Mean_DER"], width, label="DER")
plt.xticks(x, df_attr["Attribute"], rotation=90)
plt.ylabel("Error Rate")
plt.title("Attribute-Level Error Rates (SpeechOcean)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attribute_error_rates.png"))
plt.close()

# ==================== STAGE 3: DEMOGRAPHIC ANALYSIS ====================
# Get age/gender from df_pred and merge via 'text'
df_meta = df_pred[["text", "age", "gender"]].drop_duplicates()

# Join metadata with detailed phoneme results
df_phoneme_detail = df_phoneme_detail.merge(df_meta, on="text", how="left")

# Safeguard missing demographics
df_phoneme_detail["age"] = df_phoneme_detail["age"].fillna(-1)
df_phoneme_detail["gender"] = df_phoneme_detail["gender"].fillna("unknown")
df_phoneme_detail["age_group"] = df_phoneme_detail["age"].astype(str)

# --- Aggregate metrics by age group and gender ---
def aggregate_demographic(df, group_vars):
    metrics = ["TA", "FR", "FA", "TR", "CD", "DE"]
    out = df.groupby(group_vars)[metrics].sum().reset_index()
    out["FAR"] = out["FA"] / (out["FA"] + out["TR"] + 1e-8)
    out["FRR"] = out["FR"] / (out["FR"] + out["TA"] + 1e-8)
    out["DER"] = out["DE"] / (out["CD"] + out["DE"] + 1e-8)
    return out

df_demo_age = aggregate_demographic(df_phoneme_detail, ["age_group"])
df_demo_age["age_group"] = (
    pd.to_numeric(df_demo_age["age_group"], errors="coerce")
    .fillna(-1)
    .astype(int)
)

df_demo_age = df_demo_age.sort_values("age_group")
df_demo_gender = aggregate_demographic(df_phoneme_detail, ["gender"])

# Save summaries
df_demo_age.to_csv(os.path.join(OUT_DIR, "mdd_age_summary.csv"), index=False)
df_demo_gender.to_csv(os.path.join(OUT_DIR, "mdd_gender_summary.csv"), index=False)

# --- Plot error-rate trends by age ---
plt.figure(figsize=(8,5))
plt.plot(df_demo_age["age_group"], df_demo_age["FAR"], marker='o', label="FAR")
plt.plot(df_demo_age["age_group"], df_demo_age["FRR"], marker='o', label="FRR")
plt.plot(df_demo_age["age_group"], df_demo_age["DER"], marker='o', label="DER")
plt.legend()
plt.title("Error Rates by Age Group (SpeechOcean)")
plt.ylabel("Rate")
plt.xlabel("Age Group (years)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "demographic_age_trends.png"))
plt.close()

# --- Plot error-rate comparison by gender ---
plt.figure(figsize=(6,5))
x = np.arange(len(df_demo_gender["gender"]))
plt.bar(x-0.25, df_demo_gender["FAR"], 0.25, label="FAR")
plt.bar(x,      df_demo_gender["FRR"], 0.25, label="FRR")
plt.bar(x+0.25, df_demo_gender["DER"], 0.25, label="DER")
plt.xticks(x, df_demo_gender["gender"])
plt.ylabel("Rate")
plt.title("Error Rates by Gender (SpeechOcean)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "demographic_gender_trends.png"))
plt.close()


# ==================== STAGE 4: ATTRIBUTE DIFFERENCE ANALYSIS ====================

TOP_N = 10
df_far_sorted = df_phoneme.sort_values("FAR", ascending=False).head(TOP_N)
df_frr_sorted = df_phoneme.sort_values("FRR", ascending=False).head(TOP_N)
df_der_sorted = df_phoneme.sort_values("DER", ascending=False).head(TOP_N)
high_err_phonemes = pd.concat([df_far_sorted["canonical"],
                               df_frr_sorted["canonical"],
                               df_der_sorted["canonical"]]).dropna().unique()

def _to_list_of_str(x):
    if isinstance(x, list):
        return [str(y) for y in x]
    if hasattr(x, "tolist"):
        return [str(y) for y in x.tolist()]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(y) for y in v]
            except: pass
        return [s]
    return []

df_attr_pred = df_pred.copy()
df_attr_pred["target_list"] = df_attr_pred["target_text"].apply(_to_list_of_str)
df_attr_pred["pred_list"]   = df_attr_pred["pred_str"].apply(_to_list_of_str)

records = []
for ph in high_err_phonemes:
    diff_counter = Counter(); total_occurrences = 0
    samples = df_mdd[df_mdd["canonical"].astype(str).str.contains(rf"\b{ph}\b", na=False)]
    for idx, row in samples.iterrows():
        can_list = str(row["canonical"]).split()
        pos_list = [i for i, p in enumerate(can_list) if p == ph]
        if not pos_list: continue
        if idx not in df_attr_pred.index: continue
        tgt_attr_rows = df_attr_pred.at[idx, "target_list"]
        prd_attr_rows = df_attr_pred.at[idx, "pred_list"]
        if not isinstance(tgt_attr_rows, list) or not isinstance(prd_attr_rows, list): continue
        K = min(len(tgt_attr_rows), len(prd_attr_rows))
        for pos in pos_list:
            for k in range(K):
                tgt_states = str(tgt_attr_rows[k]).split()
                prd_states = str(prd_attr_rows[k]).split()
                if pos >= len(tgt_states) or pos >= len(prd_states): continue
                t_state, p_state = tgt_states[pos], prd_states[pos]
                if (t_state.startswith(("n_","p_")) and p_state.startswith(("n_","p_"))):
                    attr_t, attr_p = t_state[2:], p_state[2:]
                    if attr_t == attr_p and (t_state[0] != p_state[0]):
                        diff_counter[attr_t] += 1
            total_occurrences += len(pos_list)
    if total_occurrences > 0:
        top_diffs = diff_counter.most_common(5)
        records.append({
            "Phoneme": ph,
            "Samples": total_occurrences,
            "Top_Attribute_Differences": ", ".join([f"{k}({v})" for k,v in top_diffs]),
            "FAR": float(df_phoneme.loc[df_phoneme["canonical"]==ph,"FAR"].values[0]),
            "FRR": float(df_phoneme.loc[df_phoneme["canonical"]==ph,"FRR"].values[0]),
            "DER": float(df_phoneme.loc[df_phoneme["canonical"]==ph,"DER"].values[0])
        })

df_diff = pd.DataFrame(records)
df_diff.to_csv(os.path.join(OUT_DIR, "mdd_attribute_differences.csv"), index=False)

for metric, df_sort, color in [("FAR", df_far_sorted, "tab:red"),
                               ("FRR", df_frr_sorted, "tab:orange"),
                               ("DER", df_der_sorted, "tab:purple")]:
    plt.figure(figsize=(10,6))
    plt.bar(df_sort["canonical"], df_sort[metric], color=color)
    plt.title(f"Top {TOP_N} Phonemes with Highest {metric} (SpeechOcean)")
    plt.ylabel(metric)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"top_phonemes_{metric}.png"))
    plt.close()

# --- Plot per-attribute flip counts in high-error phonemes ---
from matplotlib.cm import get_cmap

# Accumulate total attribute flips across all high-error phonemes
attr_flip_total = Counter()
for diffs in df_diff["Top_Attribute_Differences"]:
    for entry in str(diffs).split(", "):
        if "(" in entry and entry.endswith(")"):
            attr = entry[:entry.find("(")]
            count = int(entry[entry.find("(")+1:-1])
            attr_flip_total[attr] += count

# Convert to DataFrame for plotting
df_attr_flips = pd.DataFrame(
    sorted(attr_flip_total.items(), key=lambda x: x[1], reverse=True),
    columns=["Attribute", "Flip_Count"]
)

# Plot bar chart of attribute flip frequencies
plt.figure(figsize=(12,6))
cmap = get_cmap("Set2")
plt.bar(df_attr_flips["Attribute"], df_attr_flips["Flip_Count"], color=cmap.colors[:len(df_attr_flips)])
plt.xticks(rotation=90)
plt.ylabel("Flip Count")
plt.title("Most Frequently Flipped Attributes in High-Error Canonical Phonemes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attribute_flip_counts_high_error_phonemes (SpeechOcean).png"))
plt.close()

print("Saved: attribute_flip_counts_high_error_phonemes (SpeechOcean).png")

# --- Heatmap-style stacked bar plot of attribute flips by phoneme ---
from collections import defaultdict

# Collect flip matrix: {phoneme: {attribute: count}}
flip_matrix = defaultdict(lambda: defaultdict(int))
for _, row in df_diff.iterrows():
    ph = row["Phoneme"]
    diffs = str(row["Top_Attribute_Differences"]).split(", ")
    for entry in diffs:
        if "(" in entry and entry.endswith(")"):
            attr = entry[:entry.find("(")]
            count = int(entry[entry.find("(")+1:-1])
            flip_matrix[ph][attr] += count

# Prepare matrix DataFrame
all_attrs = sorted(set(attr for ph_diffs in flip_matrix.values() for attr in ph_diffs))
rows = []
for ph in df_diff["Phoneme"]:
    row = [flip_matrix[ph].get(attr, 0) for attr in all_attrs]
    rows.append(row)

df_matrix = pd.DataFrame(rows, columns=all_attrs, index=df_diff["Phoneme"])

# Plot stacked bar chart
plt.figure(figsize=(14,7))
bottom = np.zeros(len(df_matrix))
for i, attr in enumerate(df_matrix.columns):
    plt.bar(df_matrix.index, df_matrix[attr], bottom=bottom, label=attr)
    bottom += df_matrix[attr].values

plt.xticks(rotation=90)
plt.ylabel("Flip Count")
plt.title("Attribute Flip Breakdown per High-Error Canonical Phoneme")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attribute_flip_stackedbar_by_phoneme (SpeechOcean).png"))
plt.close()

print("Saved: attribute_flip_stackedbar_by_phoneme (SpeechOcean).png")
