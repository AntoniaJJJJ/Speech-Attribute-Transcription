"""
============================================================
MDD Analytical Summary 
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
    - Attribute predictions from results_unsw_test.db
    - Phoneme→attribute mapping
    - UNSW metadata (age, gender, speech_status)

Outputs:
    - mdd_phoneme_summary.csv
    - mdd_attribute_summary.csv
    - mdd_age_summary.csv
    - mdd_gender_summary.csv
    - attribute_error_heatmap.png
    - phoneme_FAR_all.png / FRR / DER
    - demographic_age_trends.png / gender_trends.png
============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from datasets import load_from_disk
from collections import Counter


# ==================== CONFIG ====================

RESULTS_DB = "/srv/scratch/z5369417/outputs/trained_result/CU_AKT_combined/exp22/exp22_2/results_unsw_test.db"
MDD_DETAIL = "/srv/scratch/z5369417/outputs/mdd_unsw_phoneme_level_exp22/mdd_sample_detail.csv"
UNSW_META  = "/srv/scratch/z5369417/UNSW_final_deliverables/CAAP_2023-04-27/dataset_spreadsheet.xlsx"
PHONEME2ATT = "data/p2att_combined_us_au_ipa_Diph_voiced1vowels.csv"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_unsw_analysis_exp22"

os.makedirs(OUT_DIR, exist_ok=True)

# ==================== LOAD DATA ====================

df_mdd = pd.read_csv(MDD_DETAIL)
df_meta = pd.read_excel(UNSW_META)
df_p2a = pd.read_csv(PHONEME2ATT)

# --- Load attribute predictions from DB (train.py output) ---
dataset = load_from_disk(RESULTS_DB)
if "test" in dataset:
    df_pred = dataset["test"].to_pandas()
else:
    df_pred = dataset.to_pandas()

# --- Add age (years) ---
df_meta["age_year"] = (df_meta["age"].astype(float) / 12).round(0).astype(int)

# --- Merge metadata ---
df_meta.rename(columns={
    "word_phonemes": "phoneme_unsw",
    "recording_phonemes": "actual_spoken_phonemes",
    "speech_status": "speech_status",
    "audio_file": "audio_file",
    "word": "word"
}, inplace=True)

df_all = df_mdd.merge(
    df_meta[["word", "age_year", "gender", "speech_status"]],
    on="word", how="left"
)

print(f"Total merged samples: {len(df_all)}")

# ==================== STAGE 1: PHONEME-LEVEL SUMMARY ====================

"""
This section re-aligns canonical ↔ predicted and spoken ↔ predicted phoneme
sequences for each word sample in mdd_sample_detail.csv using the same
Levenshtein-based logic as in Script 4 (align_like_cm).

It then classifies each aligned triplet into TA, FR, FA, TR, CD, or DE
to build a per-phoneme dataset and computes FAR, FRR, and DER summaries.
"""
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


# --- Load datasets ---
df_mdd = pd.read_csv(MDD_DETAIL)
df_meta = pd.read_excel(UNSW_META)
df_meta.rename(columns={
    "word_phonemes": "phoneme_unsw",
    "recording_phonemes": "actual_spoken_phonemes",
    "speech_status": "speech_status",
    "audio_file": "audio_file",
    "word": "word"
}, inplace=True)
df_meta["age_year"] = (df_meta["age"].astype(float) / 12).round(0).astype(int)
df_all = df_mdd.merge(df_meta[["word", "age_year", "gender", "speech_status"]],
                      on="word", how="left")

# --- Expand aligned phonemes per sample ---
all_records = []

for _, sample in df_all.iterrows():
    can = str(sample.get("canonical", "") or "").split()
    spo = str(sample.get("spoken", "") or "").split()
    pred = str(sample.get("predicted", "") or "").split()

    align_truth = align_lists(can, spo)
    align_pred  = align_lists(can, pred)

    for (can_ph, spo_ph), (_, pred_ph) in zip(align_truth, align_pred):
        record = {
            "canonical": can_ph,
            "predicted": pred_ph,
            "spoken": spo_ph,
            "TA": 0, "FR": 0, "FA": 0, "TR": 0, "CD": 0, "DE": 0
        }

        # --- classification (same logic as MDD script) ---
        if can_ph == "" and spo_ph != "":
            # ins
            record["TR"] = 1
            record["CD" if pred_ph == spo_ph else "DE"] = 1

        elif can_ph != "" and spo_ph == "":
            # del
            record["TR"] = 1
            record["CD" if pred_ph == "" else "DE"] = 1

        elif can_ph == spo_ph:
            # match
            if pred_ph == can_ph:
                record["TA"] = 1
            else:
                record["FR"] = 1

        else:
            # sub
            if pred_ph == can_ph:
                record["FA"] = 1
            else:
                record["TR"] = 1
                record["CD" if pred_ph == spo_ph else "DE"] = 1

        # --- attach demographic info ---
        record.update({
            "word": sample["word"],
            "age_year": sample.get("age_year", np.nan),
            "gender": sample.get("gender", np.nan),
            "speech_status": sample.get("speech_status", np.nan)
        })
        all_records.append(record)

# --- Build full expanded dataframe ---
df_phoneme_detail = pd.DataFrame(all_records)
df_phoneme_detail.to_csv(os.path.join(OUT_DIR, "mdd_phoneme_expanded.csv"), index=False)

# --- Aggregate per canonical phoneme ---
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
print("Saved: mdd_phoneme_summary.csv")

plt.figure(figsize=(14,6))
x = np.arange(len(df_phoneme["canonical"]))
width = 0.25

plt.bar(x - width, df_phoneme["FAR"], width, label="FAR")
plt.bar(x,         df_phoneme["FRR"], width, label="FRR")
plt.bar(x + width, df_phoneme["DER"], width, label="DER")

plt.xticks(x, df_phoneme["canonical"], rotation=90)
plt.ylabel("Error Rate")
plt.title("Phoneme-Level Error Rates (FAR / FRR / DER)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "phoneme_error_rates.png"))
plt.close()

# ==================== STAGE 2: ATTRIBUTE-LEVEL SUMMARY ====================

# --- Load and index phoneme→attribute mapping ---
df_p2a = pd.read_csv(PHONEME2ATT)
df_p2a.set_index(df_p2a.columns[0], inplace=True)   # first column = phoneme symbols

attr_summary = []

for att in df_p2a.columns:
    # list of phonemes that possess this attribute (value==1)
    pos_phonemes = df_p2a.index[df_p2a[att] == 1].tolist()
    subset = df_phoneme[df_phoneme["canonical"].isin(pos_phonemes)]
    if len(subset) == 0:
        continue
    row = {
        "Attribute": att,
        "n_phonemes": len(pos_phonemes),
        "Mean_FAR": subset["FAR"].mean(),
        "Mean_FRR": subset["FRR"].mean(),
        "Mean_DER": subset["DER"].mean()
    }
    attr_summary.append(row)

df_attr = pd.DataFrame(attr_summary).sort_values("Mean_DER", ascending=False)
df_attr.to_csv(os.path.join(OUT_DIR, "mdd_attribute_summary.csv"), index=False)
print("Saved: mdd_attribute_summary.csv")

# --- ATTRIBUTE-LEVEL ERROR RATES ---
plt.figure(figsize=(14,6))
x = np.arange(len(df_attr["Attribute"]))
width = 0.25

plt.bar(x - width, df_attr["Mean_FAR"], width, label="FAR")
plt.bar(x,         df_attr["Mean_FRR"], width, label="FRR")
plt.bar(x + width, df_attr["Mean_DER"], width, label="DER")

plt.xticks(x, df_attr["Attribute"], rotation=90)
plt.ylabel("Error Rate")
plt.title("Attribute-Level Error Rates (FAR / FRR / DER)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attribute_error_rates.png"))
plt.close()

# ==================== STAGE 3: DEMOGRAPHIC ANALYSIS ====================
def aggregate_demographic(df, group_vars):
    """Aggregate MDD metrics per demographic group."""
    metrics = ["TA", "FR", "FA", "TR", "CD", "DE"]
    out = df.groupby(group_vars)[metrics].sum().reset_index()
    out["FAR"] = out["FA"] / (out["FA"] + out["TR"] + 1e-8)
    out["FRR"] = out["FR"] / (out["FR"] + out["TA"] + 1e-8)
    out["DER"] = out["DE"] / (out["CD"] + out["DE"] + 1e-8)
    return out


# --- Age groups ---
df_phoneme_detail["age_group"] = (
    df_phoneme_detail["age_year"]
    .astype(int)
    .astype(str)
)

# ensure correct numeric ordering on plots
df_phoneme_detail["age_group"] = pd.Categorical(
    df_phoneme_detail["age_group"],
    categories=sorted(df_phoneme_detail["age_group"].unique(), key=int),
    ordered=True
)

# --- Aggregate by age and gender ---
df_demo_age = aggregate_demographic(df_phoneme_detail, ["age_group"])
df_demo_gender = aggregate_demographic(df_phoneme_detail, ["gender"])

df_demo_age.to_csv(os.path.join(OUT_DIR, "mdd_age_summary.csv"), index=False)
df_demo_gender.to_csv(os.path.join(OUT_DIR, "mdd_gender_summary.csv"), index=False)
print("Saved: demographic summaries (age & gender)")


# --- Plot: age trend ---
plt.figure(figsize=(8,5))
plt.plot(df_demo_age["age_group"], df_demo_age["FAR"], marker='o', label="FAR")
plt.plot(df_demo_age["age_group"], df_demo_age["FRR"], marker='o', label="FRR")
plt.plot(df_demo_age["age_group"], df_demo_age["DER"], marker='o', label="DER")
plt.legend()
plt.title("Error Rates by Age Group")
plt.ylabel("Rate")
plt.xlabel("Age Group (years)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "demographic_age_trends.png"))
plt.close()


# --- Plot: gender ---
plt.figure(figsize=(6,5))
x = np.arange(len(df_demo_gender["gender"]))
width = 0.25

plt.bar(x - width, df_demo_gender["FAR"], width, label="FAR")
plt.bar(x,         df_demo_gender["FRR"], width, label="FRR")
plt.bar(x + width, df_demo_gender["DER"], width, label="DER")

plt.xticks(x, df_demo_gender["gender"])
plt.ylabel("Rate")
plt.title("Error Rates by Gender")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "demographic_gender_trends.png"))
plt.close()

# ==================== STAGE 4: ATTRIBUTE DIFFERENCE ANALYSIS ====================
TOP_N = 10

# --- Identify top-error phonemes (by FAR/FRR/DER) ---
df_far_sorted = df_phoneme.sort_values("FAR", ascending=False).head(TOP_N)
df_frr_sorted = df_phoneme.sort_values("FRR", ascending=False).head(TOP_N)
df_der_sorted = df_phoneme.sort_values("DER", ascending=False).head(TOP_N)
high_err_phonemes = pd.concat([
    df_far_sorted["canonical"],
    df_frr_sorted["canonical"],
    df_der_sorted["canonical"]
]).dropna().unique()

# --- Normalise attribute prediction columns to list[str] of per-attribute sequences ---
def _to_list_of_str(x):
    if isinstance(x, list):
        return [str(y) for y in x]
    # numpy array
    if hasattr(x, "tolist"):
        return [str(y) for y in x.tolist()]
    if isinstance(x, str):
        # stringified list? try to parse
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(y) for y in v]
            except Exception:
                pass
        return [s]
    if pd.isna(x):
        return []
    return [str(x)]

df_attr_pred = df_pred.copy()
df_attr_pred["target_list"] = df_attr_pred["target_text"].apply(_to_list_of_str)
df_attr_pred["pred_list"]   = df_attr_pred["pred_str"].apply(_to_list_of_str)

records = []

# --- Loop over high-error phonemes ---
for ph in high_err_phonemes:
    diff_counter = Counter()
    total_occurrences = 0

    # rows where canonical contains this phoneme (word-level)
    samples = df_mdd[df_mdd["canonical"].astype(str).str.contains(rf"\b{ph}\b", na=False)]

    for idx, row in samples.iterrows():
        can_list = str(row["canonical"]).split()
        pos_list = [i for i, p in enumerate(can_list) if p == ph]
        if not pos_list:
            continue

        # fetch aligned attribute rows (same row index assumption)
        if idx not in df_attr_pred.index:
            continue
        tgt_attr_rows = df_attr_pred.at[idx, "target_list"]  # list[str], each "n_attr ... p_attr"
        prd_attr_rows = df_attr_pred.at[idx, "pred_list"]

        if not isinstance(tgt_attr_rows, list) or not isinstance(prd_attr_rows, list):
            continue

        # ensure same number of attributes on both sides
        K = min(len(tgt_attr_rows), len(prd_attr_rows))
        if K == 0:
            continue

        # for each occurrence of this phoneme position
        for pos in pos_list:
            flipped_any = False

            # iterate over attributes (rows)
            for k in range(K):
                tgt_states = str(tgt_attr_rows[k]).split()  # e.g. "n_a n_a p_a ..." → ["n_a","n_a","p_a",...]
                prd_states = str(prd_attr_rows[k]).split()

                if pos >= len(tgt_states) or pos >= len(prd_states):
                    continue

                t_state = tgt_states[pos]  # "n_a"
                p_state = prd_states[pos]  # "p_a"

                # valid pair and same attribute name?
                if (isinstance(t_state, str) and isinstance(p_state, str) and
                    (t_state.startswith(("n_","p_")) and p_state.startswith(("n_","p_")))):
                    attr_t = t_state[2:]
                    attr_p = p_state[2:]
                    if attr_t == attr_p and (t_state[0] != p_state[0]):
                        diff_counter[attr_t] += 1
                        flipped_any = True

        # count this phoneme occurrence (whether flipped or not)
        total_occurrences += len(pos_list)

    if total_occurrences > 0:
        top_diffs = diff_counter.most_common(5)
        rec = {
            "Phoneme": ph,
            "Samples": total_occurrences,
            "Top_Attribute_Differences": ", ".join([f"{k}({v})" for k, v in top_diffs]) if top_diffs else "",
            "FAR": float(df_phoneme.loc[df_phoneme["canonical"] == ph, "FAR"].values[0]),
            "FRR": float(df_phoneme.loc[df_phoneme["canonical"] == ph, "FRR"].values[0]),
            "DER": float(df_phoneme.loc[df_phoneme["canonical"] == ph, "DER"].values[0]),
        }
        records.append(rec)

df_diff = pd.DataFrame(records)
df_diff.to_csv(os.path.join(OUT_DIR, "mdd_attribute_differences.csv"), index=False)


# --- Plot top-error phonemes by FAR / FRR / DER ---
plt.figure(figsize=(10,6))
plt.bar(df_far_sorted["canonical"], df_far_sorted["FAR"], color="tab:red")
plt.title(f"Top {TOP_N} Phonemes with Highest FAR")
plt.ylabel("FAR")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_phonemes_FAR.png"))
plt.close()

plt.figure(figsize=(10,6))
plt.bar(df_frr_sorted["canonical"], df_frr_sorted["FRR"], color="tab:orange")
plt.title(f"Top {TOP_N} Phonemes with Highest FRR")
plt.ylabel("FRR")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_phonemes_FRR.png"))
plt.close()

plt.figure(figsize=(10,6))
plt.bar(df_der_sorted["canonical"], df_der_sorted["DER"], color="tab:purple")
plt.title(f"Top {TOP_N} Phonemes with Highest DER")
plt.ylabel("DER")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_phonemes_DER.png"))
plt.close()

print("Saved: top_phonemes_FAR.png / FRR.png / DER.png")

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
plt.savefig(os.path.join(OUT_DIR, "attribute_flip_counts_high_error_phonemes.png"))
plt.close()

print("Saved: attribute_flip_counts_high_error_phonemes.png")

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
plt.savefig(os.path.join(OUT_DIR, "attribute_flip_stackedbar_by_phoneme.png"))
plt.close()

print("Saved: attribute_flip_stackedbar_by_phoneme.png")