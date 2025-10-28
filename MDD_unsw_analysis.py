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
from datasets import load_from_disk
from collections import Counter

# ==================== CONFIG ====================

RESULTS_DB = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_5/results_unsw_test.db"
MDD_DETAIL = "/srv/scratch/z5369417/outputs/mdd_unsw_phoneme_level_exp11/mdd_sample_detail.csv"
UNSW_META  = "/srv/scratch/z5369417/UNSW_final_deliverables/CAAP_2023-04-27/dataset_spreadsheet.xlsx"
PHONEME2ATT = "data/p2att_en_us-arpa_noDiph_camb.csv"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_unsw_analysis_exp11"

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

# alignment utilities (from script 4)
_default_transform = tr.Compose([
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    tr.ReduceToSingleSentence(),
    tr.ReduceToListOfListOfWords(),
])

def preprocess_for_alignment(ref_str, pred_str):
    truth = _default_transform(ref_str)
    hypothesis = _default_transform(pred_str)
    assert len(truth) == 1 and len(hypothesis) == 1
    truth, hypothesis = truth[0], hypothesis[0]
    mapper = dict([(k, v) for v, k in enumerate(set(truth + hypothesis))])
    truth_chars = [chr(mapper[p]) for p in truth]
    pred_chars = [chr(mapper[p]) for p in hypothesis]
    return truth, hypothesis, "".join(truth_chars), "".join(pred_chars)

def align_like_cm(ref_phonemes, pred_phonemes):
    ref_str = " ".join(ref_phonemes)
    pred_str = " ".join(pred_phonemes)
    ref_list, pred_list, ref_chars, pred_chars = preprocess_for_alignment(ref_str, pred_str)
    ops = Levenshtein.editops(ref_chars, pred_chars)
    aligned_pairs = []
    ref_i, pred_i = 0, 0
    for op, i1, i2 in ops:
        while ref_i < i1 and pred_i < i2:
            aligned_pairs.append((ref_list[ref_i], pred_list[pred_i]))
            ref_i += 1
            pred_i += 1
        if op == "insert":
            aligned_pairs.append(("", pred_list[i2])); pred_i += 1
        elif op == "delete":
            aligned_pairs.append((ref_list[i1], "")); ref_i += 1
        elif op == "replace":
            aligned_pairs.append((ref_list[i1], pred_list[i2])); ref_i += 1; pred_i += 1
    while ref_i < len(ref_list) and pred_i < len(pred_list):
        aligned_pairs.append((ref_list[ref_i], pred_list[pred_i]))
        ref_i += 1; pred_i += 1
    while ref_i < len(ref_list):
        aligned_pairs.append((ref_list[ref_i], "")); ref_i += 1
    while pred_i < len(pred_list):
        aligned_pairs.append(("", pred_list[pred_i])); pred_i += 1
    return aligned_pairs


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

    align_can = align_like_cm(can, pred)
    align_spo = align_like_cm(spo, pred)

    for (can_ph, pred_ph), (_, spo_ph) in zip(align_can, align_spo):
        record = {
            "canonical": can_ph,
            "predicted": pred_ph,
            "spoken": spo_ph,
            "TA": 0, "FR": 0, "FA": 0, "TR": 0, "CD": 0, "DE": 0
        }

        # --- classification (same logic as MDD script) ---
        if can_ph == '' and pred_ph != '':
            record["TR"] = 1
            record["CD" if pred_ph == spo_ph else "DE"] = 1
        elif can_ph != '' and pred_ph == '':
            record["TR"] = 1
            record["CD" if spo_ph == '' else "DE"] = 1
        elif can_ph == pred_ph:
            if can_ph == spo_ph:
                record["TA"] = 1
            else:
                record["FA"] = 1
        else:
            if can_ph == spo_ph:
                record["FR"] = 1
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
df_phoneme.to_csv(os.path.join(OUT_DIR, "mdd_phoneme_summary.csv"), index=False)
print("Saved: mdd_phoneme_summary.csv")

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
        "Mean_DER": subset["DER"].mean(),
        "Mean_TA": subset["TA"].sum() / (subset["TA"].sum() + subset["FR"].sum() + 1e-8),
        "Mean_FA": subset["FA"].sum() / (subset["FA"].sum() + subset["TR"].sum() + 1e-8)
    }
    attr_summary.append(row)

df_attr = pd.DataFrame(attr_summary).sort_values("Mean_TA", ascending=False)
df_attr.to_csv(os.path.join(OUT_DIR, "mdd_attribute_summary.csv"), index=False)
print("Saved: mdd_attribute_summary.csv")

# --- Attribute heatmap ---
plt.figure(figsize=(8,6))
plt.imshow(df_attr[["Mean_TA","Mean_FA","Mean_FAR","Mean_FRR","Mean_DER"]].T,
           aspect='auto', cmap="viridis")
plt.yticks(range(5), ["TA","FA","FAR","FRR","DER"])
plt.xticks(range(len(df_attr)), df_attr["Attribute"], rotation=90)
plt.title("Attribute-Level Error Distribution")
plt.colorbar(label="Mean Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attribute_error_heatmap.png"))
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
bins = [0, 5, 7, 9, 11, 13, 20]
labels = ["≤5", "6-7", "8-9", "10-11", "12-13", "≥14"]
df_phoneme_detail["age_group"] = pd.cut(df_phoneme_detail["age_year"],
                                        bins=bins, labels=labels, right=False)

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
plt.figure(figsize=(5,5))
barw = 0.3
genders = df_demo_gender["gender"]
plt.bar(genders, df_demo_gender["FAR"], width=barw, label="FAR")
plt.bar(genders, df_demo_gender["FRR"], width=barw, bottom=df_demo_gender["FAR"], label="FRR")
plt.title("Error Rates by Gender")
plt.ylabel("Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "demographic_gender_trends.png"))
plt.close()

# ==================== STAGE 4: ATTRIBUTE DIFFERENCE ANALYSIS ====================
TOP_N = 10

# --- Use actual predicted vs target attribute strings ---
df_attr_pred = df_pred[["target_text", "pred_str"]].copy()
df_attr_pred["target_list"] = df_attr_pred["target_text"].astype(str).fillna("").apply(lambda x: x.split())
df_attr_pred["pred_list"]   = df_attr_pred["pred_str"].astype(str).fillna("").apply(lambda x: x.split())

def extract_attr_name(attr_token):
    """Convert 'p_voice'/'n_voice' → 'voice'."""
    return attr_token[2:] if isinstance(attr_token, str) and "_" in attr_token else attr_token

def compare_attr_lists(target_list, pred_list):
    """Return dict of attribute name → +1 (predicted 'p'), -1 (predicted 'n') when flipped."""
    diffs = {}
    for t, p in zip(target_list, pred_list):
        if t != p and "_" in t and "_" in p:
            base = extract_attr_name(t)
            diffs[base] = diffs.get(base, 0) + 1
    return diffs

# --- Identify top-error phonemes (from Stage 1 summary) ---
df_far_sorted = df_phoneme.sort_values("FAR", ascending=False).head(TOP_N)
df_frr_sorted = df_phoneme.sort_values("FRR", ascending=False).head(TOP_N)
high_err_phonemes = pd.concat([df_far_sorted["canonical"], df_frr_sorted["canonical"]]).unique()

records = []

for ph in high_err_phonemes:
    subset = df_phoneme_detail[df_phoneme_detail["canonical"] == ph]
    if len(subset) == 0:
        continue

    diff_counter = Counter()
    valid_pairs = 0

    # compare corresponding attribute strings from df_pred (same order)
    for i in range(min(len(df_attr_pred), len(subset))):
        diffs = compare_attr_lists(df_attr_pred.iloc[i]["target_list"],
                                   df_attr_pred.iloc[i]["pred_list"])
        if diffs:
            for k, v in diffs.items():
                diff_counter[k] += v
            valid_pairs += 1

    if valid_pairs > 0:
        top_diffs = diff_counter.most_common(5)
        rec = {
            "Phoneme": ph,
            "Samples": valid_pairs,
            "Top_Attribute_Differences": ", ".join([f"{k}({v})" for k, v in top_diffs]),
            "FAR": df_phoneme.loc[df_phoneme["canonical"] == ph, "FAR"].values[0],
            "FRR": df_phoneme.loc[df_phoneme["canonical"] == ph, "FRR"].values[0],
            "DER": df_phoneme.loc[df_phoneme["canonical"] == ph, "DER"].values[0],
        }
        records.append(rec)

df_diff = pd.DataFrame(records)
df_diff.to_csv(os.path.join(OUT_DIR, "mdd_attribute_differences.csv"), index=False)
print("Saved: mdd_attribute_differences.csv")

# --- Plots of high-error phonemes ---
plt.figure(figsize=(10,6))
plt.bar(df_far_sorted["canonical"], df_far_sorted["FAR"], color="tab:red")
plt.title(f"Top {TOP_N} Phonemes – Highest FAR")
plt.ylabel("FAR")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_phonemes_FAR.png"))
plt.close()

plt.figure(figsize=(10,6))
plt.bar(df_frr_sorted["canonical"], df_frr_sorted["FRR"], color="tab:orange")
plt.title(f"Top {TOP_N} Phonemes – Highest FRR")
plt.ylabel("FRR")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_phonemes_FRR.png"))
plt.close()