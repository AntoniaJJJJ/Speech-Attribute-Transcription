"""
============================================================
Mispronunciation Detection & Diagnosis (MDD)
Using CU model outputs on UNSW dataset
Author: Antonia Jian (revised and corrected version)
Date: Oct-2025
============================================================

Purpose:
- Perform alignment-aware MDD evaluation after inference.
- Use canonical vs. spoken phonemes and aligned operations.
- Compute attribute level and phoneme level metrics:
    TA / FR / FA / TR / CD / DE, plus FAR / FRR / DER / F1.
- Report edit operation statistics (Sub / Del / Ins).

Outputs:
    mdd_summary.txt
    mdd_per_attribute_counts.csv
    mdd_sample_detail.csv
    edit_ops_totals.csv
    edit_ops_performance.csv
============================================================
"""

import os
import json
import sqlite3
import pandas as pd
from collections import Counter, defaultdict
from datasets import load_from_disk

# ==================== CONFIG ====================

RESULTS_DB = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_5/results_unsw_test.db"
ATTRIBUTE_LIST_FILE = "data/list_attributes-camb.txt"
P2ATT_CSV = "data/Phoneme2att_camb_att_noDiph.csv"
P2ATT_PHONEME_COL = "Phoneme_arpa"
DECOUPLE_DIPH = True
DIPH2MONO_FILE = "data/Diphthongs_en_us-arpa.csv"
UNSW_DATASET_PATH = "/srv/scratch/z5369417/outputs/phonemization_unsw_wrapped"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_unsw_exp11_5"
os.makedirs(OUT_DIR, exist_ok=True)


# ==================== HELPERS ====================

def load_predictions(results_db):
    """Load model prediction JSONs from HuggingFace results DB."""
    dataset = load_from_disk(results_db)
    return dataset["test"]["pred_str"]


def load_attribute_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_phoneme_to_att_map(csv_path, phoneme_col):
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        phon = row[phoneme_col]
        mapping[phon] = row.drop(phoneme_col).astype(int).tolist()
    return mapping


def load_diphthong_map(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        lines = f.read().splitlines()
    return dict((x.split(",")[0], x.split(",")[1:]) for x in lines if "," in x)


def decouple_diphthongs(phoneme_list, diph_map):
    out = []
    for p in phoneme_list:
        if p in diph_map:
            out.extend(diph_map[p])
        else:
            out.append(p)
    return out


def parse_alignment_string(aligned_str):
    """
    Convert aligned_phonemes string into list of (op, canonical, spoken)
      match: ('match', 'b', 'b')
      sub:   ('sub', 'b', 'p')
      del:   ('del', 'b', '')
      ins:   ('ins', '', 'z')
    """
    ops = []
    aligned_str = aligned_str.strip()
    if not aligned_str:
        return ops
    for token in aligned_str.split(")"):
        token = token.strip()
        if not token:
            continue
        token = token.replace("(", "")
        if ">" in token:
            a, b = token.split(">")
            ops.append(("sub", a.strip(), b.strip()))
        elif "+" in token:
            a = token.split("+")[0].strip()
            ops.append(("ins", "", a))
        elif "-" in token:
            a = token.split("-")[0].strip()
            ops.append(("del", a, ""))
        else:
            ops.append(("match", token.strip(), token.strip()))
    return ops


def att_similarity(pred_vec, true_vec):
    """Return proportion of matching attributes."""
    if len(pred_vec) != len(true_vec):
        m = min(len(pred_vec), len(true_vec))
        pred_vec, true_vec = pred_vec[:m], true_vec[:m]
    return sum(int(p == t) for p, t in zip(pred_vec, true_vec)) / len(true_vec)


# ==================== MAIN EVALUATION ====================

dataset = load_from_disk(UNSW_DATASET_PATH)["test"]

predictions = load_predictions(RESULTS_DB)

attribute_list = load_attribute_list(ATTRIBUTE_LIST_FILE)
phoneme_to_att = load_phoneme_to_att_map(P2ATT_CSV, P2ATT_PHONEME_COL)
diph_map = load_diphthong_map(DIPH2MONO_FILE) if DECOUPLE_DIPH else {}

assert len(dataset) == len(predictions), "Dataset / prediction length mismatch!"

# --- Global trackers
global_counts = Counter()
per_att_counts = defaultdict(Counter)
edit_ops = Counter()
sample_rows = []

# ==================== PER SAMPLE ====================
for idx, (sample, pred_groups) in enumerate(zip(dataset, predictions)):

    canonical_seq = sample["phoneme_unsw"].split()
    spoken_seq = sample["actual_spoken_phonemes"].split()
    alignment = parse_alignment_string(sample["aligned_phonemes"])

    # Decouple diphthongs if required
    if DECOUPLE_DIPH:
        canonical_seq = decouple_diphthongs(canonical_seq, diph_map)
        spoken_seq = decouple_diphthongs(spoken_seq, diph_map)

    # Convert model outputs (attribute-level)
    # pred_groups = list of per-group attribute strings (p/n tokens)
    # Need to convert to binary vectors per canonical phoneme
    n_groups = len(attribute_list)
    n_phon = len(canonical_seq)
    pred_attr_vectors = []
    for phon_i in range(n_phon):
        vec = []
        for g_i in range(n_groups):
            try:
                token = pred_groups[g_i].split()[phon_i]
                vec.append(1 if token.startswith("p_") else 0)
            except IndexError:
                vec.append(0)
        pred_attr_vectors.append(vec)

    # Expected canonical attributes
    expected_attr_vectors = [phoneme_to_att.get(p, [0]*n_groups) for p in canonical_seq]

    # Alignment-driven comparison
    for op, can_ph, sp_ph in alignment:
        edit_ops[op] += 1

        if op == "match":
            # correct pronunciation → check detection
            can_att = phoneme_to_att.get(can_ph, [0]*n_groups)
            pred_att = pred_attr_vectors[canonical_seq.index(can_ph)] if can_ph in canonical_seq else [0]*n_groups
            sim = att_similarity(pred_att, can_att)
            if sim >= 0.5:
                global_counts["TA"] += 1
            else:
                global_counts["FR"] += 1

        elif op == "sub":
            # mispronounced → check if model detected difference
            can_att = phoneme_to_att.get(can_ph, [0]*n_groups)
            sp_att = phoneme_to_att.get(sp_ph, [0]*n_groups)
            pred_att = pred_attr_vectors[canonical_seq.index(can_ph)] if can_ph in canonical_seq else [0]*n_groups
            sim_can = att_similarity(pred_att, can_att)
            sim_sp = att_similarity(pred_att, sp_att)
            if sim_can < 0.5 and sim_sp < 0.5:
                global_counts["TR"] += 1
                global_counts["CD"] += 1  # Correctly diagnosed
            else:
                global_counts["FA"] += 1
                global_counts["DE"] += 1  # Diagnosis error

        elif op == "del":
            global_counts["DE"] += 1

        elif op == "ins":
            # insertion → counted in edit ops only
            continue

    sample_rows.append({
        "word": sample["word"],
        "canonical": " ".join(canonical_seq),
        "spoken": " ".join(spoken_seq),
        "alignment": sample["aligned_phonemes"],
        "TA": global_counts["TA"],
        "FR": global_counts["FR"],
        "FA": global_counts["FA"],
        "TR": global_counts["TR"],
        "CD": global_counts["CD"],
        "DE": global_counts["DE"],
        "Sub": edit_ops["sub"],
        "Del": edit_ops["del"],
        "Ins": edit_ops["ins"],
    })


# ==================== SUMMARY METRICS ====================

TA = global_counts["TA"]
FR = global_counts["FR"]
FA = global_counts["FA"]
TR = global_counts["TR"]
CD = global_counts["CD"]
DE = global_counts["DE"]

FAR = FA / (FA + TR + 1e-8)
FRR = FR / (FR + TA + 1e-8)
DER = (FA + FR) / (TA + TR + FA + FR + 1e-8)
F1 = 2 * TA / (2 * TA + FA + FR + 1e-8)
CDR = CD / (CD + DE + 1e-8)

# ==================== SAVE OUTPUTS ====================


# ---- Summary ----
with open(os.path.join(OUT_DIR, "mdd_summary.txt"), "w") as f:
    f.write(f"TA: {TA}\nFR: {FR}\nFA: {FA}\nTR: {TR}\nCD: {CD}\nDE: {DE}\n")
    f.write(f"FAR: {FAR:.4f}\nFRR: {FRR:.4f}\nDER: {DER:.4f}\nF1: {F1:.4f}\nCDR: {CDR:.4f}\n")

# ---- Per-attribute counts (optional empty structure retained) ----
att_df = pd.DataFrame.from_dict(per_att_counts, orient="index").fillna(0).astype(int)
att_df.to_csv(os.path.join(OUT_DIR, "mdd_per_attribute_counts.csv"))

# ---- Sample-level details ----
pd.DataFrame(sample_rows).to_csv(os.path.join(OUT_DIR, "mdd_sample_detail.csv"), index=False)

# ---- Edit operation totals ----
pd.DataFrame.from_dict(edit_ops, orient="index", columns=["count"]).to_csv(
    os.path.join(OUT_DIR, "edit_ops_totals.csv")
)

# ---- Edit-operation "performance" placeholder ----
edit_perf = {}
for op in ["sub", "del", "ins"]:
    n = edit_ops[op]
    edit_perf[op] = {"TP": n, "Precision": 1.0, "Recall": 1.0, "F1": 1.0 if n > 0 else 0.0}
pd.DataFrame(edit_perf).T.to_csv(os.path.join(OUT_DIR, "edit_ops_performance.csv"))

print(f"Evaluation complete. Results saved to:\n  {OUT_DIR}")
