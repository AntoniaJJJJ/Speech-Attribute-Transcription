"""
============================================================
Phoneme-level Mispronunciation Detection & Diagnosis (MDD)
Using transcribe_SA outputs (CU model on UNSW dataset)
Author: Antonia Jian
Date: Oct 2025
============================================================

Purpose:
- Perform alignment-consistent MDD evaluation after phoneme decoding.
- Use canonical vs spoken phonemes, predicted phonemes, and alignments.
- Compute detection and diagnosis metrics (TA, FR, FA, TR, CD, DE)
  and edit operation statistics (Sub / Del / Ins).

Outputs:
    mdd_summary.txt
    mdd_sample_detail.csv
    edit_ops_totals.csv
    edit_ops_performance.csv
============================================================
"""

import os
import pandas as pd
from collections import Counter
from datasets import load_from_disk
import jiwer.transforms as tr
import Levenshtein

# ==================== CONFIG ====================

PREDICTED_DATASET = "/srv/scratch/z5369417/outputs/transcriber_result/cu_model_exp11_test_on_unsw"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_unsw_phoneme_level_exp11"
os.makedirs(OUT_DIR, exist_ok=True)

PHONEME_CANONICAL = "phoneme_unsw"
PHONEME_SPOKEN = "actual_spoken_phonemes"
PHONEME_PREDICTED = "pred_phoneme"
DECOUPLE_DIPH = True
DIPHTHONG_MAP_FILE = "data/Diphthongs_en_us-arpa.csv"

# ==================== HELPERS ====================

def load_diphthong_map(path):
    """Load diphthong→monophthong mapping file."""
    if not os.path.exists(path):
        print(f"[Warning] Diphthong map not found: {path}")
        return {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if "," in l]
    return dict(
        (x.split(",")[0].strip(), [p.strip() for p in x.split(",")[1:]])
        for x in lines
    )

def decouple_diphthongs(phoneme_str, diph_map, decouple=False):
    """Apply diphthong-to-monophthong mapping."""
    phs = phoneme_str.strip().split()
    out = []
    for p in phs:
        if decouple and p in diph_map:
            out.extend(diph_map[p])
        else:
            out.append(p)
    return out

# ==================== ALIGNMENT (same as cm.py) ====================

_default_transform = tr.Compose([
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    tr.ReduceToSingleSentence(),
    tr.ReduceToListOfListOfWords(),
])

def preprocess_for_alignment(ref_str, pred_str):
    """Exact preprocessing from cm.py."""
    truth = _default_transform(ref_str)
    hypothesis = _default_transform(pred_str)
    assert len(truth) == 1 and len(hypothesis) == 1

    truth = truth[0]
    hypothesis = hypothesis[0]

    # Create a shared symbol map (1 unique char per unique phoneme)
    mapper = dict([(k, v) for v, k in enumerate(set(truth + hypothesis))])
    truth_chars = [chr(mapper[p]) for p in truth]
    pred_chars = [chr(mapper[p]) for p in hypothesis]

    return truth, hypothesis, "".join(truth_chars), "".join(pred_chars)

def align_like_cm(ref_phonemes, pred_phonemes):
    """
    Reproduce cm.py alignment using Levenshtein.editops on char-mapped sequences.
    Returns a list of (ref_phoneme, pred_phoneme) tuples.
    """
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
            aligned_pairs.append(("", pred_list[i2]))
            pred_i += 1
        elif op == "delete":
            aligned_pairs.append((ref_list[i1], ""))
            ref_i += 1
        elif op == "replace":
            aligned_pairs.append((ref_list[i1], pred_list[i2]))
            ref_i += 1
            pred_i += 1

    # Append remaining aligned tail (if sequences end unevenly)
    while ref_i < len(ref_list) and pred_i < len(pred_list):
        aligned_pairs.append((ref_list[ref_i], pred_list[pred_i]))
        ref_i += 1
        pred_i += 1
    while ref_i < len(ref_list):
        aligned_pairs.append((ref_list[ref_i], ""))
        ref_i += 1
    while pred_i < len(pred_list):
        aligned_pairs.append(("", pred_list[pred_i]))
        pred_i += 1

    return aligned_pairs

# ==================== LOAD DATA ====================

dataset = load_from_disk(PREDICTED_DATASET)
diph_map = load_diphthong_map(DIPHTHONG_MAP_FILE)

global_counts = Counter()
edit_ops = Counter()
sample_rows = []


# ==================== MAIN EVALUATION ====================

for sample in dataset:
    canonical = decouple_diphthongs(sample[PHONEME_CANONICAL], diph_map, decouple=DECOUPLE_DIPH)
    spoken = decouple_diphthongs(sample[PHONEME_SPOKEN], diph_map, decouple=DECOUPLE_DIPH)
    predicted = decouple_diphthongs(sample[PHONEME_PREDICTED], diph_map, decouple=DECOUPLE_DIPH)

    align_can = align_like_cm(canonical, predicted)
    align_spo = align_like_cm(spoken, predicted)

    for (can_ph, pred_ph), (_, spo_ph) in zip(align_can, align_spo):
        if can_ph == '' and pred_ph != '':
            # Insertion
            edit_ops["ins"] += 1
            global_counts["TR"] += 1
            if pred_ph == spo_ph:
                global_counts["CD"] += 1
            else:
                global_counts["DE"] += 1

        elif can_ph != '' and pred_ph == '':
            # Deletion
            edit_ops["del"] += 1
            global_counts["TR"] += 1
            if spo_ph == '':
                global_counts["CD"] += 1  # correctly omitted
            else:
                global_counts["DE"] += 1  # wrongly diagnosed

        elif can_ph == pred_ph:
            # Match
            edit_ops["match"] += 1
            if can_ph == spo_ph:
                global_counts["TA"] += 1
            else:
                global_counts["FA"] += 1

        else:
            # Substitution
            edit_ops["sub"] += 1
            if can_ph == spo_ph:
                global_counts["FR"] += 1
            else:
                global_counts["TR"] += 1
                if pred_ph == spo_ph:
                    global_counts["CD"] += 1
                else:
                    global_counts["DE"] += 1

    sample_rows.append({
        "word": sample["word"],
        "canonical": sample[PHONEME_CANONICAL],
        "spoken": sample[PHONEME_SPOKEN],
        "predicted": sample[PHONEME_PREDICTED],
    })

# ==================== PER-OPERATION SUMMARY ====================

# prepare per-op matrix
per_op_counts = {
    "match": Counter(),
    "sub": Counter(),
    "del": Counter(),
    "ins": Counter()
}

# recompute counts per operation type
for sample in dataset:
    canonical = decouple_diphthongs(sample[PHONEME_CANONICAL], diph_map)
    spoken = decouple_diphthongs(sample[PHONEME_SPOKEN], diph_map)
    predicted = decouple_diphthongs(sample[PHONEME_PREDICTED], diph_map)

    align_can = align_like_cm(canonical, predicted)
    align_spo = align_like_cm(spoken, predicted)

    for (can_ph, pred_ph), (_, spo_ph) in zip(align_can, align_spo):

        # determine op type exactly as editops semantic
        if can_ph == '' and pred_ph != '':
            op = "ins"
            per_op_counts[op]["TR"] += 1
            per_op_counts[op]["CD" if pred_ph == spo_ph else "DE"] += 1

        elif can_ph != '' and pred_ph == '':
            op = "del"
            per_op_counts[op]["TR"] += 1
            per_op_counts[op]["CD" if spo_ph == '' else "DE"] += 1

        elif can_ph == pred_ph:
            op = "match"
            per_op_counts[op]["TA" if can_ph == spo_ph else "FA"] += 1

        else:
            op = "sub"
            if can_ph == spo_ph:
                per_op_counts[op]["FR"] += 1
            else:
                per_op_counts[op]["TR"] += 1
                per_op_counts[op]["CD" if pred_ph == spo_ph else "DE"] += 1

# convert to DataFrame (rows = ops, cols = metrics)
edit_ops_perf_df = pd.DataFrame(per_op_counts).T.fillna(0).astype(int)
edit_ops_perf_df = edit_ops_perf_df[["TA", "FR", "TR", "FA", "CD", "DE"]]  # ordered columns

# save to CSV
edit_ops_perf_path = os.path.join(OUT_DIR, "edit_ops_performance.csv")
edit_ops_perf_df.to_csv(edit_ops_perf_path)

# ==================== METRICS ====================

TA = global_counts["TA"]
FR = global_counts["FR"]
FA = global_counts["FA"]
TR = global_counts["TR"]
CD = global_counts["CD"]
DE = global_counts["DE"]


FAR = FA / (FA + TR + 1e-8)
FRR = FR / (FR + TA + 1e-8)
DER = DE / (CD + DE + 1e-8)
F1 = 2 * TA / (2 * TA + FA + FR + 1e-8)
CDR = CD / (CD + DE + 1e-8)

# ==================== SAVE OUTPUTS ====================

with open(os.path.join(OUT_DIR, "mdd_summary.txt"), "w") as f:
    f.write(f"TA: {TA}\nFR: {FR}\nFA: {FA}\nTR: {TR}\nCD: {CD}\nDE: {DE}\n")
    f.write(f"FAR: {FAR:.4f}\nFRR: {FRR:.4f}\nDER: {DER:.4f}\nF1: {F1:.4f}\nCDR: {CDR:.4f}\n")

pd.DataFrame(sample_rows).to_csv(os.path.join(OUT_DIR, "mdd_sample_detail.csv"), index=False)
pd.DataFrame.from_dict(edit_ops, orient="index", columns=["count"]).to_csv(
    os.path.join(OUT_DIR, "edit_ops_totals.csv")
)

print(f"Phoneme-level MDD evaluation complete.\nResults saved to:\n  {OUT_DIR}")
