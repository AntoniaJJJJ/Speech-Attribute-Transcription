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
- Guarantee 100% consistency with train.py / transcribe_SA evaluation.

Outputs:
    mdd_summary.txt
    mdd_sample_detail.csv
    edit_ops_totals.csv
============================================================
"""

import os
import pandas as pd
from collections import Counter
from datasets import load_from_disk
from metrics.cm import phoneme_confusion_matrix

# ==================== CONFIG ====================

PREDICTED_DATASET = "/srv/scratch/z5369417/outputs/transcriber_result/cu_model_exp11_test_on_unsw"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_phoneme_level_exp11"
os.makedirs(OUT_DIR, exist_ok=True)

PHONEME_CANONICAL = "phoneme_unsw"
PHONEME_SPOKEN = "actual_spoken_phonemes"
PHONEME_PREDICTED = "pred_phoneme"
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

def decouple_diphthongs(phoneme_str, diph_map):
    """Apply diphthong-to-monophthong mapping."""
    phs = phoneme_str.strip().split()
    out = []
    for p in phs:
        if p in diph_map:
            out.extend(diph_map[p])
        else:
            out.append(p)
    return out

# ==================== LOAD DATA ====================

dataset = load_from_disk(PREDICTED_DATASET)
diph_map = load_diphthong_map(DIPHTHONG_MAP_FILE)

# Use same alignment logic as transcribe_SA / cm.py
cm_metric = phoneme_confusion_matrix()
align_fn = cm_metric._levenshtein_align  # ensures exact match with cm.py behavior

global_counts = Counter()
edit_ops = Counter()
sample_rows = []

# ==================== MAIN EVALUATION ====================

for sample in dataset:
    canonical = decouple_diphthongs(sample[PHONEME_CANONICAL], diph_map)
    spoken = decouple_diphthongs(sample[PHONEME_SPOKEN], diph_map)
    predicted = decouple_diphthongs(sample[PHONEME_PREDICTED], diph_map)

    align_can = align_fn(canonical, predicted)
    align_spo = align_fn(spoken, predicted)

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
