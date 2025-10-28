"""
============================================================
Phoneme-level Mispronunciation Detection & Diagnosis (MDD)
Using transcribe_SA outputs (CU model on SpeechOcean dataset)
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

PREDICTED_DATASET = "/srv/scratch/z5369417/outputs/transcriber_result/cu_model_exp11_test_on_speechocean"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_speechocean_phoneme_level_exp11"
os.makedirs(OUT_DIR, exist_ok=True)

PHONEME_CANONICAL = "phoneme_speechocean"
PHONEME_SPOKEN = "actual_spoken_phonemes"
PHONEME_PREDICTED = "pred_phoneme"
DECOUPLE_DIPH = True
DIPHTHONG_MAP_FILE = "data/Diphthongs_en_us-arpa.csv"

# ==================== HELPERS ====================

def load_diphthong_map(path):
    if not os.path.exists(path):
        print(f"[Warning] Diphthong map not found: {path}")
        return {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if "," in l]
    return {x.split(",")[0].strip(): [p.strip() for p in x.split(",")[1:]] for x in lines}


def decouple_diphthongs(phoneme_str, diph_map, decouple=False):
    phs = phoneme_str.strip().split()
    out = []
    for p in phs:
        if decouple and p in diph_map:
            out.extend(diph_map[p])
        else:
            out.append(p)
    return out


# ==================== ALIGNMENT ====================

_default_transform = tr.Compose([
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    tr.ReduceToSingleSentence(),
    tr.ReduceToListOfListOfWords(),
])

def preprocess_for_alignment(ref_str, pred_str):
    truth = _default_transform(ref_str)[0]
    hypothesis = _default_transform(pred_str)[0]
    mapper = {p: i for i, p in enumerate(set(truth + hypothesis))}
    ref_chars = "".join(chr(mapper[p]) for p in truth)
    pred_chars = "".join(chr(mapper[p]) for p in hypothesis)
    return truth, hypothesis, ref_chars, pred_chars


def align_like_cm(ref_phonemes, pred_phonemes):
    ref_str, pred_str = " ".join(ref_phonemes), " ".join(pred_phonemes)
    ref_list, pred_list, ref_chars, pred_chars = preprocess_for_alignment(ref_str, pred_str)
    ops = Levenshtein.editops(ref_chars, pred_chars)
    aligned_pairs, ref_i, pred_i = [], 0, 0
    for op, i1, i2 in ops:
        while ref_i < i1 and pred_i < i2:
            aligned_pairs.append((ref_list[ref_i], pred_list[pred_i]))
            ref_i += 1; pred_i += 1
        if op == "insert":
            aligned_pairs.append(("", pred_list[i2])); pred_i += 1
        elif op == "delete":
            aligned_pairs.append((ref_list[i1], "")); ref_i += 1
        elif op == "replace":
            aligned_pairs.append((ref_list[i1], pred_list[i2])); ref_i += 1; pred_i += 1
    while ref_i < len(ref_list):
        aligned_pairs.append((ref_list[ref_i], "")); ref_i += 1
    while pred_i < len(pred_list):
        aligned_pairs.append(("", pred_list[pred_i])); pred_i += 1
    return aligned_pairs


# ==================== MAIN ====================

dataset = load_from_disk(PREDICTED_DATASET)
diph_map = load_diphthong_map(DIPHTHONG_MAP_FILE)

global_counts = Counter()
edit_ops = Counter()
sample_rows = []
per_op_counts = {
    "match": Counter(),
    "sub": Counter(),
    "del": Counter(),
    "ins": Counter()
}

for s in dataset:
    can = decouple_diphthongs(s[PHONEME_CANONICAL], diph_map, DECOUPLE_DIPH)
    spo = decouple_diphthongs(s[PHONEME_SPOKEN], diph_map, DECOUPLE_DIPH)
    pred = decouple_diphthongs(s[PHONEME_PREDICTED], diph_map, DECOUPLE_DIPH)

    align_can, align_spo = align_like_cm(can, pred), align_like_cm(spo, pred)

    for (c, p), (_, sp) in zip(align_can, align_spo):
        if c == '' and p != '':
            edit_ops["ins"] += 1
            global_counts["TR"] += 1
            if p == sp:
                global_counts["CD"] += 1; per_op_counts["ins"]["CD"] += 1
            else:
                global_counts["DE"] += 1; per_op_counts["ins"]["DE"] += 1
            per_op_counts["ins"]["TR"] += 1

        elif c != '' and p == '':
            edit_ops["del"] += 1
            global_counts["TR"] += 1
            if sp == '':
                global_counts["CD"] += 1; per_op_counts["del"]["CD"] += 1
            else:
                global_counts["DE"] += 1; per_op_counts["del"]["DE"] += 1
            per_op_counts["del"]["TR"] += 1

        elif c == p:
            edit_ops["match"] += 1
            if c == sp:
                global_counts["TA"] += 1; per_op_counts["match"]["TA"] += 1
            else:
                global_counts["FA"] += 1; per_op_counts["match"]["FA"] += 1

        else:
            edit_ops["sub"] += 1
            if c == sp:
                global_counts["FR"] += 1; per_op_counts["sub"]["FR"] += 1
            else:
                global_counts["TR"] += 1; per_op_counts["sub"]["TR"] += 1
                if p == sp:
                    global_counts["CD"] += 1; per_op_counts["sub"]["CD"] += 1
                else:
                    global_counts["DE"] += 1; per_op_counts["sub"]["DE"] += 1

    sample_rows.append({
        "canonical": s[PHONEME_CANONICAL],
        "spoken": s[PHONEME_SPOKEN],
        "predicted": s[PHONEME_PREDICTED],
    })

# ==================== METRICS ====================

TA, FR, FA, TR, CD, DE = [global_counts[k] for k in ["TA","FR","FA","TR","CD","DE"]]
FAR = FA / (FA + TR + 1e-8)
FRR = FR / (FR + TA + 1e-8)
DER = DE / (CD + DE + 1e-8)
F1  = 2 * TA / (2 * TA + FA + FR + 1e-8)
CDR = CD / (CD + DE + 1e-8)

# ==================== SAVE OUTPUTS ====================

# 1. Global summary
with open(os.path.join(OUT_DIR, "mdd_summary.txt"), "w") as f:
    f.write(f"TA: {TA}\nFR: {FR}\nFA: {FA}\nTR: {TR}\nCD: {CD}\nDE: {DE}\n")
    f.write(f"FAR: {FAR:.4f}\nFRR: {FRR:.4f}\nDER: {DER:.4f}\nF1: {F1:.4f}\nCDR: {CDR:.4f}\n")

# 2. Sample-level detail
pd.DataFrame(sample_rows).to_csv(os.path.join(OUT_DIR, "mdd_sample_detail.csv"), index=False)

# 3. Edit-operation totals
pd.DataFrame.from_dict(edit_ops, orient="index", columns=["count"]).to_csv(
    os.path.join(OUT_DIR, "edit_ops_totals.csv"))

# 4. Per-operation performance table
edit_ops_perf_df = pd.DataFrame(per_op_counts).T.fillna(0).astype(int)
edit_ops_perf_df = edit_ops_perf_df[["TA","FR","TR","FA","CD","DE"]]  # consistent column order
edit_ops_perf_df.to_csv(os.path.join(OUT_DIR, "edit_ops_performance.csv"))

print(f"SpeechOcean MDD complete.\nResults saved in:\n  {OUT_DIR}")