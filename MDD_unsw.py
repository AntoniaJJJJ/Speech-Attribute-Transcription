"""
Mispronunciation Detection & Diagnosis on UNSW using CU model evaluation outputs
 - Attribute-level MDD: TA/FR/FA/TR, CD/DE, FAR/FRR/DER/F1
 - Phoneme-level: Substitution / Deletion / Insertion counts
 - Model detection performance with reference to edit operations (Precision, Recall, F1)
 
 Outputs here:
   - mdd_summary.txt                      (global totals + rates)
   - mdd_per_attribute_counts.csv         (TA/FR/FA/TR per attribute token)
   - mdd_sample_detail.csv                (per-sample & per-position detail summary)
   - edit_ops_totals.csv                  (total S/D/I)
   - edit_ops_performance.csv             (Precision/Recall/F1 per edit operation)

If diphthongs are decoupled in training, set DECOUPLE_DIPH=True and provide diph map.
"""

import os
import re
import json
import pandas as pd
from collections import Counter
from datasets import load_from_disk

# ============ CONFIG ============
RESULTS_DB = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_5/results_unsw_test.db"

# Use the SAME attribute list & p2att mapping the CU model used
ATTRIBUTE_LIST_FILE = "data/list_attributes-camb.txt"                 # one attribute per line
P2ATT_CSV           = "data/Phoneme2att_camb_att_noDiph.csv"         # has a phoneme column + binary attribute columns
P2ATT_PHONEME_COL   = "Phoneme_arpa"                                 # column name in the CSV for phoneme symbols

# Diphthong handling (match exp11 training)
DECOUPLE_DIPH = True
DIPH2MONO_FILE = "data/Diphthongs_en_us-arpa.csv"                    # "ai,a i" style rows

# Output dir
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_unsw_exp11_5"
os.makedirs(OUT_DIR, exist_ok=True)
# =================================

def load_attribute_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_p2att_map(csv_path, phoneme_col, attribute_list):
    df = pd.read_csv(csv_path)
    cols_missing = [a for a in attribute_list if a not in df.columns]
    if cols_missing:
        print(f"[WARN] Attributes missing in {csv_path}: {cols_missing} (will be ignored)")
    # Build mapper per attribute: phoneme -> p_/n_
    phoneme_binary_mappers = []
    for att in attribute_list:
        if att not in df.columns:
            phoneme_binary_mappers.append({})
            continue
        mapper = {}
        for _, r in df.iterrows():
            p = str(r[phoneme_col]).strip().lower()
            v = r[att]
            if pd.isna(v) or v not in [0, 1]:
                continue
            mapper[p] = f"p_{att}" if v == 1 else f"n_{att}"
        phoneme_binary_mappers.append(mapper)
    return phoneme_binary_mappers

def load_diph_map(path):
    # file lines: "ai,a i"
    d = {}
    with open(path) as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",") if x.strip()]
            if not parts: 
                continue
            diph = parts[0]
            monos = parts[1:]  # may be ["a i"] or ["a","i"] depending on file
            if len(monos) == 1 and " " in monos[0]:
                d[diph] = monos[0].split()
            else:
                d[diph] = monos
    return d

def decouple_diphthongs(seq, diph_map):
    toks = seq.split()
    out = []
    for t in toks:
        if t in diph_map:
            out.extend(diph_map[t])
        else:
            out.append(t)
    return " ".join(out)

def phonemes_to_attr_rows(phoneme_seq, phoneme_binary_mappers, attribute_list):
    """
    Map a phoneme sequence to a list of per-phoneme attribute-tag rows.
    Output shape: list of length T (T = #phonemes), each row is [p_/n_ tag for att1, att2, ...]
    """
    phs = phoneme_seq.lower().split()
    rows = []
    for p in phs:
        tags = []
        for mapper, att in zip(phoneme_binary_mappers, attribute_list):
            tag = mapper.get(p, None)  # None if phoneme unknown in mapping
            tags.append(tag)
        rows.append(tags)
    return rows  # T x A

def transpose_groups_text_to_perphoneme(group_str_list):
    """
    Convert the dataset fields (list of strings per attribute group) into per-phoneme rows.
    Input example: ["p_consonantal n_consonantal ...", "p_voiced n_voiced ...", ...]
    Output: list rows length T, each row a list of tags across attributes positions.
    """
    if not isinstance(group_str_list, list) or len(group_str_list) == 0:
        return []

    split_groups = [g.split() for g in group_str_list]
    # Ensure equal lengths; if not, truncate to min length
    min_len = min(len(g) for g in split_groups)
    split_groups = [g[:min_len] for g in split_groups]

    # transpose to T x A
    rows = []
    for i in range(min_len):
        rows.append([g[i] for g in split_groups])
    return rows

def levenshtein_alignment(src_tokens, tgt_tokens):
    """
    Return alignment of src -> tgt with ops (M/S, D, I).
    Outputs tuples: (src_tok_orNone, tgt_tok_orNone, op) where op in {"M","S","D","I"}.
    """
    n, m = len(src_tokens), len(tgt_tokens)
    dp = [[0]*(m+1) for _ in range(n+1)]
    back = [[None]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = i
        back[i][0] = ("D",)
    for j in range(1, m+1):
        dp[0][j] = j
        back[0][j] = ("I",)

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if src_tokens[i-1] == tgt_tokens[j-1] else 1
            choices = [
                (dp[i-1][j] + 1, "D"),           # delete src
                (dp[i][j-1] + 1, "I"),           # insert tgt
                (dp[i-1][j-1] + cost, "M" if cost == 0 else "S"),  # match/sub
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])

    # backtrack
    i, j = n, m
    align = []
    while i > 0 or j > 0:
        op = back[i][j][0]
        if op == "M" or op == "S":
            align.append((src_tokens[i-1], tgt_tokens[j-1], op))
            i -= 1; j -= 1
        elif op == "D":
            align.append((src_tokens[i-1], None, "D"))
            i -= 1
        else:  # I
            align.append((None, tgt_tokens[j-1], "I"))
            j -= 1
    align.reverse()
    return align

def f1(prec, rec):
    return 0.0 if (prec + rec) == 0 else 2*prec*rec/(prec+rec)

def main():
    # Load evaluated dataset (has target_text & pred_str)
    ds = load_from_disk(RESULTS_DB)
    if isinstance(ds, dict) or "test" in ds:
        ds = ds["test"]

    # Attribute inventory and mapper
    attributes = load_attribute_list(ATTRIBUTE_LIST_FILE)
    p2att_mappers = load_p2att_map(P2ATT_CSV, P2ATT_PHONEME_COL, attributes)

    # Diphthong map
    diph_map = load_diph_map(DIPH2MONO_FILE) if DECOUPLE_DIPH else {}

    # Global counters
    TA = FR = FA = TR = CD = DE = 0
    # per-attribute-token (p_att/n_att) counts
    att_tokens = [f"p_{a}" for a in attributes] + [f"n_{a}" for a in attributes]
    TA_attr = {t: 0 for t in att_tokens}
    FR_attr = {t: 0 for t in att_tokens}
    FA_attr = {t: 0 for t in att_tokens}
    TR_attr = {t: 0 for t in att_tokens}

    total_S = total_D = total_I = 0
    op_perf = Counter()
    op_stats = {"M": Counter(), "S": Counter(), "D": Counter(), "I": Counter()}

    # per-sample rows (summaries)
    sample_rows = []

    for ex in ds:
        # Canonical & Spoken phonemes (strings)
        canon = str(ex.get("phoneme_unsw", "")).strip().lower()
        spoken = str(ex.get("actual_spoken_phonemes", "")).strip().lower()

        if not canon or not spoken:
            continue

        if DECOUPLE_DIPH:
            canon = decouple_diphthongs(canon, diph_map)
            spoken = decouple_diphthongs(spoken, diph_map)

        canon_toks = canon.split()
        spoken_toks = spoken.split()

        # Alignment for S/D/I
        ali = levenshtein_alignment(canon_toks, spoken_toks)
        S = sum(1 for _,_,op in ali if op == "S")
        D = sum(1 for _,_,op in ali if op == "D")
        I = sum(1 for _,_,op in ali if op == "I")
        total_S += S; total_D += D; total_I += I

        # Attribute rows:
        # - canonical: from mapping
        # - spoken: from mapping
        canon_attr_rows = phonemes_to_attr_rows(" ".join(canon_toks), p2att_mappers, attributes)
        spoken_attr_rows = phonemes_to_attr_rows(" ".join(spoken_toks), p2att_mappers, attributes)

        # Predicted attributes per-phoneme (from pred_str)
        pred_group_strs = ex.get("pred_str", None)   # list of strings per attribute
        if not pred_group_strs:
            # skip if no prediction present
            continue
        pred_attr_rows = transpose_groups_text_to_perphoneme(pred_group_strs)

        # We need a per-position comparison in aligned space.
        # Build sequences of canonical-index and spoken-index along alignment:
        canon_idx = -1
        spoken_idx = -1

        # We'll also need canonical attribute row at current canonical index, etc.
        # If any tag is None (unmappable phoneme), we skip that position for attribute metrics.
        for (c_tok, s_tok, op) in ali:
            if op in ("M", "S"):   # both advance
                canon_idx += 1
                spoken_idx += 1

                # Defensive bounds
                if not (0 <= canon_idx < len(canon_attr_rows)) or not (0 <= spoken_idx < len(spoken_attr_rows)):
                    continue

                canon_vec  = canon_attr_rows[canon_idx]
                spoken_vec = spoken_attr_rows[spoken_idx]

                # Pick predicted row: align by canonical index (model predicts attributes for canonical positions)
                if not (0 <= canon_idx < len(pred_attr_rows)):
                    continue
                pred_vec = pred_attr_rows[canon_idx]

                # If any None in canonical mapping for this position, skip it (unseen phoneme in p2att)
                if any(t is None for t in canon_vec) or any(t is None for t in spoken_vec) or any(t is None for t in pred_vec):
                    continue

                gt_correct = (c_tok == s_tok)  # ground truth: correct or mispronounced at phoneme level
                model_accepts = (pred_vec == canon_vec)

                # edit-op detection stats
                pred_correct = model_accepts
                true_correct = (op=="M")
                # Only evaluate detection for M and S (skip D and I, since model has no predictions there)
                if op in ("M", "S"):
                    if  true_correct and  pred_correct:
                        op_perf["TP"] += 1
                        op_stats[op]["TP"] += 1
                    elif true_correct and not pred_correct:
                        op_perf["FN"] += 1
                        op_stats[op]["FN"] += 1
                    elif not true_correct and  pred_correct:
                        op_perf["FP"] += 1
                        op_stats[op]["FP"] += 1
                    elif not true_correct and not pred_correct:
                        op_perf["TN"] += 1
                        op_stats[op]["TN"] += 1

                if gt_correct:
                    if model_accepts:
                        TA += 1
                        for t in canon_vec:
                            TA_attr[t] += 1
                    else:
                        FR += 1
                        for t in canon_vec:
                            FR_attr[t] += 1
                else:
                    if model_accepts:
                        FA += 1
                        for t in canon_vec:
                            FA_attr[t] += 1
                    else:
                        TR += 1
                        # diagnosis correct if pred matches spoken attribute vector
                        if pred_vec == spoken_vec:
                            CD += 1
                        else:
                            DE += 1
                        for t in canon_vec:
                            TR_attr[t] += 1

            elif op == "D":
                # canonical advances only
                canon_idx += 1
                # In a deletion, there is no spoken phoneme; we skip attribute comparison
                continue
            else:  # "I"
                # insertion: spoken advances only
                spoken_idx += 1
                # No canonical index to compare against — skip
                continue

        # Save a per-sample summary row
        row = {
            "word": ex.get("word", ""),
            "age": ex.get("age", None),
            "gender": ex.get("gender", None),
            "speech_status": ex.get("speech_status", None),
            "S": S, "D": D, "I": I,
        }
        sample_rows.append(row)

    # Global metrics
    FAR = FA / (FA + TR) if (FA + TR) > 0 else 0.0
    FRR = FR / (FR + TA) if (FR + TA) > 0 else 0.0
    DER = DE / (CD + DE) if (CD + DE) > 0 else 0.0

    # Treat “error detected” as positive (TR), precision = TR/(TR+FA), recall = TR/(TR+FR)
    precision = TR / (TR + FA) if (TR + FA) > 0 else 0.0
    recall    = TR / (TR + FR) if (TR + FR) > 0 else 0.0
    F1        = f1(precision, recall)

    # Edit-op model performance
    op_perf_overall_prec = op_perf["TP"]/(op_perf["TP"]+op_perf["FP"]+1e-9)
    op_perf_overall_rec  = op_perf["TP"]/(op_perf["TP"]+op_perf["FN"]+1e-9)
    op_perf_overall_f1   = f1(op_perf_overall_prec, op_perf_overall_rec)
    per_op_rows=[]
    for op in ["M","S"]:  # Only include M and S, since model has no predictions for D or I
        TP, FP, FN = op_stats[op]["TP"], op_stats[op]["FP"], op_stats[op]["FN"]
        prec = TP / (TP + FP + 1e-9)
        rec  = TP / (TP + FN + 1e-9)
        f1s  = f1(prec, rec)
        per_op_rows.append({"op": op, "Precision": prec, "Recall": rec, "F1": f1s})

     # === Save outputs ===
    # 1) Global summary
    with open(os.path.join(OUT_DIR, "mdd_summary.txt"), "w") as f:
        f.write("===== MDD Summary (Attribute-level) =====\n")
        f.write(f"TA = {TA}\nFR = {FR}\nFA = {FA}\nTR = {TR}\n")
        f.write(f"  → CD = {CD}\n  → DE = {DE}\n\n")
        f.write(f"FAR = {FAR:.4f}\nFRR = {FRR:.4f}\nDER = {DER:.4f}\n")
        f.write(f"Precision = {precision:.4f}\nRecall = {recall:.4f}\nF1 = {F1:.4f}\n\n")
        f.write("===== Edit Operations (Phoneme-level) =====\n")
        f.write(f"Substitutions = {total_S}\nDeletions = {total_D}\nInsertions = {total_I}\n\n")
        f.write("===== Model Detection vs Edit Operations =====\n")
        f.write(f"Overall Precision={op_perf_overall_prec:.3f}, Recall={op_perf_overall_rec:.3f}, F1={op_perf_overall_f1:.3f}\n")

    # 2) Per-attribute counts
    per_att = []
    for t in att_tokens:
        per_att.append({
            "attr_token": t,
            "TA": TA_attr.get(t, 0),
            "FR": FR_attr.get(t, 0),
            "FA": FA_attr.get(t, 0),
            "TR": TR_attr.get(t, 0),
        })
    pd.DataFrame(per_att).to_csv(os.path.join(OUT_DIR, "mdd_per_attribute_counts.csv"), index=False)

    # 3) Per-sample detail (S/D/I + demographics)
    pd.DataFrame(sample_rows).to_csv(os.path.join(OUT_DIR, "mdd_sample_detail.csv"), index=False)

    # 4) Edit op totals
    pd.DataFrame([{"S": total_S, "D": total_D, "I": total_I}]).to_csv(
        os.path.join(OUT_DIR, "edit_ops_totals.csv"), index=False
    )
    
    # 5) Edit op performance (per operation)
    pd.DataFrame(per_op_rows).to_csv(os.path.join(OUT_DIR, "edit_ops_performance.csv"), index=False)

    print("MDD done.")
    print(f"  - {os.path.join(OUT_DIR,'mdd_summary.txt')}")
    print(f"  - {os.path.join(OUT_DIR,'mdd_per_attribute_counts.csv')}")
    print(f"  - {os.path.join(OUT_DIR,'mdd_sample_detail.csv')}")
    print(f"  - {os.path.join(OUT_DIR,'edit_ops_totals.csv')}")

if __name__ == "__main__":
    main()