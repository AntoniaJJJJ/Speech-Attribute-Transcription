import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Levenshtein
from datasets import load_from_disk

sns.set(style="whitegrid", font_scale=1.2)

# ==================== CONFIG ====================
CONFIG = {
    "CU Model": {
        "RESULTS_DB": "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_5/results_unsw_test.db",
        "MDD_DETAIL": "/srv/scratch/z5369417/outputs/mdd_unsw_phoneme_level_exp11/mdd_sample_detail.csv",
    },
    "Combined Model": {
        "RESULTS_DB": "/srv/scratch/z5369417/outputs/trained_result/CU_AKT_combined/exp22/exp22_2/results_unsw_test.db",
        "MDD_DETAIL": "/srv/scratch/z5369417/outputs/mdd_unsw_phoneme_level_exp22/mdd_sample_detail.csv",
    },
}
UNSW_META = "/srv/scratch/z5369417/UNSW_final_deliverables/CAAP_2023-04-27/dataset_spreadsheet.xlsx"
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_unsw_demographic_compare_split"
os.makedirs(OUT_DIR, exist_ok=True)


# ==================== HELPERS ====================
def align_lists(ref, hyp):
    vocab = list(set(ref + hyp))
    char_map = {p: chr(i + 33) for i, p in enumerate(vocab)}
    ref_str = "".join(char_map[p] for p in ref)
    hyp_str = "".join(char_map[p] for p in hyp)
    ops = Levenshtein.editops(ref_str, hyp_str)
    aligned, r_i, h_i = [], 0, 0
    for op, i1, i2 in ops:
        while r_i < i1 and h_i < i2:
            aligned.append((ref[r_i], hyp[h_i])); r_i += 1; h_i += 1
        if op == "insert":
            aligned.append(("", hyp[h_i])); h_i += 1
        elif op == "delete":
            aligned.append((ref[r_i], "")); r_i += 1
        else:
            aligned.append((ref[r_i], hyp[h_i])); r_i += 1; h_i += 1
    while r_i < len(ref):
        aligned.append((ref[r_i], "")); r_i += 1
    while h_i < len(hyp):
        aligned.append(("", hyp[h_i])); h_i += 1
    return aligned


def aggregate_demographic(df, group_vars):
    """Aggregate MDD metrics and compute DER."""
    metrics = ["TA", "FR", "FA", "TR", "CD", "DE"]
    out = df.groupby(group_vars)[metrics].sum().reset_index()
    out["DER"] = out["DE"] / (out["CD"] + out["DE"] + 1e-8)
    return out


# ==================== LOAD META ====================
meta = pd.read_excel(UNSW_META)
meta.rename(columns={
    "word_phonemes": "phoneme_unsw",
    "recording_phonemes": "actual_spoken_phonemes",
    "speech_status": "speech_status",
    "audio_file": "audio_file",
    "word": "word"
}, inplace=True)

meta["age_year"] = (meta["age"].astype(float) / 12).round(0).astype(int)
meta["gender"] = meta["gender"].astype(str)
meta["speech_status"] = meta["speech_status"].fillna(-1).astype(int)
meta = meta[~meta["gender"].isin(["11", "11.0"])]

# ==================== MAIN LOOP ====================
records_all = []

for model_name, paths in CONFIG.items():
    print(f"Processing {model_name}")

    df_mdd = pd.read_csv(paths["MDD_DETAIL"])
    df = df_mdd.merge(meta[["word", "age_year", "gender", "speech_status"]],
                      on="word", how="left")

    expanded = []
    for _, s in df.iterrows():
        can = str(s.get("canonical", "")).split()
        spo = str(s.get("spoken", "")).split()
        pred = str(s.get("predicted", "")).split()
        align_truth = align_lists(can, spo)
        align_pred  = align_lists(can, pred)

        for (c, sp), (_, pr) in zip(align_truth, align_pred):
            r = {"canonical": c, "spoken": sp, "predicted": pr,
                 "TA":0,"FR":0,"FA":0,"TR":0,"CD":0,"DE":0}
            # Classification
            if c == "" and sp != "":
                r["TR"]=1; r["CD" if pr==sp else "DE"]=1
            elif c != "" and sp == "":
                r["TR"]=1; r["CD" if pr=="" else "DE"]=1
            elif c == sp:
                if pr==c: r["TA"]=1
                else: r["FR"]=1
            else:
                if pr==c: r["FA"]=1
                else: r["TR"]=1; r["CD" if pr==sp else "DE"]=1

            r.update({
                "word": s["word"],
                "age_year": s["age_year"],
                "gender": s["gender"],
                "speech_status": s["speech_status"],
                "model": model_name
            })
            expanded.append(r)

    df_exp = pd.DataFrame(expanded)
    records_all.append(df_exp)

df_all = pd.concat(records_all, ignore_index=True)

# ==================== AGE-GROUP SPLITTING ====================
# speech_status: 1 = SSD, 0 = TD
split_groups = {
    "All Speakers": df_all,
    "SSD Speakers": df_all[df_all["speech_status"] == 1],
    "TD Speakers": df_all[df_all["speech_status"] == 0],
}

records = []
for group_name, subset in split_groups.items():
    if subset.empty:
        continue
    df_demo = aggregate_demographic(subset, ["model", "age_year"])
    df_demo["Group"] = group_name
    records.append(df_demo)
df_age_all = pd.concat(records, ignore_index=True)

# ==================== PLOT AGE (6 LINES) -- FINAL POLISHED ====================
fig, ax = plt.subplots(figsize=(10.5, 6))  # widened figure
sns.lineplot(
    data=df_age_all,
    x="age_year", y="DER",
    hue="Group", style="model",
    style_order=["CU Model", "Combined Model"],  # consistent styling
    dashes={"CU Model": "", "Combined Model": (3, 2)},  # CU solid, Combined dashed
    markers=True, ax=ax
)

ax.set_title("Diagnostic Error Rate (DER) by Age (PEDZSTAR)", pad=10)
ax.set_xlabel("Age (years)")
ax.set_ylabel("DER")

leg = ax.legend_
if leg is not None:
    leg.set_title(None)
    for txt in leg.get_texts():
        if txt.get_text().strip().lower() == "model":
            txt.set_text("Model")
    leg._legend_box.align = "left"
    leg.set_bbox_to_anchor((1.05, 0.5))   # slightly closer than before
    leg.set_loc("center left")
    leg.borderpad = 0.4

fig.subplots_adjust(right=0.85)  # leave sufficient margin for legend
sns.despine()
fig.savefig(os.path.join(OUT_DIR, "PS_DER_by_age_split_final.png"), dpi=300)
plt.close(fig)


# ==================== PLOT GENDER (2 BARS) -- FINAL POLISHED ====================
df_gender = aggregate_demographic(df_all, ["model", "gender"])

fig, ax = plt.subplots(figsize=(7.5, 5))  # slightly wider
sns.barplot(
    data=df_gender,
    x="gender", y="DER", hue="model",
    errorbar=None, ax=ax
)

ax.set_title("Diagnostic Error Rate (DER) by Gender (PEDZSTAR)", pad=10)
ax.set_xlabel("Gender")
ax.set_ylabel("DER")

leg = ax.legend(title="Model")
leg.set_bbox_to_anchor((1.05, 0.5))
leg.set_loc("center left")
leg.borderpad = 0.4

fig.subplots_adjust(right=0.85)
sns.despine()
fig.savefig(os.path.join(OUT_DIR, "PS_DER_by_gender_final.png"), dpi=300)
plt.close(fig)
