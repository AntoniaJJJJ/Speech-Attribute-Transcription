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
        "RESULTS_DB": "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_4/results_speechocean_test.db",
        "MDD_DETAIL": "/srv/scratch/z5369417/outputs/mdd_speechocean_phoneme_level_exp11/mdd_sample_detail.csv",
    },
    "Combined Model": {
        "RESULTS_DB": "/srv/scratch/z5369417/outputs/trained_result/CU_AKT_combined/exp22/exp22_3/results_speechocean_test.db",
        "MDD_DETAIL": "/srv/scratch/z5369417/outputs/mdd_speechocean_phoneme_level_exp22/mdd_sample_detail.csv",
    },
}
OUT_DIR = "/srv/scratch/z5369417/outputs/mdd_speechocean_demographic_compare"
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
    metrics = ["TA", "FR", "FA", "TR", "CD", "DE"]
    out = df.groupby(group_vars)[metrics].sum().reset_index()
    out["DER"] = out["DE"] / (out["CD"] + out["DE"] + 1e-8)
    return out

# ==================== MAIN LOOP ====================
records_all = []

for model_name, paths in CONFIG.items():
    print(f"Processing {model_name}")
    df_mdd = pd.read_csv(paths["MDD_DETAIL"])
    dataset = load_from_disk(paths["RESULTS_DB"])
    df_pred = dataset["test"].to_pandas() if "test" in dataset else dataset.to_pandas()

    # Merge demographics
    df_meta = df_pred[["text", "speaker", "age", "gender"]].drop_duplicates()
    df_meta = df_meta.rename(columns={"text": "word"})
    df_meta["age_year"] = df_meta["age"].fillna(-1).astype(int)
    df_meta["gender"] = df_meta["gender"].fillna("unknown")

    # Join MDD output with demographics
    df = df_mdd.merge(df_meta[["word", "age_year", "gender"]], on="word", how="left")

    # Expand to phoneme-level
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
                "model": model_name
            })
            expanded.append(r)

    df_exp = pd.DataFrame(expanded)
    records_all.append(df_exp)

df_all = pd.concat(records_all, ignore_index=True)

# ==================== DER BY AGE ====================
df_age = aggregate_demographic(df_all, ["model", "age_year"])
df_age = df_age.sort_values(["model", "age_year"])

fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(
    data=df_age,
    x="age_year", y="DER",
    hue="model", style="model",
    dashes={"CU Model": "", "Combined Model": (3, 2)},
    markers=True, ax=ax
)

ax.set_title("Diagnostic Error Rate (DER) by Age (SpeechOcean)", pad=10)
ax.set_xlabel("Age (years)")
ax.set_ylabel("DER")

leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderpad=0.4)
leg.set_title("Model")

sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "SO_DER_by_age_final.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ==================== DER BY GENDER ====================
df_gender = aggregate_demographic(df_all, ["model", "gender"])

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=df_gender,
    x="gender", y="DER", hue="model",
    errorbar=None, ax=ax
)

ax.set_title("Diagnostic Error Rate (DER) by Gender (SpeechOcean)", pad=10)
ax.set_xlabel("Gender")
ax.set_ylabel("DER")

leg = ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.01, 0.5), borderpad=0.4)

sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "SO_DER_by_gender_final.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
