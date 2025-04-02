import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
import re
from itertools import chain

# === CONFIGURATION ===
results_db_path = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_4/results_speechocean_test.db"
phoneme2att_map_file = "data/Phoneme2att_camb_att_noDiph.csv"
phonetic_alphabet = "arpa"
attribute_list_file = 'data/list_attributes-camb.txt'
diphthongs_to_monophthongs_map_file = "data/Diphthongs_en_us-arpa.csv"
decouple_diphthongs = True


# === Load allowed attributes ===
with open(attribute_list_file) as f:
    allowed_attributes = [line.strip() for line in f if line.strip()]

# === LOAD MAPPING FILE ===
df_map = pd.read_csv(phoneme2att_map_file)
phoneme_column = f"Phoneme_{phonetic_alphabet}"
df_map = df_map.set_index(phoneme_column)

# Filter allowed attributes only
attribute_groups = [attr for attr in allowed_attributes if attr in df_map.columns]

# === HELPER: Create phoneme → attribute mappers for each attribute group ===
def create_phoneme_binary_mappers(df, attribute_list):
    phoneme_binary_mappers = []
    for att in attribute_list:
        mapper = {}
        for ph, row in df.iterrows():
            ph = str(ph).lower()
            val = row.get(att)
            if pd.isna(val) or val not in [0, 1]:
                continue
            mapper[ph] = f'p_{att}' if val == 1 else f'n_{att}'
        if len(mapper) == 0:
            print(f"[WARN] Empty mapper for attribute {att}")
        phoneme_binary_mappers.append(mapper)
    return phoneme_binary_mappers

phoneme_binary_mappers = create_phoneme_binary_mappers(df_map, attribute_groups)

# ========= LOAD DIPH-TO-MONOPHTHONG MAPPING =========
def load_diphthong_map(filepath):
    with open(filepath, 'r') as f:
        return dict([line.strip().split(',', 1) for line in f if line.strip()])

def decouple_diphthongs(phoneme_seq, diph_map):
    tokens = phoneme_seq.split()
    return " ".join(" ".join(diph_map[t]) if t in diph_map else t for t in tokens)

diph_map = load_diphthong_map(diphthongs_to_monophthongs_map_file) if decouple_diphthongs else {}

# === HELPER: Map canonical phoneme sequence → groupwise attribute sequence ===
def map_phonemes_to_groupwise_attrs(phoneme_seq, phoneme_binary_mappers):
    phonemes = phoneme_seq.lower().split()
    groupwise_attrs = []
    for mapper in phoneme_binary_mappers:
        group_seq = []
        for p in phonemes:
            if p not in mapper:
                raise KeyError(f"{p}")
            group_seq.append(mapper[p])
        groupwise_attrs.append(group_seq)
    return list(zip(*groupwise_attrs))  # group-major → phoneme-major

# === INITIALIZE METRICS ===
TA = FR = FA = TR = CD = DE = 0

# === LOAD DATA ===
dataset = load_from_disk(results_db_path)
if isinstance(dataset, dict):  
    dataset = dataset["test"]

# === EVALUATION ===
for example in tqdm(dataset):
    canonical = example["phoneme_speechocean"].split()
    spoken = example["actual_spoken_phonemes"].split()
    labels = example["labels"]
    pred_str = example["pred_str"]  # List of group strings
    target_text = example["target_text"]

    # Transpose pred/target from group-major to phoneme-major
    pred_by_phoneme = list(zip(*[group.split() for group in pred_str]))
    spoken_attr_by_phoneme = list(zip(*[group.split() for group in target_text]))

    # === Decouple & map canonical ===
    canonical_seq = " ".join(canonical)
    if decouple_diphthongs:
        canonical_seq = decouple_diphthongs(canonical_seq, diph_map)

    try:
        canonical_attr_by_phoneme = map_phonemes_to_groupwise_attrs(canonical_seq, phoneme_binary_mappers)
    except KeyError as e:
        print(f"[WARN] Skipping sample due to missing mapping for: {e} | Canonical: {canonical_seq}")
    continue

    for i in range(min(len(labels), len(pred_by_phoneme), len(spoken_attr_by_phoneme), len(canonical_attr_by_phoneme))):
            label = labels[i]
            pred_attr = list(pred_by_phoneme[i])
            spoken_attr = list(spoken_attr_by_phoneme[i])
            canonical_attr = list(canonical_attr_by_phoneme[i])

            if label == 1:
                if pred_attr == canonical_attr:
                    TA += 1
                else:
                    FR += 1
            else:  # mispronounced
                if pred_attr == canonical_attr:
                    FA += 1
                else:
                    TR += 1
                    if pred_attr == spoken_attr:
                        CD += 1
                    else:
                        DE += 1


# === REPORT ===
FAR = FA / (FA + TR) if (FA + TR) > 0 else 0
FRR = FR / (FR + TA) if (FR + TA) > 0 else 0
DER = DE / (CD + DE) if (CD + DE) > 0 else 0

print("===== MDD Evaluation (Phoneme-level) =====")
print(f"TA = {TA}")
print(f"FR = {FR}")
print(f"FA = {FA}")
print(f"TR = {TR}")
print(f"  ↳ CD = {CD}")
print(f"  ↳ DE = {DE}")
print()
print(f"False Acceptance Rate (FAR): {FAR:.4f}")
print(f"False Rejection Rate (FRR): {FRR:.4f}")
print(f"Diagnostic Error Rate (DER): {DER:.4f}")