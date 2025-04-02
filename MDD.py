import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
import re
from itertools import chain

# === CONFIGURATION ===
results_db_path = "/path/to/your/results.db"
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
df_map = df_map.set_index(f"Phoneme_{phonetic_alphabet}")

# Generate canonical attribute strings per group
attribute_groups = [attr for attr in allowed_attributes if attr in df_map.columns]
phoneme2attr_groupwise = {}

for ph, row in df_map.iterrows():
    groupwise = []
    for attr in attribute_groups:
        val = row[attr]
        token = f"p_{attr}" if val == 1 else f"n_{attr}"
        groupwise.append(token)
    phoneme2attr_groupwise[ph] = groupwise  # List of tokens (ordered by group)

# ========= LOAD DIPH-TO-MONOPHTHONG MAPPING =========
def load_diphthong_map(filepath):
    with open(filepath, 'r') as f:
        return dict([
            (line.split(',')[0].strip(), line.strip().split(',')[1:])
            for line in f if line.strip()
        ])

diph_map = {}
if decouple_diphthongs:
    diph_map = load_diphthong_map(diphthongs_to_monophthongs_map_file)

# === INITIALIZE METRICS ===
TA = FR = FA = TR = CD = DE = 0

# === LOAD DATA ===
dataset = load_from_disk(results_db_path)
if isinstance(dataset, dict):  # if DatasetDict
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

    # Decouple + map canonical to attributes
    canonical_attr_by_phoneme = []
    for ph in canonical:
        if decouple_diphthongs and ph in diph_map:
            for sub_ph in diph_map[ph]:
                if sub_ph in phoneme2attr_groupwise:
                    canonical_attr_by_phoneme.append(phoneme2attr_groupwise[sub_ph])
        else:
            if ph in phoneme2attr_groupwise:
                canonical_attr_by_phoneme.append(phoneme2attr_groupwise[ph])

    # Flatten canonical_attr
    canonical_attr_tokens = list(chain.from_iterable(canonical_attr_by_phoneme))

    for i in range(min(len(labels), len(pred_by_phoneme), len(spoken_attr_by_phoneme), len(canonical_attr_tokens))):
            label = labels[i]
            pred_attr = list(pred_by_phoneme[i])
            spoken_attr = list(spoken_attr_by_phoneme[i])
            canonical_attr = canonical_attr_tokens[i]

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