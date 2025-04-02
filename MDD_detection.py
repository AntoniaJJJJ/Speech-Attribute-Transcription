import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from itertools import chain

# === CONFIGURATION ===
results_db_path = "/srv/scratch/z5369417/outputs/trained_result/cu/exp11/exp11_4/results_speechocean_test.db"
phoneme2att_map_file = "data/Phoneme2att_camb_att_noDiph.csv"
phonetic_alphabet = "arpa"
attribute_list_file = 'data/list_attributes-camb.txt'
diphthongs_to_monophthongs_map_file = "data/Diphthongs_en_us-arpa.csv"
decouple_diphthongs = True

# === Mapping Functions ===
def load_diphthong_map(filepath):
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        return {l.split(',')[0]: l.split(',')[1:] for l in lines}

# Replace diphthongs in a sequence with corresponding monophthongs
def decouple_diphthongs(phoneme_seq, diph_map):
    tokens = phoneme_seq.split()
    return " ".join(token if token not in diph_map else " ".join(diph_map[token]) for token in tokens)

# Create a mapper from phoneme to binary attributes for each attribute group
def create_phoneme_binary_mappers(df, attribute_list, phoneme_column):
    phoneme_binary_mappers = []
    for att in attribute_list:
        mapper = {}
        for _, row in df.iterrows():
            phoneme = str(row[phoneme_column]).lower()
            val = row.get(att)
            if pd.isna(val) or val not in [0, 1]:
                continue
            tag = f'p_{att}' if val == 1 else f'n_{att}'
            mapper[phoneme] = tag
        if len(mapper) == 0:
            print(f"[WARN] Skipping empty group for attribute: {att}")
        phoneme_binary_mappers.append(mapper)
    return phoneme_binary_mappers

# Map a canonical phoneme sequence to a list of attribute labels (groupwise)
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

# === INITIALIZE ===
with open(attribute_list_file) as f:
    attribute_list = [line.strip() for line in f if line.strip()]

# === LOAD MAPPING FILE ===
df_map = pd.read_csv(phoneme2att_map_file)
phoneme_column = f"Phoneme_{phonetic_alphabet}"
phoneme_binary_mappers = create_phoneme_binary_mappers(df_map, attribute_list, phoneme_column)

# === LOAD DIPH-TO-MONOPHTHONG MAPPING ===
diphthong_map = load_diphthong_map(diphthongs_to_monophthongs_map_file) if decouple_diphthongs else {}

# === LOAD DATA ===
dataset = load_from_disk(results_db_path)
if isinstance(dataset, dict):
    dataset = dataset["test"]

# === INITIALIZE METRICS ===
TA = FR = FA = TR = CD = DE = 0
TA_attr = {}
FR_attr = {}
FA_attr = {}
TR_attr = {}

# Create full attribute token list
attribute_tokens = [f'p_{att}' for att in attribute_list] + [f'n_{att}' for att in attribute_list]
TA_attr = {att: 0 for att in attribute_tokens}
FR_attr = {att: 0 for att in attribute_tokens}
FA_attr = {att: 0 for att in attribute_tokens}
TR_attr = {att: 0 for att in attribute_tokens}

# === EVALUATION ===
for example in tqdm(dataset):
    canonical = example["phoneme_speechocean"].split()
    spoken_attr_by_phoneme = list(zip(*[g.split() for g in example["target_text"]]))
    pred_by_phoneme = list(zip(*[g.split() for g in example["pred_str"]]))
    labels = example["labels"]

    canonical_seq = " ".join(canonical)
    if decouple_diphthongs:
        canonical_seq = decouple_diphthongs(canonical_seq, diphthong_map)

    try:
        canonical_attr_by_phoneme = map_phonemes_to_groupwise_attrs(canonical_seq, phoneme_binary_mappers)
    except KeyError as e:
        print(f"[WARN] Skipping sample due to missing mapping for: {e}")
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
        else:
            if pred_attr == canonical_attr:
                FA += 1
            else:
                TR += 1
                if pred_attr == spoken_attr:
                    CD += 1
                else:
                    DE += 1

        # Per-attribute counting
        for attr_pred, attr_canon in zip(pred_attr, canonical_attr):
            if label == 1:
                if attr_pred == attr_canon:
                    TA_attr[attr_canon] += 1
                else:
                    FR_attr[attr_canon] += 1
            else:
                if attr_pred == attr_canon:
                    FA_attr[attr_canon] += 1
                else:
                    TR_attr[attr_canon] += 1

FAR = FA / (FA + TR) if (FA + TR) > 0 else 0
FRR = FR / (FR + TA) if (FR + TA) > 0 else 0
DER = DE / (CD + DE) if (CD + DE) > 0 else 0

print("===== MDD Evaluation (Phoneme-level) =====")
print(f"TA = {TA}")
print(f"FR = {FR}")
print(f"FA = {FA}")
print(f"TR = {TR}")
print(f"  → CD = {CD}")
print(f"  → DE = {DE}")
print()
print(f"False Acceptance Rate (FAR): {FAR:.4f}")
print(f"False Rejection Rate (FRR): {FRR:.4f}")
print(f"Diagnostic Error Rate (DER): {DER:.4f}")

# === PER-ATTRIBUTE REPORT ===
print("\n===== Per-Attribute Counts =====")
print("Attribute\tTA\tFR\tFA\tTR")
for att in attribute_tokens:
    print(f"{att}\t{TA_attr[att]}\t{FR_attr[att]}\t{FA_attr[att]}\t{TR_attr[att]}")