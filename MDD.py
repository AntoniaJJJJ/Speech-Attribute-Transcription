import argparse
import pandas as pd
from datasets import load_from_disk

# Load Mapping Charts
def load_mappings(phoneme2att_map_path):
    df = pd.read_csv(phoneme2att_map_path)
    attributes = [col for col in df.columns if col.startswith('p_') or col.startswith('n_') or col == 'Attributes']
    phoneme_column = [col for col in df.columns if col.startswith('Phoneme_')][0]
    list_att = list(set(attr[2:] for attr in attributes if '_' in attr))
    phoneme2att_map = {}
    for _, row in df.iterrows():
        phoneme = row[phoneme_column].lower()
        phoneme2att_map[phoneme] = []
        for att in list_att:
            if att in row:
                tag = 'p_' + att if row[att] == 1 else 'n_' + att
                phoneme2att_map[phoneme].append(tag)
    return phoneme2att_map, list_att

def load_diphthong_map():
    with open("data/Diphthongs_en_us-arpa.csv", "r") as f:
        lines = f.read().splitlines()
        return {l.split(',')[0]: l.split(',')[1:] for l in lines}

def decouple_diphthongs(phoneme_seq, diph_map):
    tokens = phoneme_seq.split()
    return " ".join(token if token not in diph_map else " ".join(diph_map[token]) for token in tokens)

# === MDD Logic ===
def compute_ta_tr_fa_fr(dataset, phoneme2att_map, diphthong_map, should_decouple):
    TA, TR, FA, FR = 0, 0, 0, 0

    for sample in dataset:
        canonical = sample["phoneme_speechocean"]
        actual_attrs = sample["target_text"]
        predicted_attrs = sample["pred_str"]

        if should_decouple:
            canonical = decouple_diphthongs(canonical, diphthong_map)

        try:
            canon_attrs = [" ".join(phoneme2att_map[p]) for p in canonical.split()]
        except KeyError as e:
            print(f"Missing phoneme mapping for: {e}")
            continue

        for can, act, pred in zip(canon_attrs, actual_attrs, predicted_attrs):
            if act == can and pred == act:
                TA += 1
            elif act != can and pred == act:
                TR += 1
            elif act == can and pred != act:
                FR += 1
            elif act != can and pred != act and pred == can:
                FA += 1

    return TA, TR, FA, FR

# === Main Script ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mispronunciation Detection Metrics")
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluated .db dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Output Excel file path")
    parser.add_argument("--phoneme2att_path", type=str, required=True, help="Path to phoneme2att CSV")
    parser.add_argument("--decouple_diphthongs", type=lambda x: x.lower() == "true", default=False,
                    help="Whether to decouple diphthongs")
    args = parser.parse_args()

    phonetic_alphabet = "arpa"
    phoneme_column = f"Phoneme_{phonetic_alphabet}"
    diphthong_map_file = "data/Diphthongs_en_us-arpa.csv"

    ds = load_from_disk(args.data_path)
    dataset = ds["test"] if "test" in ds else ds

    phoneme2att_map, _ = load_mappings(args.phoneme2att_path)
    diphthong_map = load_diphthong_map() if args.decouple_diphthongs else {}

    TA, TR, FA, FR = compute_ta_tr_fa_fr(dataset, phoneme2att_map, diphthong_map, args.decouple_diphthongs)

    result_df = pd.DataFrame({
        "Metric": ["True Acceptance (TA)", "True Rejection (TR)", "False Acceptance (FA)", "False Rejection (FR)"],
        "Count": [TA, TR, FA, FR]
    })

    result_df.to_excel(args.output_path, index=False)
    print("Results saved to:", args.output_path)