import argparse
import pandas as pd
from datasets import load_from_disk

# === Mapping Functions ===

def create_phoneme_binary_mappers(df, attribute_list, phoneme_column):
    phoneme_binary_mappers = []
    for att in attribute_list:
        mapper = {}
        for _, row in df.iterrows():
            phoneme = str(row[phoneme_column]).lower()
            val = row.get(att)
            if pd.isna(val) or val not in [0, 1]:
                continue
            mapper[phoneme] = f'p_{att}' if val == 1 else f'n_{att}'
        if len(mapper) == 0:
            print(f"[WARN] Skipping empty group for attribute: {att}")
        phoneme_binary_mappers.append(mapper)
    return phoneme_binary_mappers

def map_phonemes_to_groupwise_attrs(canonical: str, phoneme_binary_mappers: list) -> list:
    phonemes = canonical.lower().split()
    groupwise_attrs = []

    for mapper in phoneme_binary_mappers:
        group_seq = []
        for p in phonemes:
            if p not in mapper:
                raise KeyError(f"{p}")
            group_seq.append(mapper[p])
        groupwise_attrs.append(" ".join(group_seq))

    return groupwise_attrs

# === Optional Diphthong Handling ===

def load_diphthong_map():
    with open("data/Diphthongs_en_us-arpa.csv", "r") as f:
        lines = f.read().splitlines()
        return {l.split(',')[0]: l.split(',')[1:] for l in lines}

def decouple_diphthongs(phoneme_seq, diph_map):
    tokens = phoneme_seq.split()
    return " ".join(token if token not in diph_map else " ".join(diph_map[token]) for token in tokens)

# === Metric Computation ===

def compute_ta_tr_fa_fr(dataset, phoneme_binary_mappers, diphthong_map, should_decouple):
    TA, TR, FA, FR = 0, 0, 0, 0

    for sample in dataset:
        canonical = sample["phoneme_speechocean"]
        actual_attrs = sample["target_text"]
        predicted_attrs = sample["pred_str"]

        if should_decouple:
            canonical = decouple_diphthongs(canonical, diphthong_map)

        try:
            canon_attrs = map_phonemes_to_groupwise_attrs(canonical, phoneme_binary_mappers)
        except KeyError as e:
            print(f"Missing phoneme mapping for: {e}")
            continue

        if not (len(canon_attrs) == len(actual_attrs) == len(predicted_attrs)):
            print("Skipping sample due to attribute length mismatch")
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
    parser.add_argument("--attribute_list_path", type=str, required=True, help="Path to attribute list file")
    parser.add_argument("--decouple_diphthongs", type=lambda x: x.lower() == "true", default=False,
                        help="Whether to decouple diphthongs")

    args = parser.parse_args()

    phoneme_column = "Phoneme_arpa"
    ds = load_from_disk(args.data_path)
    dataset = ds["test"] if "test" in ds else ds

    # Load attribute list from file
    with open(args.attribute_list_path) as f:
        attribute_list = [line.strip() for line in f if line.strip()]

    # Load mapping chart and build mappers
    df = pd.read_csv(args.phoneme2att_path)
    phoneme_binary_mappers = create_phoneme_binary_mappers(df, attribute_list, phoneme_column)
    diphthong_map = load_diphthong_map() if args.decouple_diphthongs else {}

    # Compute metrics
    TA, TR, FA, FR = compute_ta_tr_fa_fr(dataset, phoneme_binary_mappers, diphthong_map, args.decouple_diphthongs)

    # Output
    result_df = pd.DataFrame({
        "Metric": ["True Acceptance (TA)", "True Rejection (TR)", "False Acceptance (FA)", "False Rejection (FR)"],
        "Count": [TA, TR, FA, FR]
    })

    result_df.to_excel(args.output_path, index=False)
    print("Results saved to:", args.output_path)