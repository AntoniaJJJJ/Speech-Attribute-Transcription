import argparse
import pandas as pd
from datasets import load_from_disk

phonetic_alphabet = "arpa"
phoneme_column = f"Phoneme_{phonetic_alphabet}"
diphthong_map_file = "data/Diphthongs_en_us-arpa.csv"

# Load Mapping Charts
def load_phoneme_to_attribute_mapping(phoneme2att_path, decouple_diphthongs):
    phoneme2att_df = pd.read_csv(phoneme2att_path)
    phoneme2att_map = {
        row[phoneme_column].lower(): row["Attributes"].split()
        for _, row in phoneme2att_df.iterrows()
    }

    diphthong_map = {}
    if decouple_diphthongs:
        diph_df = pd.read_csv(diphthong_map_file)
        for _, row in diph_df.iterrows():
            diph = row["Diphthong"].lower()
            mono_seq = row["Monophthong Sequence"].lower().split()
            diphthong_map[diph] = mono_seq

    return phoneme2att_map, diphthong_map

def phonemes_to_attributes(phoneme_string, phoneme2att_map, diphthong_map, decouple_diphthongs):
    phonemes = phoneme_string.strip().split()
    attr_sequence = []

    for ph in phonemes:
        ph = ph.lower()
        if decouple_diphthongs and ph in diphthong_map:
            for mono in diphthong_map[ph]:
                if mono in phoneme2att_map:
                    attr_sequence.extend(phoneme2att_map[mono])
        elif ph in phoneme2att_map:
            attr_sequence.extend(phoneme2att_map[ph])
        else:
            print(f" Warning: phoneme '{ph}' not in mapping chart — skipped.")

    return attr_sequence

# === MDD Logic ===
def compute_ta_tr_fa_fr(dataset, phoneme2att_map, diphthong_map, decouple_diphthongs):
    TA, TR, FA, FR = 0, 0, 0, 0

    for sample in dataset:
        canonical = sample["phoneme_speechocean"]
        actual_attrs = sample["target_text"]
        predicted_attrs = sample["pred_str"]

        if not canonical or not actual_attrs or not predicted_attrs:
            continue

        canonical_attrs = phonemes_to_attributes(
            canonical, phoneme2att_map, diphthong_map, decouple_diphthongs
        )

        for can, act, pred in zip(canonical_attrs, actual_attrs, predicted_attrs):
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

    ds = load_from_disk(args.data_path)
    dataset = ds["test"] if "test" in ds else ds

    phoneme2att_map, diphthong_map = load_phoneme_to_attribute_mapping(
        args.phoneme2att_path, args.decouple_diphthongs
    )

    TA, TR, FA, FR = compute_ta_tr_fa_fr(dataset, phoneme2att_map, diphthong_map, args.decouple_diphthongs)

    result_df = pd.DataFrame({
        "Metric": ["True Acceptance (TA)", "True Rejection (TR)", "False Acceptance (FA)", "False Rejection (FR)"],
        "Count": [TA, TR, FA, FR]
    })

    result_df.to_excel(args.output_path, index=False)
    print("Results saved to:", args.output_path)