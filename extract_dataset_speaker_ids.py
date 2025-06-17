import sys
import os
import pandas as pd
from datasets import load_from_disk

def extract_speaker_ids_to_excel(dataset_paths, output_excel="speaker_ids.xlsx"):
    writer = pd.ExcelWriter(output_excel, engine="xlsxwriter")

    for i, path in enumerate(dataset_paths):
        tag = f"dataset{i+1}"

        if not os.path.exists(path):
            print(f"Path '{path}' does not exist. Skipping.")
            continue

        try:
            dataset_dict = load_from_disk(path)
        except Exception as e:
            print(f"Error loading dataset at {path}: {e}")
            continue

        print(f"Processing {tag} from: {path}")
        print(f"Splits found: {list(dataset_dict.keys())}")

        for split in dataset_dict.keys():
            dataset = dataset_dict[split]

            if "speaker_id" not in dataset.column_names:
                print(f"'{split}' split in {tag} does not contain 'speaker_id'. Skipping.")
                continue

            speaker_ids = sorted(set(dataset["speaker_id"]))
            if not speaker_ids:
                print(f"No speaker IDs found in {tag}_{split}. Skipping.")
                continue

            df = pd.DataFrame(speaker_ids, columns=["speaker_id"])
            sheet_name = f"{tag}_{split}"[:31]  # Excel sheet name max length = 31
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"{sheet_name}: {len(speaker_ids)} unique speaker IDs written.")

    writer.close()
    print(f"Excel file '{output_excel}' created.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_multiple_datasets_speaker_ids.py <dataset1_path> <dataset2_path> <dataset3_path>")
        sys.exit(1)

    dataset_paths = sys.argv[1:4]
    extract_speaker_ids_to_excel(dataset_paths)
