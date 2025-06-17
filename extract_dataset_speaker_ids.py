import sys
import os
import pandas as pd
from datasets import load_from_disk

def extract_speaker_ids_to_excel(dataset_paths, output_excel="speaker_ids.xlsx"):
    writer = pd.ExcelWriter(output_excel, engine="xlsxwriter")

    for i, path in enumerate(dataset_paths):
        if not os.path.exists(path):
            print(f"Warning: Path '{path}' does not exist. Skipping.")
            continue

        try:
            dataset_dict = load_from_disk(path)
        except Exception as e:
            print(f"Error loading dataset at {path}: {e}")
            continue

        tag = f"dataset{i+1}"
        for split in ["train", "test"]:
            if split in dataset_dict:
                speaker_ids = sorted(set(dataset_dict[split]["speaker_id"]))
                df = pd.DataFrame(speaker_ids, columns=["speaker_id"])
                sheet_name = f"{tag}_{split}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"{sheet_name}: {len(speaker_ids)} unique speaker IDs")
            else:
                print(f"{tag}_{split}: Split not found")

    writer.close()
    print(f"\nExcel file '{output_excel}' created successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_multiple_datasets_speaker_ids.py <dataset1_path> <dataset2_path> <dataset3_path>")
        sys.exit(1)

    dataset_paths = sys.argv[1:4]
    extract_speaker_ids_to_excel(dataset_paths)