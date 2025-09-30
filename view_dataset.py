import os
import sys
from datasets import load_from_disk, Dataset, DatasetDict

def main(input_dataset_path, write_to_file, output_file_path=None):
    try:
        dataset = load_from_disk(input_dataset_path)

        # If it's a DatasetDict (train/test), handle each split separately
        if isinstance(dataset, DatasetDict):
            splits = dataset.keys()
        else:
            splits = [None]  # single dataset, no splits

        if write_to_file.lower() == 'true':
            if output_file_path is None:
                raise ValueError("Output file path must be provided when write_to_file is True.")

            with open(output_file_path, 'w') as out_file:
                out_file.write("FULL DATASET:\n")

                for split in splits:
                    current_ds = dataset[split] if split else dataset
                    out_file.write(f"\n===== SPLIT: {split or 'full'} =====\n")
                    for idx, example in enumerate(current_ds):
                        out_file.write(f"Example {idx}:\n{example}\n\n")

            print(f"Data successfully written to {output_file_path}")

        else:
            for split in splits:
                current_ds = dataset[split] if split else dataset
                print(f"\n===== SPLIT: {split or 'full'} =====\n")
                for idx, example in enumerate(current_ds):
                    print(f"Example {idx}:\n{example}\n")
                    if idx == 19:  # only show first 20
                        break

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_dataset_path> <write_to_file: True|False> [output_file_path]")
        sys.exit(1)

    input_dataset_path = sys.argv[1]
    write_to_file = sys.argv[2]
    output_file_path = sys.argv[3] if len(sys.argv) > 3 else None

    main(input_dataset_path, write_to_file, output_file_path)
