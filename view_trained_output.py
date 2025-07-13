import os
import sys
from datasets import load_from_disk

def main(input_dataset_path, write_to_file, output_file_path=None):
    try:
        dataset = load_from_disk(input_dataset_path)

        if write_to_file.lower() == 'true':
            if output_file_path is None:
                raise ValueError("Output file path must be provided when write_to_file is True.")

            with open(output_file_path, 'w') as out_file:
                out_file.write("FULL DATASET:\n")
                for idx, example in enumerate(dataset):
                    out_file.write(f"Example {idx}:\n{example}\n\n")

            print(f"Data successfully written to {output_file_path}")

        else:
            print("Displaying the first 20 rows of the dataset:\n")
            for idx, example in enumerate(dataset):
                print(f"Example {idx}:\n{example}\n")
                if idx == 19:
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
