import argparse
from datasets import DatasetDict, load_from_disk
import evaluate
import os
from collections import defaultdict

# Load the 'wer' metric as specified
metric = evaluate.load("wer")

# Hardcoded path for the attributes list
attributes_file = "data/list_attributes-hce_Diph.txt"

def calculate_aer_by_age_for_experiment(exp_path: str):
    """
    Calculate AER by age for each attribute in a single experiment and save the output to a text file
    in the same experiment directory.

    Parameters:
    - exp_path: Path to the experiment dataset to analyze
    """
    # Load attributes
    with open(attributes_file, 'r') as f:
        attributes = [line.strip() for line in f if line.strip()]
    
    # Define the output file path within the experiment directory
    output_file = os.path.join(exp_path, "aer_by_age.txt")
    
    # Load dataset and select the "test" split
    dataset_dict = load_from_disk(exp_path)
    if "test" not in dataset_dict:
        raise ValueError("The specified dataset does not contain a 'test' split.")
    dataset = dataset_dict["test"]  # Extract the 'test' split

    # Group entries by age
    age_groups = defaultdict(list)
    for entry in dataset:
        age_groups[entry["age"]].append(entry)

    # Open output file for writing results
    with open(output_file, 'w') as f:
        f.write("Age Group AER Results\n")
        f.write("=====================\n")

        # Calculate AER for each age group
        for age, entries in age_groups.items():
            age_aer_results = {}
            
            # Process each attribute group
            for attr_idx, attr in enumerate(attributes):
                # Collect predictions and targets for this attribute across the age group
                preds = [item["pred_str"][attr_idx] for item in entries]
                targets = [item["target_text"][attr_idx] for item in entries]

                # Debugging output to inspect the values
                print(f"\n--- Debug: Age {age}, Attribute {attr} ---")
                print("Predictions (pred_str):", preds)
                print("Targets (target_text):", targets)
                
                # Calculate AER using metric.compute
                wer = metric.compute(predictions=preds, references=targets)
                aer = 1 - wer  # Convert WER to AER
                age_aer_results[attr] = aer

            # Write AER results for this age group to the file
            f.write(f"\nAge: {age}\n")
            for attr, aer in age_aer_results.items():
                f.write(f"{attr}: {aer:.5f}\n")

    print(f"AER results saved to {output_file}")

# Set up argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate AER by age for a specific experiment.")
    parser.add_argument("exp_path", type=str, help="Path to the experiment dataset (e.g., path to Exp1)")

    args = parser.parse_args()

    # Run the function with the provided experiment path
    calculate_aer_by_age_for_experiment(args.exp_path)