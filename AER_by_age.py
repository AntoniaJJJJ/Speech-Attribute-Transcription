import argparse
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd
import evaluate
import os

# Load the 'wer' metric as specified
metric = evaluate.load("wer")

# Hardcoded path for the attributes list
attributes_file = "data/list_attributes-hce_Diph.txt"

def calculate_aer_by_age_for_experiment(exp_path: str):
    """
    Calculate AER by age for each attribute in a single experiment and save the output to a text file
    in the same experiment directory.

    Parameters:
    - exp_path: Path to the experiment dataset to analyze.
    """
    # Load attributes
    with open(attributes_file, 'r') as f:
        attributes = [line.strip() for line in f if line.strip()]
    
    # Define the output file path within the experiment directory
    output_file = os.path.join(exp_path, "aer_by_age.txt")
    
    # Load dataset
    dataset = load_from_disk(exp_path)["test"]
    age_groups = dataset.groupby("age")  # Group by age

    # Open output file for writing results
    with open(output_file, 'w') as f:
        f.write("Age Group AER Results\n")
        f.write("=====================\n")

        # Calculate AER for each age group
        for age, entries in age_groups:
            age_aer_results = {}
            
            # Process each attribute group
            for attr_idx, attr in enumerate(attributes):
                # Collect predictions and targets for this attribute across the age group
                preds = [item["pred_str"][attr_idx] for item in entries]
                targets = [item["target_text"][attr_idx] for item in entries]
                
                # Calculate AER using metric.compute
                accuracy = metric.compute(predictions=preds, references=targets)["accuracy"]
                aer = 1 - accuracy  # Convert accuracy to AER
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