from datasets import load_from_disk
import os
import argparse

def truncate_dataset(dataset_path, output_path, max_duration_hours=3):
    # Convert hours to seconds
    max_duration_sec = max_duration_hours * 60 * 60
    
    # Load the dataset
    data = load_from_disk(dataset_path)
    
    # Initialize variables for tracking duration
    total_duration = 0
    selected_indices = []

    # Calculate the total duration and select the subset for 3 hours
    for i, example in enumerate(data):
        # Assuming 'audio' column has the 'array' and 'sampling_rate' information
        duration = len(example['audio']['array']) / example['audio']['sampling_rate']
        total_duration += duration
        
        if total_duration <= max_duration_sec:
            selected_indices.append(i)
        else:
            break

    # Select the subset of data that corresponds to the specified duration
    truncated_data = data.select(selected_indices)

    # Save the truncated dataset to disk
    truncated_data.save_to_disk(output_path)
    print(f"Truncated dataset with {max_duration_hours} hours of data saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Truncate a dataset to a specified number of hours")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the original dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the truncated dataset")
    parser.add_argument('--duration_hours', type=float, default=3, help="Maximum duration in hours (default: 3)")

    args = parser.parse_args()
    
    # Run the truncate function with provided arguments
    truncate_dataset(args.dataset_path, args.output_path, args.duration_hours)

