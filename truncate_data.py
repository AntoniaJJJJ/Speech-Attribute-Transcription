import argparse
from datasets import load_from_disk, DatasetDict

def truncate_all_splits(dataset_path, output_path, max_duration_hours=3):
    # Convert hours to seconds
    max_duration_sec = max_duration_hours * 60 * 60
    
    # Load the dataset dictionary
    dataset_dict = load_from_disk(dataset_path)
    
    # Initialize the truncated dataset dictionary
    truncated_dataset_dict = DatasetDict()
    
    # Iterate over each split in the dataset dictionary
    for split_name in dataset_dict.keys():
        data_split = dataset_dict[split_name]
        
        # Initialize variables for tracking duration
        total_duration = 0
        selected_indices = []

        # Calculate the total duration and select the subset up to 3 hours
        for i, example in enumerate(data_split):
            # Assuming 'audio' column has the 'array' and 'sampling_rate' information
            duration = len(example['audio']['array']) / example['audio']['sampling_rate']
            total_duration += duration
            
            if total_duration <= max_duration_sec:
                selected_indices.append(i)
            else:
                break

        # Select the subset of data for the split that corresponds to the specified duration
        truncated_data_split = data_split.select(selected_indices)
        
        # Add the truncated split to the dataset dictionary
        truncated_dataset_dict[split_name] = truncated_data_split
    
    # Save the truncated dataset dictionary to the specified output path
    truncated_dataset_dict.save_to_disk(output_path)
    print(f"Truncated dataset with all splits limited to {max_duration_hours} hours saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Truncate all dataset splits to a specified number of hours")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the original dataset dictionary")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the truncated dataset")
    parser.add_argument('--duration_hours', type=float, default=3, help="Maximum duration in hours for each split (default: 3)")

    args = parser.parse_args()
    
    # Run the truncate function with provided arguments
    truncate_all_splits(args.dataset_path, args.output_path, args.duration_hours)
