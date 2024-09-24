import argparse
from datasets import load_from_disk, DatasetDict

def truncate_all_splits(dataset_path, output_path, max_duration_hours=3, min_words=10):
    # Convert hours to seconds
    max_duration_sec = max_duration_hours * 60 * 60
    
    # Load the dataset dictionary
    dataset_dict = load_from_disk(dataset_path)
    
    # Initialize the truncated dataset dictionary
    truncated_dataset_dict = DatasetDict()
    
    # Iterate over each split in the dataset dictionary
    for split_name in dataset_dict.keys():
        data_split = dataset_dict[split_name]
        
        # First pass: select samples with more than the initial min_words
        total_duration = 0
        selected_indices = []

        for i, example in enumerate(data_split):
            word_count = len(example['text'].split())
            if word_count > min_words:
                duration = len(example['audio']['array']) / example['audio']['sampling_rate']
                if total_duration + duration <= max_duration_sec:
                    total_duration += duration
                    selected_indices.append(i)
                else:
                    break
        
        # Check if the total duration is less than the max_duration_sec
        if total_duration < max_duration_sec:
            print(f"Second pass needed for split '{split_name}' - current duration: {total_duration / 3600:.2f} hours")
            
            # Second pass: gradually decrease the min_words requirement
            while total_duration < max_duration_sec and min_words > 0:
                min_words -= 1  # Reduce word count threshold
                for i, example in enumerate(data_split):
                    if i not in selected_indices:
                        word_count = len(example['text'].split())
                        if word_count > min_words:
                            duration = len(example['audio']['array']) / example['audio']['sampling_rate']
                            if total_duration + duration <= max_duration_sec:
                                total_duration += duration
                                selected_indices.append(i)
                            else:
                                break
                if min_words == 0:
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
    parser = argparse.ArgumentParser(description="Truncate all dataset splits to a specified number of hours with connected sentences")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the original dataset dictionary")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the truncated dataset")
    parser.add_argument('--duration_hours', type=float, default=3, help="Maximum duration in hours for each split (default: 3)")
    parser.add_argument('--min_words', type=int, default=10, help="Minimum word count to consider a sample connected (default: 10)")

    args = parser.parse_args()
    
    # Run the truncate function with provided arguments
    truncate_all_splits(args.dataset_path, args.output_path, args.duration_hours, args.min_words)