import argparse
from datasets import load_from_disk, DatasetDict

def truncate_connected_splits(dataset_path, output_path, max_duration_hours=3):
    max_duration_sec = max_duration_hours * 60 * 60
    dataset_dict = load_from_disk(dataset_path)
    truncated_dataset_dict = DatasetDict()

    for split_name in dataset_dict.keys():
        data_split = dataset_dict[split_name]
        total_duration = 0
        selected_indices = []

        # Calculate the total duration and select connected sentences up to 3 hours
        for i, example in enumerate(data_split):
            text = example['text']
            # Check if the 'text' field contains more than 10 words to determine if it's connected
            if len(text.split()) > 10:
                duration = len(example['audio']['array']) / example['audio']['sampling_rate']
                total_duration += duration
                
                if total_duration <= max_duration_sec:
                    selected_indices.append(i)
                else:
                    break

        truncated_data_split = data_split.select(selected_indices)
        truncated_dataset_dict[split_name] = truncated_data_split

    truncated_dataset_dict.save_to_disk(output_path)
    print(f"Truncated dataset with connected sentences, limited to {max_duration_hours} hours, saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate dataset splits to 3 hours with connected sentences")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the original dataset dictionary")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the truncated dataset")
    parser.add_argument('--duration_hours', type=float, default=3, help="Maximum duration in hours for each split (default: 3)")

    args = parser.parse_args()
    truncate_connected_splits(args.dataset_path, args.output_path, args.duration_hours)