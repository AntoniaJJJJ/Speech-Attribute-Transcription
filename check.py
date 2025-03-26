from datasets import load_from_disk, DatasetDict

input_path = "/srv/scratch/z5369417/outputs/phonemization_speechocean_exp11_4"
output_path = "/srv/scratch/z5369417/outputs/phonemization_speechocean_training"

# Load dataset
dataset = load_from_disk(input_path)

# Remove 'labels' field
def remove_labels(example):
    if "labels" in example:
        del example["labels"]
    return example

# Apply to each split
dataset_cleaned = DatasetDict()
for split in dataset:
    dataset_cleaned[split] = dataset[split].map(remove_labels)

# Save cleaned dataset
dataset_cleaned.save_to_disk(output_path)