from datasets import load_from_disk, DatasetDict

# Define paths
input_path = "/srv/scratch/z5369417/outputs/phonemization_speechocean_exp11_4"
output_path = "/srv/scratch/z5369417/outputs/phonemization_speechocean_training/"

# Load dataset
ds = load_from_disk(input_path)

# Remove 'labels' field from each split
ds_cleaned = DatasetDict({
    split: ds[split].remove_columns("labels")
    for split in ds
})

# Save to new location
ds_cleaned.save_to_disk(output_path)
print("New training dataset saved to:", output_path)