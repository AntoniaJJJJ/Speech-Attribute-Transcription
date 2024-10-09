from datasets import load_from_disk

# Path to the full dataset directory (not just 'train')
output_dataset_path = "/srv/scratch/z5369417/created_dataset_0808/AKT_dataset"  # Adjust path to the parent folder

# Load the dataset (this will load the 'train' and any other splits, if available)
dataset = load_from_disk(output_dataset_path)


# Iterate over all examples and print them
for example in dataset:
    print(example)