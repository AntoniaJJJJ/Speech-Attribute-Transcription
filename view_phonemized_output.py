from datasets import load_from_disk

# Path to the processed dataset
output_dataset_path = "/srv/scratch/z5369417/created_dataset_3009/AKT_exclude_noise/train"

# Load the dataset
dataset = load_from_disk(output_dataset_path)

# Iterate over all examples and print them
for example in dataset:
    print(example)