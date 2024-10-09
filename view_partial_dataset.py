from datasets import load_from_disk, concatenate_datasets
import os

# Path to the folder containing all partial datasets
output_dataset_path = "/srv/scratch/z5369417/created_dataset_0808/AKT_dataset"

# Function to load all partial dataset batches and concatenate them
def load_partial_dataset(output_dir):
    # List all directories (batches) inside the output folder
    batch_dirs = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    
    # Load each batch and append to a list
    all_datasets = []
    for batch_dir in batch_dirs:
        batch_path = os.path.join(output_dir, batch_dir)
        dataset = load_from_disk(batch_path)
        all_datasets.append(dataset)
    
    # Concatenate all datasets into a single dataset
    if all_datasets:
        full_dataset = concatenate_datasets(all_datasets)
        return full_dataset
    else:
        print("No datasets found in the specified directory.")
        return None

# Load all partial datasets
dataset = load_partial_dataset(output_dataset_path)

# Iterate over all examples and print them
for example in dataset:
    print(example)