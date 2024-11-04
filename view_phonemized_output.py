from datasets import load_from_disk

# Path to the processed dataset
output_dataset_path = "/srv/scratch/z5369417/created_dataset_3009/AKT_exclude_noise/test"

# Load the dataset
dataset = load_from_disk(output_dataset_path)

# Iterate over all examples and print them
for example in dataset:
    print(example)

# Iterate over the first 2000 examples and print them
#for i, example in enumerate(dataset):
    #if i >= 2000:
      #  break
    #print(example)