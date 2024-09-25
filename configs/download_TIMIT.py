from datasets import load_dataset

# Define the target directory
target_directory = '/srv/scratch/z5369417/TIMIT'

# Download the TIMIT dataset and save it to the target directory
dataset = load_dataset("timit_asr", "timit_asr")
dataset.save_to_disk(target_directory)