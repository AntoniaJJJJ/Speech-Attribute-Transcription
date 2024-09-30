"""
Author: Antonia Jian
Date(Last modified): 30/09/2024
Description: This script creates a dataset for the CU data by excluding pairs that contain the <noise> tag.
"""

from datasets import DatasetDict, Dataset, Audio
import pandas as pd
import os

# Function to read the text file and filter out lines containing "<noise>"
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    text_dict = {}
    current_key = None
    current_value = []

    for line in lines:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            if current_key is not None and '<noise>' not in ' '.join(current_value).strip():
                text_dict[current_key] = ' '.join(current_value).strip()
            current_key = parts[0]
            current_value = [parts[1].strip()]
        elif current_key is not None:
            current_value.append(line.strip())

    # Add the last entry if it doesn't contain <noise>
    if current_key is not None and '<noise>' not in ' '.join(current_value).strip():
        text_dict[current_key] = ' '.join(current_value).strip()

    return text_dict

# Function to read the wav.scp file (same as original)
def read_wav_scp_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    wav_paths = {}
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                wav_paths[parts[0]] = parts[1].strip()

    # Modify paths for CU dataset
    for key in wav_paths:
        wav_paths[key] = wav_paths[key].replace(
            "/srv/scratch/z5173707/Dataset/CU_2/corpus/data",
            "/srv/scratch/speechdata/children/TD/CU_2/corpus/data"
        )

    return wav_paths

# Function to load mapping files for speaker ID and age
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

# Function to create the CU dataset excluding <noise> pairs
def create_cu_dataset(base_path):
    dataset_dict = {}

    for split in ['train', 'valid', 'test']:
        text_path = os.path.join(base_path, split, 'text')
        wav_scp_path = os.path.join(base_path, split, 'wav.scp')

        texts = read_text_file(text_path)
        wav_paths = read_wav_scp_file(wav_scp_path)

        # Filter data based on available keys in texts
        data = {
            'audio': [wav_paths[key] for key in texts.keys() if key in wav_paths],
            'text': [texts[key] for key in texts.keys() if key in wav_paths],
            'speaker_id': [speaker_id_mapping['cu'].get(key, None) for key in texts.keys() if key in wav_paths],
            'age': [age_mapping['cu'].get(key, None) for key in texts.keys() if key in wav_paths]
        }

        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column('audio', Audio())
        dataset_dict[split] = dataset

    return DatasetDict(dataset_dict)

# Load speaker ID and age mappings
speaker_id_mapping = {
    'cu': load_mapping('/srv/scratch/z5369417/speakers_info/utt2spk_cu')
}
age_mapping = {
    'cu': load_mapping('/srv/scratch/z5369417/speakers_info/utt2age_cu')
}

# Base path for CU data
cu_base_path = '/srv/scratch/z5369417/children_text_data/cu'
cu_dataset_dict = create_cu_dataset(cu_base_path)

# Save the created dataset
cu_save_path = '/srv/scratch/z5369417/created_dataset_3009/cu_dataset_exclude_noise'
cu_dataset_dict.save_to_disk(cu_save_path)
print(f'CU dataset saved to {cu_save_path}')
print(f'CU DatasetDict:')
print(cu_dataset_dict)