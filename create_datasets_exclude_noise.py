"""
Author: Antonia Jian
Date(Last modified): 09/10/2024
Description: This script creates a dataset for the CU, Myst and Ogi data by excluding pairs that contain the <noise> tag.
"""

import argparse
from datasets import DatasetDict, Dataset, Audio
import pandas as pd
import os

# Add argument parsing to select the dataset
def parse_arguments():
    parser = argparse.ArgumentParser(description="Create datasets by excluding noise for specific datasets.")
    parser.add_argument('--dataset', type=str, choices=['cu', 'myst', 'ogi'], required=True,
                        help='Select the dataset: cu, myst, or ogi')
    args = parser.parse_args()
    return args

# Function to read the text file and filter out lines containing "<noise>"
def read_text_file(file_path, dataset_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Handle 'cu' dataset with multi-line transcriptions
    if dataset_name == 'cu':
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
        
        if current_key is not None and '<noise>' not in ' '.join(current_value).strip():
            text_dict[current_key] = ' '.join(current_value).strip()

        return text_dict

    # Handle 'myst' and 'ogi' datasets (each line represents a full transcription)
    else:
        return {
            line.split(' ', 1)[0]: line.split(' ', 1)[1].strip()
            for line in lines if '<noise>' not in line
        }

# Function to read the wav.scp file and adjust paths for cu, myst, ogi
def read_wav_scp_file(file_path, dataset_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    wav_paths = {}
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                wav_paths[parts[0]] = parts[1].strip()

    # Modify paths based on the dataset
    if dataset_name == 'cu':
        for key in wav_paths:
            wav_paths[key] = wav_paths[key].replace(
                "/srv/scratch/z5173707/Dataset/CU_2/corpus/data",
                "/srv/scratch/speechdata/children/TD/CU_2/corpus/data"
            )
    elif dataset_name == 'myst':
        for key in wav_paths:
            wav_paths[key] = wav_paths[key].replace(
                "/srv/scratch/z5173707/Dataset/MyST//data",
                "/srv/scratch/speechdata/children/TD/myst-v0.3.0-171fbda/corpora/myst/data"
            )
    elif dataset_name == 'ogi':
        for key in wav_paths:
            wav_paths[key] = wav_paths[key].replace(
                "srv/scratch/z5173707/Dataset/OGI/speech",
                "/srv/scratch/speechdata/children/TD/OGI/speech"
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

# function to create a 'DataDict' from the base path where the dataset dicectories are located
def create_dataset_dict(base_path, source_name, speaker_id_mapping, age_mapping):
    # a dictionary to store datasets for each split
    dataset_dict = {}
    for split in ['train', 'valid', 'test']:
        # constructure the paths
        text_path = os.path.join(base_path, split, 'text')
        wav_scp_path = os.path.join(base_path, split, 'wav.scp')
        
        # read the files and store the transciption and audio path
        texts = read_text_file(text_path, source_name)
        wav_paths = read_wav_scp_file(wav_scp_path, source_name)
        
        # constructs a dictionary with two lists
        # audio containing the audio file paths and text containing the corresponding transcriptions
        data = {
            'audio': [wav_paths[key] for key in texts.keys()],
            'text': [texts[key] for key in texts.keys()],
            'speaker_id': [speaker_id_mapping[source_name].get(key, None) for key in texts.keys()],
            'age': [age_mapping[source_name].get(key, None) if source_name in ['cu', 'ogi'] else None for key in texts.keys()]
        }
        
        # creates a pandas DataFrame from the data dictionary
        df = pd.DataFrame(data)
        # convert to Hugging Face 'Dataset'
        dataset = Dataset.from_pandas(df)
        # casts the audio column to the Audio type, enabling audio data handling
        dataset = dataset.cast_column('audio', Audio())
        # adds the dataset to the dataset_dict with the split name as the key
        dataset_dict[split] = dataset
    
    # returns a DatasetDict created from the dataset_dict
    return DatasetDict(dataset_dict)

# Main function to process the dataset
def process_dataset():
    args = parse_arguments()
    dataset_name = args.dataset

    # Reuse paths and mappings from create_datasets.py
    sources = {
        'ogi': '/srv/scratch/z5369417/children_text_data/ogi',
        'myst': '/srv/scratch/z5369417/children_text_data/myst',
        'cu': '/srv/scratch/z5369417/children_text_data/cu'
    }

    speaker_id_mapping = {
        'cu': load_mapping('/srv/scratch/z5369417/speakers_info/utt2spk_cu'),
        'myst': load_mapping('/srv/scratch/z5369417/speakers_info/utt2spk_myst'),
        'ogi': load_mapping('/srv/scratch/z5369417/speakers_info/utt2spk_ogi')
    }

    age_mapping = {
        'cu': load_mapping('/srv/scratch/z5369417/speakers_info/utt2age_cu'),
        'ogi': load_mapping('/srv/scratch/z5369417/speakers_info/utt2age_ogi')
    }

    # Use the selected dataset's paths
    base_path = sources[dataset_name]
    
    # Create and save the dataset
    dataset_dict = create_dataset_dict(base_path, dataset_name, speaker_id_mapping, age_mapping)
    # Define the folder structure
    save_dir = f'/srv/scratch/z5369417/created_dataset_3009/{dataset_name}_exclude_noise'
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    

    dataset_dict.save_to_disk(save_dir)

    print(f'{dataset_name} dataset saved to {save_dir}')

if __name__ == '__main__':
    process_dataset()