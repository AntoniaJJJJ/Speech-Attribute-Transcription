"""
Author: Antonia Jian
Date(Last modified): 10/07/2024
Description: 
This script reads the text and audio paths, 
create individual datasets for each split (train, valid, test), 
and combine them into a DatasetDict.

"""
from datasets import DatasetDict, Dataset, Audio
import pandas as pd
import os

# function to read the text file
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        # read all lines from the file into a list
        lines = f.readlines()
    # creates and returns a dictionary where the key is a identifier - first part of each line
    # the value is the text transcription
    return {line.split(' ', 1)[0]: line.split(' ', 1)[1].strip() for line in lines}

# function to read the wav.scp file
def read_wav_scp_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # creates a dictionary where the key is the identifier
    # and the value is the audio file path
    wav_paths = {line.split(' ')[0]: line.split(' ')[1].strip() for line in lines}

    # modify paths for OGI dataset
    if source_name == 'ogi':
        for key in wav_paths:
            wav_paths[key] = wav_paths[key].replace(
                "srv/scratch/z5173707/Dataset/OGI/speech/scripted",
                "/srv/scratch/speechdata/children/TD/OGI/speech/scripted"
            )
    
    # modify paths for MyST dataset
    if source_name == 'myst':
        for key in wav_paths:
            wav_paths[key] = wav_paths[key].replace(
                "/srv/scratch/z5173707/Dataset/MyST/data",
                "/srv/scratch/speechdata/children/TD/myst-v0.3.0-171fbda/corpora/myst/data"
            )
    
    # modify paths for CU dataset
    if source_name == 'cu':
        for key in wav_paths:
            wav_paths[key] = wav_paths[key].replace(
                "/srv/scratch/z5173707/Dataset/CU_2/corpus/data",
                "/srv/scratch/speechdata/children/TD/CU_2/corpus/data"
            )
    
    return wav_paths

# function to create a 'DataDict' from the base path where the dataset dicectories are located
def create_dataset_dict(base_path):
    # a dictionary to store datasets for each split
    dataset_dict = {}
    for split in ['train', 'valid', 'test']:
        # constructure the paths
        text_path = os.path.join(base_path, split, 'text')
        wav_scp_path = os.path.join(base_path, split, 'wav.scp')
        
        # read the files and store the transciption and audio path
        texts = read_text_file(text_path)
        wav_paths = read_wav_scp_file(wav_scp_path)
        
        # constructs a dictionary with two lists
        # audio containing the audio file paths and text containing the corresponding transcriptions
        data = {
            'audio': [wav_paths[key] for key in texts.keys()],
            'text': [texts[key] for key in texts.keys()]
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

# paths to the three sources
sources = {
    'ogi': '/srv/scratch/z5369417/children_text_data/ogi',
    'myst': '/srv/scratch/z5369417/children_text_data/myst',
    'cu': '/srv/scratch/z5369417/children_text_data/cu'
}

# path to save the created dataset dictionaries
save_dir = '/srv/scratch/z5369417/created_dataset_1007'

# create and save a DatasetDict for each source
for name, path in sources.items():
    dataset_dict = create_dataset_dict(path)
    save_path = os.path.join(save_dir, f'{name}_dataset')
    dataset_dict.save_to_disk(save_path)
    print(f'{name} dataset saved to {save_path}')
    print(f'{name} DatasetDict:')
    print(dataset_dict)