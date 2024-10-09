"""
Author: Antonia Jian
Date(Last modified): 09/10/2024
Description: 
This script create dataset with AKT (audio and CSV)!!! for training using Hugging Face' library datasets.
It utilizes demographic information (speaker ID, age, and gender) and outputs the dataset in 
a DatasetDict format (train, valid, test).

The script reads word annotations from CSV files, splits the corresponding audio into segments, 
and adds demographic info for each segment. 
The output dataset will have the following structure:

DatasetDict({
    train: Dataset({
        features: ['audio', 'text', 'speaker_id', 'age', 'gender'],
        num_rows: <number_of_rows>
    }),
    valid: Dataset({
        features: ['audio', 'text', 'speaker_id', 'age', 'gender'],
        num_rows: <number_of_rows>
    }),
    test: Dataset({
        features: ['audio', 'text', 'speaker_id', 'age', 'gender'],
        num_rows: <number_of_rows>
    })
})
- 'audio': Contains audio data as a NumPy array and sampling rate
- 'text': The corresponding transcription
- 'speaker_id': The ID of the speaker
- 'age': The age of the speaker
- 'gender': The gender of the speaker

For the 'read_csv' function:
It reads a csv file and extracts the intervals with word annotations.
The output is a list of dictionaries, each containing:
    - 'start_time': the start time of the word interval (in seconds)
    - 'end_time': the end time of the word interval (in seconds)
    - 'word': the text of the word

For the 'split_audio' function:
It splits a .wav audio file into segments based on the provided annotations 
and stores each segment along with its corresponding word.
Inputs are path to .wav file and a list of dictionaries for the word annotations
The output is the same as 'create_data' function

"""
import os
import pandas as pd
import numpy as np
from datasets import DatasetDict, Dataset, Audio
from pydub import AudioSegment

# Load demographic data (SpeakerID, Age, Gender) for AKT
def load_demographic_data(demographic_csv):
    # Read the demographic data CSV file and create a dictionary indexed by SpeakerID
    demographic_df = pd.read_csv(demographic_csv)
    demographic_dict = demographic_df.set_index("SpeakerID").T.to_dict()
    return demographic_dict

# Function to read CSV files and extract intervals with word annotations
def read_csv(csv_path):
    # read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
     # extract start time, end time, and word text for each row
    data = [{"start_time": row['tmin'], "end_time": row['tmax'], "word": row['text']} for _, row in df.iterrows()]
    return data

# Function to split the audio file into segments based on the annotations
def split_audio(wav_path, segments):
    # load the .wav file using pydub
    audio = AudioSegment.from_wav(wav_path)
    # initialize an empty list to store audio segments
    audio_data = []

    # Process each segment, extracting the corresponding audio segment
    for segment in segments:
        # convert start time / end time from seconds to milliseconds
        start_ms = segment["start_time"] * 1000
        end_ms = segment["end_time"] * 1000
        # transcription associated with the segment
        word = segment["word"]
        # slice the audio segment
        segment_audio = audio[start_ms:end_ms]
        # append the audio segment data and transcription to the list
        audio_data.append({
            "audio": {
                "array": segment_audio.get_array_of_samples(),
                "sampling_rate": segment_audio.frame_rate
            },
            "text": word
        })

    return audio_data

# Function to create the Hugging Face dataset for a single CSV/WAV pair
def create_dataset_AKT(csv_path, wav_path, speaker_id, speaker_data):
    # Creates a Hugging Face dataset by reading CSV and audio data and attaching demographic info
    # Extract the word intervals from the CSV file
    segments = read_csv(csv_path)
    # Extract audio segments based on those intervals
    audio_segments = split_audio(wav_path, segments)

    # Get the age and gender information from the speaker data
    speaker_info = speaker_data.get(int(speaker_id), {})
    age = speaker_info.get("Age_yrs", 30)  # Default age is 30 if not available
    gender = speaker_info.get("Gender", "Unknown")  # Default gender is 'Unknown' if not available

    # Build the dataset for each audio segment with demographic info
    data = {
        "audio": [segment["audio"] for segment in audio_segments],
        "text": [segment["text"] for segment in audio_segments],
        "speaker_id": [speaker_id] * len(audio_segments),
        "age": [age] * len(audio_segments),
        "gender": [gender] * len(audio_segments)
    }

    # Create a Hugging Face Dataset object from the dictionary
    dataset = Dataset.from_dict(data)
    return dataset

# Main function to create the DatasetDict for AKT data
def create_dataset_dict_AKT(data_dir, demographic_csv, output_dir):
    # Processes all splits (train, valid, test) and creates a DatasetDict
    # Load the demographic data (age, gender) for all speakers
    demographic_data = load_demographic_data(demographic_csv)
    splits = ["train", "valid", "test"]
    datasets = {}

    # Loop through each split directory (train, valid, test)
    for split in splits:
        split_dir = os.path.join(data_dir, split)  # Get the directory for this split

         # Create dictionaries to hold paths of wav and csv files, adjusting names
        wav_files = {os.path.splitext(f)[0].replace('_task1', ''): os.path.join(split_dir, f) 
                     for f in os.listdir(split_dir) if f.endswith('_task1.wav')}
        csv_files = {os.path.splitext(f)[0].replace('_kaldi', ''): os.path.join(split_dir, f) 
                     for f in os.listdir(split_dir) if f.endswith('_task1_kaldi.csv')}

        # Find common base names between wav and csv files
        common_files = set(wav_files.keys()).intersection(csv_files.keys())
        split_datasets = [] # List to store datasets for this split

        # Process matching wav and csv files
        for file in common_files:
            wav_path = wav_files[file]
            csv_path = csv_files[file]
            speaker_id = file.split("_")[0]  # Extract speaker ID from the file name
            dataset = create_dataset_AKT(csv_path, wav_path, speaker_id, demographic_data)  # Create dataset
            split_datasets.append(dataset)

         # Combine datasets for the current split and add to the DatasetDict
        if split_datasets:
            datasets[split] = Dataset.from_concat(split_datasets)  # Concatenate all datasets in the current split

     # Create a DatasetDict and save the combined dataset to the output directory
    dataset_dict = DatasetDict(datasets)
    dataset_dict.save_to_disk(output_dir)


data_directory = "/srv/scratch/z5369417/AKT_data/"  # Both CSV and WAV files are in the same folder
demographic_csv = "/srv/scratch/z5369417/AKT_data_processing/AKT_demographic.csv"
output_directory = "/srv/scratch/z5369417/created_dataset_0808/AKT_dataset"
create_dataset_dict_AKT(data_directory, demographic_csv, output_directory)
print(f'Dataset saved to {output_directory}')