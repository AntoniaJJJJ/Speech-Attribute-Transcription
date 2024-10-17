"""
Author: Antonia Jian
Date(Last modified): 17/10/2024
Description: 
This script create dataset with AKT (audio and CSV)!!! for training using Hugging Face' library datasets.
It utilizes demographic information (speaker ID, age, and gender) and outputs the dataset in 
a DatasetDict format (train).

The script reads word annotations from CSV files, splits the corresponding audio into segments, 
and adds demographic info for each segment. 
The script has limited to process limited files and limited batches
The output dataset will have the following structure:

DatasetDict({
    train: Dataset({
        features: ['audio', 'text', 'speaker_id', 'age', 'gender'],
        num_rows: <number_of_rows>
    }),
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
from io import BytesIO
from datasets import DatasetDict, Dataset, Audio
from datasets import concatenate_datasets
from pydub import AudioSegment

# Load demographic data (SpeakerID, Age, Gender) for AKT
def load_demographic_data(demographic_csv):
    # Read the demographic data CSV file and handle any issues with SpeakerID conversion
    demographic_df = pd.read_csv(demographic_csv)

    # Remove duplicate columns (if any) (such error is shown when running)
    demographic_df = demographic_df.loc[:, ~demographic_df.columns.duplicated()]
    
    # Convert SpeakerID to numeric, dropping rows where conversion fails
    demographic_df['SpeakerID'] = pd.to_numeric(demographic_df['SpeakerID'], errors='coerce')
    demographic_df = demographic_df.dropna(subset=['SpeakerID'])
    
    # Create a dictionary indexed by SpeakerID
    demographic_dict = demographic_df[['SpeakerID', 'Gender', 'Age_yrs']].set_index('SpeakerID').T.to_dict()
    
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
    # Downsample the audio to 16kHz
    audio = audio.set_frame_rate(16000)
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
        # Append the audio segment data and transcription to the list
        audio_data.append({
            "audio": {
                "array": segment_audio.get_array_of_samples(),
                "sampling_rate": 16000  # Set the sampling rate to 16kHz
            },
            "text": word
        })

    return audio_data

# Function to create the Hugging Face dataset for a single CSV/WAV pair
def create_dataset_AKT(csv_path, wav_path, speaker_id, speaker_data, batch_size=200):
    # Creates a Hugging Face dataset by reading CSV and audio data and attaching demographic info
    # Extract the word intervals from the CSV file
    segments = read_csv(csv_path)

    # Ensure speaker_id is handled properly by converting to the correct type
    try:
        speaker_id = int(speaker_id)  # Ensure the speaker_id is an integer
    except ValueError:
        speaker_id = None

     # Get the age and gender information from the speaker data
    speaker_info = speaker_data.get(speaker_id, {}) if speaker_id is not None else {}
    age = speaker_info.get("Age_yrs", None)  # Set age to None if not available
    gender = str(speaker_info.get("Gender", "Unknown"))  # Set gender to Unknown if not available

    # Limit the segments to only the batch size (e.g., 500 segments)
    limited_segments = segments[:batch_size]

    # Extract audio segments based on those intervals
    audio_segments = split_audio(wav_path, limited_segments)

    # Build the dataset for each audio segment with demographic info
    data = {
        "audio": [segment["audio"] for segment in audio_segments],  # In-memory WAV data
        "text": [segment["text"] for segment in audio_segments],
        "speaker_id": [speaker_id] * len(audio_segments),
        "age": [age] * len(audio_segments),
        "gender": [gender] * len(audio_segments)
    }

    # Create a Hugging Face Dataset object from the dictionary
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))  # Tell datasets to treat 'audio' as an Audio feature
    return dataset

# Main function to create the DatasetDict for AKT data
def create_dataset_dict_AKT(data_dir, demographic_csv, output_dir, num_files_to_process=20, batch_size=500):
    # Processes 50 files in the AKT data directory and creates a DatasetDict with the 'train' split.
    # Load the demographic data (age, gender) for all speakers
    demographic_data = load_demographic_data(demographic_csv)

    # Create dictionaries to hold paths of wav and csv files, adjusting names
    wav_files = {os.path.splitext(f)[0].replace('_task1', ''): os.path.join(data_dir, f) 
             for f in os.listdir(data_dir) if f.endswith('_task1.wav')}
    csv_files = {os.path.splitext(f)[0].replace('_task1_kaldi', ''): os.path.join(data_dir, f) 
             for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv') and not f.endswith('_log.csv')}

    # Find common base names between wav and csv files
    common_files = list(set(wav_files.keys()).intersection(csv_files.keys()))[:num_files_to_process]  # Limit the files processed
    all_datasets = []


    # Process matching wav and csv files
    for file in common_files:
        wav_path = wav_files[file]
        csv_path = csv_files[file]
        speaker_id = file.split("_")[0]  # Extract speaker ID from the file name
        dataset = create_dataset_AKT(csv_path, wav_path, speaker_id, demographic_data, batch_size=batch_size)  # Create dataset with batch size
        all_datasets.append(dataset)

    # Combine all datasets into a single dataset for the 'train' split
    train_dataset = concatenate_datasets(all_datasets)

    # Create a DatasetDict with 'train' split
    dataset_dict = DatasetDict({"train": train_dataset})
    # Save the DatasetDict to disk
    dataset_dict.save_to_disk(output_dir)


data_directory = "/srv/scratch/z5369417/AKT_data/"  # Both CSV and WAV files are in the same folder
demographic_csv = "/srv/scratch/z5369417/AKT_data_processing/AKT_demographic.csv"
output_directory = "/srv/scratch/z5369417/created_dataset_0808/AKT_dataset"
create_dataset_dict_AKT(data_directory, demographic_csv, output_directory, num_files_to_process=20, batch_size=200)
print(f'Dataset saved to {output_directory}')