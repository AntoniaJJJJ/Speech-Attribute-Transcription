"""
Author: Antonia Jian
Date(Last modified): 24/07/2024
Description: 
This script create dataset with AKT!!! for training using Hugging Face' library datasets.

For the 'create_dataset_AKT' function:
The dataset is created to store each set as a dictionary with the following structure:
    - 'start_time': the start time of the word interval (in seconds)
    - 'end_time': the end time of the word interval (in seconds)
    - 'word': the text of the word
    - 'audio': A dictionary with the following structure
               . 'array': the actual audio data as an array of samples
               . 'sampling_rate': the sampling rate of the audio data

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
from datasets import Dataset, Audio
from pydub import AudioSegment

# function to read CSV files and extract intervals with word annotations
def read_csv(csv_path):
    # read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
     # extract start time, end time, and word text for each row
    data = [{"start_time": row['tmin'], "end_time": row['tmax'], "word": row['text']} for index, row in df.iterrows()]
    return data

# function to split the audio file into segments based on the annotations
def split_audio(wav_path, segments):
    # load the .wav file using pydub
    audio = AudioSegment.from_wav(wav_path)
    # initialize an empty list to store audio segments
    audio_segments = []

    for segment in segments:
        # convert start time / end time from seconds to milliseconds
        start_ms = segment["start_time"] * 1000
        end_ms = segment["end_time"] * 1000
        # slice the audio segment
        audio_segment = audio[start_ms:end_ms]
        # convert to numpy array
        audio_array = np.array(audio_segment.get_array_of_samples())
        # convert numpy array to list 
        audio_array = audio_array.tolist()

        # add the audio data and sampling rate to the segment
        segment["audio"] = {
            "array": audio_array,
            "sampling_rate": audio.frame_rate
        }
        # append the segment to the list
        audio_segments.append(segment)

    return audio_segments

def create_dataset_AKT(directory_path):
    # List all files in the directory
    all_files = os.listdir(directory_path)
    
    # Create dictionaries to hold paths of wav and csv files
    wav_files = {os.path.splitext(f)[0].replace('_task1', ''): os.path.join(directory_path, f) for f in all_files if f.endswith('_task1.wav')}
    csv_files = {os.path.splitext(f)[0].replace('_kaldi', ''): os.path.join(directory_path, f) for f in all_files if f.endswith('_task1_kaldi.csv')}

    # Find common base names between wav and csv files
    common_files = set(wav_files.keys()).intersection(csv_files.keys())
    all_segments = []

    for file in common_files:
        wav_path = wav_files[file]
        csv_path = csv_files[file]
        segments = read_csv(csv_path)
        audio_segments = split_audio(wav_path, segments)
        all_segments.extend(audio_segments)

    df = pd.DataFrame(all_segments)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio())
    
    return dataset

directory_path = '/srv/scratch/z5369417/AKT_data'
save_path = '/srv/scratch/z5369417/created_dataset_1007/AKT_dataset'

dataset = create_dataset_AKT(directory_path)
dataset.save_to_disk(save_path)
print(f'Dataset saved to {save_path}')