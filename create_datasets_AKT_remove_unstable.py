"""
Author: Antonia Jian
Date(Last modified): 26/10/2024
Description: 
This program is an extension of create_datasets_AKT_updated, with the following features:
    1. Exclude 3-year-old speech from the datasets
    2. Distribute the samples based on the number of errors in the speech to 'train' or
       'test' split

"""
import os
import pandas as pd
import numpy as np
import librosa
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

    # Exclude 3-year-olds
    demographic_df = demographic_df[demographic_df['Age_yrs'] != 3]
    
    # Create a dictionary indexed by SpeakerID
    demographic_dict = demographic_df[['SpeakerID', 'Gender', 'Age_yrs']].set_index('SpeakerID').T.to_dict()
    
    return demographic_dict

# Function to read CSV files and extract intervals with word annotations
def read_csv(csv_path):
    # read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    # Count errors in 'Difference' columns (count any non-NA values as errors)
    error_cols = [col for col in df.columns if 'Difference' in col]
    df['error_count'] = df[error_cols].notna().sum(axis=1)
    # extract start time, end time, and word text for each row
    data = [{"start_time": row['tmin'], "end_time": row['tmax'], "word": row['text']} for _, row in df.iterrows()]
    return data, df['error_count'].tolist()

# Function to split the audio file into segments based on the annotations
def split_audio(wav_path, segments):
      # Load the original .wav file at 44100Hz using librosa and downsample to 16kHz
    audio_data, original_sr = librosa.load(wav_path, sr=44100)  # Load original at 44100Hz
    audio_data_16k = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)  # Downsample to 16kHz
    
    # Initialize an empty list to store audio segments
    audio_data_list = []

    # Process each segment, extracting the corresponding audio segment
    for segment in segments:
        # Convert start time / end time from seconds to sample indices
        start_sample = int(segment["start_time"] * 16000)  # 16kHz sampling rate
        end_sample = int(segment["end_time"] * 16000)  # 16kHz sampling rate
        # Transcription associated with the segment
        word = segment["word"]
        # Slice the downsampled audio segment
        segment_audio = audio_data_16k[start_sample:end_sample]
        # Append the audio segment data and transcription to the list
        audio_data_list.append({
            "audio": {
                "path": wav_path,  # Include the correct path to the WAV file
                "array": np.array(segment_audio, dtype=np.float32),  # Store as NumPy array
                "sampling_rate": 16000  # Set the correct sampling rate
            },
            "text": word
        })

    return audio_data_list

# Function to create the Hugging Face dataset for a single CSV/WAV pair
def create_dataset_AKT(csv_path, wav_path, speaker_id, speaker_data):
    # Creates a Hugging Face dataset by reading CSV and audio data and attaching demographic info
    # Extract the word intervals from the CSV file
    segments, error_counts = read_csv(csv_path)

    # Ensure speaker_id is handled properly by converting to the correct type
    try:
        speaker_id = int(speaker_id)  # Ensure the speaker_id is an integer
    except ValueError:
        speaker_id = None

     # Get the age and gender information from the speaker data
    speaker_info = speaker_data.get(speaker_id, {}) if speaker_id is not None else {}
    age = speaker_info.get("Age_yrs", None)  # Set age to None if not available
    gender = str(speaker_info.get("Gender", "Unknown"))  # Set gender to Unknown if not available

    # Extract audio segments based on those intervals
    audio_segments = split_audio(wav_path, segments)

    # Build the dataset for each audio segment with demographic info
    data = {
        "audio": [segment["audio"] for segment in audio_segments],  # Provide the path to the wav file
        "text": [segment["text"] for segment in audio_segments],
        "speaker_id": [speaker_id] * len(audio_segments),
        "age": [age] * len(audio_segments),
        "gender": [gender] * len(audio_segments)
    }

    # Create a Hugging Face Dataset object from the dictionary
    dataset = Dataset.from_dict(data)

    dataset = dataset.cast_column("audio", Audio())  # Tell datasets to treat 'audio' as an Audio feature
    return dataset, error_counts

# Main function to create the DatasetDict with 'train' and 'test' splits
def create_dataset_dict_AKT(data_dir, demographic_csv, annotation_file, output_dir):
    # Load the demographic data (age, gender) for all speakers
    demographic_data = load_demographic_data(demographic_csv)

    # Load the annotation file to get SSD and hand-corrected info
    annotation_df = pd.read_excel(annotation_file)

    # Filter out children without hand-corrected data
    handcorrected_df = annotation_df[annotation_df['T1_Handcorrection_Completed'] == 1]
    not_handcorrected_ids = annotation_df[annotation_df['T1_Handcorrection_Completed'] == 0]['Child_ID'].tolist()

    # Create dictionaries to hold paths of wav and csv files, adjusting names
    wav_files = {os.path.splitext(f)[0].replace('_task1', ''): os.path.join(data_dir, f) 
             for f in os.listdir(data_dir) if f.endswith('_task1.wav')}
    csv_files = {os.path.splitext(f)[0].replace('_task1_kaldi', ''): os.path.join(data_dir, f) 
             for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv') and not f.endswith('_log.csv')}

    # Find common base names between wav and csv files
    common_files = set(wav_files.keys()).intersection(csv_files.keys())
   
    processed_ids = []
    missing_data_ids = []
    train_datasets = []
    test_datasets = []
    all_error_counts = []


    # Process matching wav and csv files
    for _, row in handcorrected_df.iterrows():
        speaker_id = row['Child_ID']

        # Check if the speaker_id has corresponding WAV and CSV files
        if str(speaker_id) in common_files:
            wav_path = wav_files[str(speaker_id)]
            csv_path = csv_files[str(speaker_id)]
            ssd_status = row['SSD']

            # Create the dataset
            dataset, error_counts = create_dataset_AKT(csv_path, wav_path, speaker_id, demographic_data)
            all_error_counts.extend(error_counts)

            # Append the dataset to train or test split based on SSD status
            # Apply first and second sorting criteria
            train_segments = []
            test_segments = []
            for i, segment in enumerate(dataset):
                # Convert each segment to a Dataset object
                segment_data = {key: [value] for key, value in segment.items()}  # Ensure segment is structured properly
                segment_dataset = Dataset.from_dict(segment_data)
                # Add to train or test based on SSD and error count criteria
                if ssd_status == 0 and error_counts[i] < 2:
                    train_segments.append(segment_dataset)
                else:
                    test_segments.append(segment_dataset)
            # Combine segments back into one Dataset for each speaker
            if train_segments:
                train_datasets.append(concatenate_datasets(train_segments))
            if test_segments:
                test_datasets.append(concatenate_datasets(test_segments))

            # Add to processed_ids
            processed_ids.append(speaker_id)
        else:
            # If the speaker_id doesn't have WAV/CSV, add to missing_data_ids
            missing_data_ids.append(speaker_id)

    # Combine all datasets for train and test splits
    if train_datasets:
        train_dataset = concatenate_datasets(train_datasets)
    if test_datasets:
        test_dataset = concatenate_datasets(test_datasets)

    # Create a DatasetDict with 'train' and 'test' splits
    dataset_dict = DatasetDict({
        "train": train_dataset if train_datasets else None,
        "test": test_dataset if test_datasets else None
    })

    # Save the DatasetDict to disk
    dataset_dict.save_to_disk(output_dir)

    # Print out unique numbers of errors in the whole dataset
    unique_error_counts = set(all_error_counts)
    print("Unique error counts in dataset:", unique_error_counts)

    # Print out the results
    print(f"Number of IDs processed (hand-corrected, with data): {len(processed_ids)}")
    print(f"Processed IDs: {processed_ids}")
    print(f"Number of IDs with hand-corrected transcriptions but missing data: {len(missing_data_ids)}")
    print(f"Missing data IDs: {missing_data_ids}")
    print(f"Number of IDs removed (not hand-corrected but have data): {len(not_handcorrected_ids)}")
    print(f"Not hand-corrected IDs: {not_handcorrected_ids}")


data_directory = "/srv/scratch/z5369417/AKT_data/"  # Both CSV and WAV files are in the same folder
demographic_csv = "/srv/scratch/z5369417/AKT_data_processing/AKT_demographic.csv"
output_directory = "/srv/scratch/z5369417/created_dataset_3009/AKT_exclude_noise"
annotation_file = "/srv/scratch/z5369417/AKT_data_processing/AKT_id2diagnosis.xlsx" 
create_dataset_dict_AKT(data_directory, demographic_csv, annotation_file, output_directory)
print(f'Dataset saved to {output_directory}')