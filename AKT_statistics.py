import os
import pandas as pd
import numpy as np
import librosa
from datasets import load_from_disk, DatasetDict, Dataset
from collections import Counter

def load_demographic_data(demographic_csv):
    """Load demographic data and return a dictionary indexed by SpeakerID."""
    demographic_df = pd.read_csv(demographic_csv)
    demographic_df = demographic_df.loc[:, ~demographic_df.columns.duplicated()]
    demographic_df['SpeakerID'] = pd.to_numeric(demographic_df['SpeakerID'], errors='coerce')
    demographic_df = demographic_df.dropna(subset=['SpeakerID'])
    demographic_dict = demographic_df[['SpeakerID', 'Gender', 'Age_yrs']].set_index('SpeakerID').T.to_dict()
    return demographic_dict

def read_csv(csv_path):
    """Read the CSV file for a given speaker and extract intervals with word annotations."""
    df = pd.read_csv(csv_path)
    # Extract start time, end time, and word text for each row
    data = [{"start_time": row['tmin'], "end_time": row['tmax'], "word": row['text']} for _, row in df.iterrows()]
    return data

def split_audio(wav_path, segments):
    """Split the audio file into segments based on start and end times from the CSV."""
    audio_data, original_sr = librosa.load(wav_path, sr=44100)
    audio_data_16k = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)

    # Initialize a list to hold segment durations
    segment_durations = []
    for segment in segments:
        # Convert start/end times from seconds to sample indices
        start_sample = int(segment["start_time"] * 16000)
        end_sample = int(segment["end_time"] * 16000)
        # Slice the downsampled audio segment
        segment_audio = audio_data_16k[start_sample:end_sample]
        # Calculate duration in seconds
        duration = len(segment_audio) / 16000
        segment_durations.append(duration)

    return segment_durations

def calculate_statistics_for_experiments(experiment_paths, demographic_csv, annotation_file, data_dir):
    demographic_data = load_demographic_data(demographic_csv)
    annotation_df = pd.read_excel(annotation_file)
    experiment_results = {}

    for exp_name, dataset_path in experiment_paths.items():
        # Load the dataset for this experiment
        dataset = load_from_disk(dataset_path)

        # Create dictionaries to hold paths of wav and csv files
        wav_files = {os.path.splitext(f)[0].replace('_task1', ''): os.path.join(data_dir, f) 
                     for f in os.listdir(data_dir) if f.endswith('_task1.wav')}
        csv_files = {os.path.splitext(f)[0].replace('_task1_kaldi', ''): os.path.join(data_dir, f) 
                     for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv') and not f.endswith('_log.csv')}

        # Initialize statistics for this experiment
        total_train_duration, total_test_duration = 0, 0
        train_segments, test_segments = 0, 0
        train_error_count, test_error_count = 0, 0

        # Process each speaker in the `train` and `test` splits
        for split in ['train', 'test']:
            speakers = set(dataset[split]['speaker_id'])
            for speaker_id in speakers:
                # Skip if speaker ID does not have corresponding WAV/CSV files
                if str(speaker_id) not in wav_files or str(speaker_id) not in csv_files:
                    continue

                wav_path = wav_files[str(speaker_id)]
                csv_path = csv_files[str(speaker_id)]

                # Read the CSV to get segment start/end times and load the audio file for duration
                segments = read_csv(csv_path)
                segment_durations = split_audio(wav_path, segments)

                # Aggregate statistics based on the split
                if split == 'train':
                    total_train_duration += sum(segment_durations)
                    train_segments += len(segment_durations)
                else:
                    total_test_duration += sum(segment_durations)
                    test_segments += len(segment_durations)

                # Count errors based on annotation data for this speaker
                error_cols = [col for col in annotation_df.columns if 'Difference' in col]
                speaker_errors = annotation_df[annotation_df['Child_ID'] == speaker_id][error_cols].notna().sum(axis=1).sum()
                if split == 'train':
                    train_error_count += speaker_errors
                else:
                    test_error_count += speaker_errors

        # Collect demographic statistics
        train_speakers = set(dataset['train']['speaker_id'])
        test_speakers = set(dataset['test']['speaker_id'])
        unique_speakers = train_speakers | test_speakers

        # Total number of speakers
        total_speakers = len(unique_speakers)

        # Age range and gender distribution
        def get_age_range(speakers):
            ages = [demographic_data[speaker_id]['Age_yrs'] for speaker_id in speakers if speaker_id in demographic_data]
            return (min(ages), max(ages)) if ages else (None, None)

        def get_gender_distribution(speakers):
            genders = [demographic_data[speaker_id]['Gender'] for speaker_id in speakers if speaker_id in demographic_data]
            return Counter(genders)

        age_range_overall = get_age_range(unique_speakers)
        age_range_train = get_age_range(train_speakers)
        age_range_test = get_age_range(test_speakers)

        gender_overall = get_gender_distribution(unique_speakers)
        gender_train = get_gender_distribution(train_speakers)
        gender_test = get_gender_distribution(test_speakers)

        # Store the statistics for this experiment
        experiment_results[exp_name] = {
            "Total Dataset Size (number of speakers)": total_speakers,
            "Total Train Set Size (number of speakers)": len(train_speakers),
            "Total Test Set Size (number of speakers)": len(test_speakers),
            "Total Dataset Size (Segments)": train_segments + test_segments,
            "Total Train Set Size (Segments)": train_segments,
            "Total Test Set Size (Segments)": test_segments,
            "Total Segment Duration (Train)": total_train_duration,
            "Total Segment Duration (Test)": total_test_duration,
            "Average Error Count per Segment (Train)": train_error_count / train_segments if train_segments else 0,
            "Average Error Count per Segment (Test)": test_error_count / test_segments if test_segments else 0,
            "Age Range of Speakers (Overall)": age_range_overall,
            "Age Range of Speakers (Train)": age_range_train,
            "Age Range of Speakers (Test)": age_range_test,
            "Gender Distribution (Overall)": gender_overall,
            "Gender Distribution (Train)": gender_train,
            "Gender Distribution (Test)": gender_test
        }

    # Display the statistics for each experiment
    for exp_name, stats in experiment_results.items():
        print(f"--- Statistics for {exp_name} ---")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value}")
        print("\n")

# Define the paths for each experiment's output dataset
experiment_paths = {
    "Exp14": "/srv/scratch/z5369417/created_dataset_0808/AKT_dataset",
    "Exp16": "/srv/scratch/z5369417/created_dataset_3009/AKT_exclude_noise",
    "Exp17": "/srv/scratch/z5369417/created_dataset_3009/AKT_complete_removal",
    "Exp18": "/srv/scratch/z5369417/created_dataset_3009/AKT_move"
}

# Define paths to demographic and annotation files and the raw data directory
demographic_csv = "/srv/scratch/z5369417/AKT_data_processing/AKT_demographic.csv"
annotation_file = "/srv/scratch/z5369417/AKT_data_processing/AKT_id2diagnosis.xlsx"
data_dir = "/srv/scratch/z5369417/AKT_data/"

# Run the function to calculate statistics for all experiments
calculate_statistics_for_experiments(experiment_paths, demographic_csv, annotation_file, data_dir)