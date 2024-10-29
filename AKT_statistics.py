import os
import pandas as pd
import numpy as np
from datasets import load_from_disk
from collections import Counter

def calculate_durations(data_split):
    """Calculate total duration and segment count from the audio array in each entry."""
    durations = []
    for entry in data_split:
        duration = len(entry['audio']['array']) / entry['audio']['sampling_rate']
        durations.append(duration)
    return sum(durations), len(durations)

def count_segments_with_errors(speaker_id, data_dir):
    """Count the number of segments with one or more errors for each speaker."""
    csv_path = os.path.join(data_dir, f"{speaker_id}_task1_kaldi.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found for speaker {speaker_id}")
        return 0

    df = pd.read_csv(csv_path)
    error_cols = [col for col in df.columns if 'Difference' in col]
    segments_with_errors = df[error_cols].notna().any(axis=1).sum()
    return segments_with_errors

def get_age_range_and_gender_distribution(data_split, exclude_age=None):
    """Get age range and gender count by unique speakers in a split, with optional age exclusion."""
    unique_speakers = {entry['speaker_id']: (entry['age'], entry['gender']) for entry in data_split}
    
    # Filter out speakers with the specified age if exclude_age is given
    if exclude_age is not None:
        unique_speakers = {k: v for k, v in unique_speakers.items() if v[0] != exclude_age}

    ages = [age for age, gender in unique_speakers.values() if age is not None]
    genders = [gender if gender else 'Unknown' for _, gender in unique_speakers.values()]
    
    age_range = (min(ages), max(ages)) if ages else (None, None)
    gender_counts = Counter(genders)
    return age_range, gender_counts, len(unique_speakers)

def calculate_statistics(dataset_path, data_dir, experiment_name, exclude_age=None):
    # Load the output dataset for the experiment
    dataset = load_from_disk(dataset_path)

    # Define train and test splits
    train_split = dataset['train']
    test_split = dataset['test']
    
    # Calculate total durations and segment counts for train and test
    total_train_duration, train_segments = calculate_durations(train_split)
    total_test_duration, test_segments = calculate_durations(test_split)

    # Calculate the number of segments with one or more errors for train and test
    train_segments_with_errors = sum(count_segments_with_errors(entry['speaker_id'], data_dir) for entry in train_split)
    test_segments_with_errors = sum(count_segments_with_errors(entry['speaker_id'], data_dir) for entry in test_split)

    # Calculate age range and gender distribution for train and test, with optional age exclusion
    age_range_train, gender_distribution_train, train_speaker_count = get_age_range_and_gender_distribution(train_split, exclude_age)
    age_range_test, gender_distribution_test, test_speaker_count = get_age_range_and_gender_distribution(test_split, exclude_age)

    # Print statistics for the experiment
    print(f"--- Statistics for {experiment_name} ---")
    print(f"Total Dataset Size (number of speakers): {train_speaker_count + test_speaker_count}")
    print(f"Total Train Set Size (number of speakers): {train_speaker_count}")
    print(f"Total Test Set Size (number of speakers): {test_speaker_count}")
    print(f"Total Dataset Size (Segments): {train_segments + test_segments}")
    print(f"Total Train Set Size (Segments): {train_segments}")
    print(f"Total Test Set Size (Segments): {test_segments}")
    print(f"Total Segment Duration (Train): {total_train_duration}")
    print(f"Total Segment Duration (Test): {total_test_duration}")
    print(f"Number of Segments with Errors (Train): {train_segments_with_errors}")
    print(f"Number of Segments with Errors (Test): {test_segments_with_errors}")
    print(f"Age Range of Speakers (Train): {age_range_train}")
    print(f"Age Range of Speakers (Test): {age_range_test}")
    print(f"Gender Distribution (Train): Male: {gender_distribution_train['Male']}, Female: {gender_distribution_train['Female']}, Unknown: {gender_distribution_train.get('Unknown', 0)}")
    print(f"Gender Distribution (Test): Male: {gender_distribution_test['Male']}, Female: {gender_distribution_test['Female']}, Unknown: {gender_distribution_test.get('Unknown', 0)}")
    print("\n")


# Example usage for multiple experiments
data_dir = "/srv/scratch/z5369417/AKT_data"  # Directory containing raw CSV files for each speaker

# Paths to the output datasets for each experiment
experiments = {
    "Exp14": "/srv/scratch/z5369417/created_dataset_0808/AKT_dataset",
    "Exp16": "/srv/scratch/z5369417/created_dataset_3009/AKT_exclude_noise",
    "Exp17": "/srv/scratch/z5369417/created_dataset_3009/AKT_complete_removal",
    "Exp18": "/srv/scratch/z5369417/created_dataset_3009/AKT_move"
}

for exp_name, dataset_path in experiments.items():
    calculate_statistics(dataset_path, data_dir, exp_name)