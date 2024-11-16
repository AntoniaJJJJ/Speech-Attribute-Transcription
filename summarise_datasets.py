import os
from datasets import load_from_disk
from collections import Counter

def calculate_durations(data_split):
    """Calculate total duration and segment count from the audio array in each entry."""
    durations = []
    for entry in data_split:
        # Calculate duration from the audio array length
        duration = len(entry['audio']['array']) / entry['audio']['sampling_rate']
        durations.append(duration)
    return sum(durations), len(durations)

def calculate_statistics(dataset_path, dataset_name):
    """
    Summarizes statistics for the given dataset, including unique speakers,
    total segments, total audio duration, and speaker demographics.
    """
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    print(f"Dataset: {dataset_name}")
    for split in ['train', 'test']:
        if split not in dataset:
            print(f"  Split '{split}' not found.")
            continue
        
        data_split = dataset[split]
        
        # Calculate total durations and segment counts
        total_duration_seconds, total_segments = calculate_durations(data_split)
        total_duration_hours = total_duration_seconds / 3600
        
        # Calculate unique speakers
        unique_speakers = set(entry['speaker_id'] for entry in data_split)
        num_unique_speakers = len(unique_speakers)
        
        # Calculate age range and gender distribution
        speaker_info = {entry['speaker_id']: (entry['age'], entry.get('gender', 'Unknown')) for entry in data_split}
        ages = [int(age) for age, _ in speaker_info.values() if age is not None]
        age_range = (min(ages), max(ages)) if ages else (None, None)
        genders = [gender for _, gender in speaker_info.values()]
        gender_distribution = Counter(genders)
        
        print(f"  {split.capitalize()} Split:")
        print(f"    - Unique Speakers: {num_unique_speakers}")
        print(f"    - Total Segments: {total_segments}")
        print(f"    - Total Duration (hours): {total_duration_hours:.2f}")
        print(f"    - Age Range: {age_range}")
        print(f"    - Gender Distribution: Male: {gender_distribution['Male']}, Female: {gender_distribution['Female']}, Unknown: {gender_distribution.get('Unknown', 0)}")
    print()

# Paths to the dataset directories
cu_dataset_path = "/srv/scratch/z5369417/created_dataset_3009/cu_dataset_exclude_noise"
akt_dataset_path = "/srv/scratch/z5369417/created_dataset_3009/AKT_exclude_noise"

# Summarize CU and AKT datasets
calculate_statistics(cu_dataset_path, "CU")
calculate_statistics(akt_dataset_path, "AKT")