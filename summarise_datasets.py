import os
from datasets import load_from_disk

def summarize_dataset_statistics(dataset_path, dataset_name):
    """
    Summarizes statistics for the given dataset, including unique speakers,
    total segments, and total audio duration.
    
    Arguments:
    - dataset_path: str, Path to the dataset directory
    - dataset_name: str, Name of the dataset (e.g., CU or AKT)
    """
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    print(f"Dataset: {dataset_name}")
    for split in ['train', 'test']:
        if split not in dataset:
            print(f"  Split '{split}' not found.")
            continue
        
        data = dataset[split]
        
        # Number of unique speakers
        unique_speakers = len(set(data['speaker_id']))
        
        # Number of segments
        total_segments = len(data)
        
        # Compute total duration in seconds
        total_duration_seconds = sum(
            len(item['audio']['array']) / item['audio']['sampling_rate']
            for item in data
        )
        total_duration_hours = total_duration_seconds / 3600
        
        print(f"  {split.capitalize()} Split:")
        print(f"    - Unique Speakers: {unique_speakers}")
        print(f"    - Total Segments: {total_segments}")
        print(f"    - Total Duration (hours): {total_duration_hours:.2f}")
    print()

# Paths to the dataset directories
cu_dataset_path = "/srv/scratch/z5369417/created_dataset_3009/cu_dataset_exclude_noise"
akt_dataset_path = "/srv/scratch/z5369417/created_dataset_3009/AKT_exclude_noise"

# Summarize CU and AKT datasets
summarize_dataset_statistics(cu_dataset_path, "CU")
summarize_dataset_statistics(akt_dataset_path, "AKT")
