from datasets import load_from_disk
import numpy as np

# Path to your preprocessed Hugging Face dataset
dataset_path = "/srv/scratch/z5369417/outputs/phonemization_speechocean/"

# Load the preprocessed dataset
dataset = load_from_disk(dataset_path)

# Function to calculate statistics for a given split (train/test)
def calculate_statistics(split):
    total_segments = 0
    total_speakers = set()  # Use a set to store unique speakers
    total_duration = 0  # in seconds

    # Iterate over the dataset and extract statistics
    for sample in dataset[split]:
        total_segments += 1
        
        # Add speaker metadata to the set of unique speakers
        total_speakers.add(sample['speaker'])  # Use 'speaker' to track unique speakers

        # Assuming 'audio' contains the audio waveform and 'sampling_rate'
        audio_length_in_seconds = len(sample['audio']['array']) / sample['audio']['sampling_rate']
        total_duration += audio_length_in_seconds

    # Convert duration from seconds to hours
    total_duration_in_hours = total_duration / 3600
    
    return total_duration_in_hours, len(total_speakers), total_segments

# Calculate statistics for both train and test splits
train_duration, train_speakers, train_segments = calculate_statistics('train')
test_duration, test_speakers, test_segments = calculate_statistics('test')

# Print the statistics
print(f"Train Set - Duration: {train_duration:.2f} hours, Speakers: {train_speakers}, Segments: {train_segments}")
print(f"Test Set - Duration: {test_duration:.2f} hours, Speakers: {test_speakers}, Segments: {test_segments}")