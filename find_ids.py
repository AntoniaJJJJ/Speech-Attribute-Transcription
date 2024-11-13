import os
import pandas as pd

# Function to read a CSV file and count errors
def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    error_cols = [col for col in df.columns if 'Difference' in col]
    df['error_count'] = df[error_cols].notna().sum(axis=1)
    return df['SpeakerID'], df['error_count']

# Main function to get children IDs with error_count >= 2 and matching WAV files
def get_children_with_high_errors(data_dir):
    high_error_ids = []

    # Retrieve all relevant CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        
        # Check if the corresponding WAV file exists
        speaker_id_base = csv_file.replace('_task1_kaldi.csv', '')
        wav_path = os.path.join(data_dir, f"{speaker_id_base}_task1.wav")
        
        if os.path.exists(wav_path):
            # Get SpeakerID and error counts from CSV
            speaker_ids, error_counts = read_csv(csv_path)
            
            # Append IDs with error_count >= 2 to the list
            high_error_ids.extend(speaker_ids[error_counts >= 2].tolist())

    return high_error_ids

# Example usage
data_directory = "/srv/scratch/z5369417/AKT_data/"
children_with_high_errors = get_children_with_high_errors(data_directory)
print("Children with errors >= 2 and matching WAV files:", children_with_high_errors)