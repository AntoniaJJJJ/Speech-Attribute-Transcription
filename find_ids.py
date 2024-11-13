import os
import pandas as pd

# Function to read a CSV file and count errors
def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    error_cols = [col for col in df.columns if 'Difference' in col]
    df['error_count'] = df[error_cols].notna().sum(axis=1)
    return df['error_count']

# Main function to get children IDs with error_count >= 2 and matching WAV files
def get_children_with_high_errors(data_dir):
    high_error_ids = []

    # Retrieve all relevant CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv')]

    for csv_file in csv_files:
        # Extract the speaker ID from the filename
        speaker_id = csv_file.replace('_task1_kaldi.csv', '')
        
        # Define paths for the CSV and WAV files
        csv_path = os.path.join(data_dir, csv_file)
        wav_path = os.path.join(data_dir, f"{speaker_id}_task1.wav")
        
        # Check if the corresponding WAV file exists
        if os.path.exists(wav_path):
            # Get error counts from CSV
            error_counts = read_csv(csv_path)
            
            # If any error count >= 2, add speaker_id to the list
            if any(error >= 2 for error in error_counts):
                high_error_ids.append(speaker_id)

    return high_error_ids

# Example usage
data_directory = "/srv/scratch/z5369417/AKT_data/"
children_with_high_errors = get_children_with_high_errors(data_directory)
print("Children with errors >= 2 and matching WAV files:", children_with_high_errors)
