import pandas as pd

# Function to get list of children with error counts >= 2
def get_high_error_children(data_dir):
    # Get all CSV files for processing
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv') and not f.endswith('_log.csv')]
    
    high_error_children = []
    
    for csv_path in csv_files:
        # Retrieve error counts from each CSV file
        _, error_counts = read_csv(csv_path)
        
        # Check if any error count is >= 2
        if any(error >= 2 for error in error_counts):
            # Extract the speaker ID from the filename (assuming filename contains SpeakerID)
            speaker_id = os.path.splitext(os.path.basename(csv_path))[0].replace('_task1_kaldi', '')
            high_error_children.append(speaker_id)
    
    return high_error_children

# Example usage
data_directory = "/srv/scratch/z5369417/AKT_data/"  # Directory containing the CSV files
children_with_high_errors = get_high_error_children(data_directory)
print("Children with errors >= 2:", children_with_high_errors)