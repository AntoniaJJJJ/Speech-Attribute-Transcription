import os
import pandas as pd

# Function to read a CSV file and compute total error count for a speaker
def compute_total_error(csv_path):
    df = pd.read_csv(csv_path)
    error_cols = [col for col in df.columns if 'Difference' in col]
    df['error_count'] = df[error_cols].notna().sum(axis=1)
    return df['error_count'].sum()

# Main function to process CSV files and compute error counts
def generate_error_report(data_dir, output_excel):
    speaker_error_counts = []

    # Retrieve all relevant CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv')]

    for csv_file in csv_files:
        # Extract the speaker ID from the filename
        speaker_id = csv_file.replace('_task1_kaldi.csv', '')

        # Define path for the CSV file
        csv_path = os.path.join(data_dir, csv_file)

        # Compute total error count for the speaker
        total_error_count = compute_total_error(csv_path)

        # Append to the results list
        speaker_error_counts.append({'SpeakerID': speaker_id, 'TotalErrorCount': total_error_count})

    # Create a DataFrame and save to an Excel file
    error_df = pd.DataFrame(speaker_error_counts)
    error_df.to_excel(output_excel, index=False)

    return error_df

# Example usage
data_directory = "/srv/scratch/z5369417/AKT_data/"  # Replace with your directory path
output_excel_path = "speaker_error_report.xlsx"

# Generate the error report and save to an Excel file
generate_error_report(data_directory, output_excel_path)