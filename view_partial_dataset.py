import os
import pyarrow.parquet as pq
import pyarrow as pa

# Path to the folder containing the .arrow files
arrow_files_dir = "/srv/scratch/z5369417/created_dataset_0808/AKT_dataset/train"

# Function to manually load .arrow files and display the partial dataset
def load_partial_arrow_files(arrow_files_dir):
    # List all .arrow files
    arrow_files = [f for f in os.listdir(arrow_files_dir) if f.endswith('.arrow')]

    # Load and concatenate the data from the .arrow files
    all_data = []
    for arrow_file in arrow_files:
        file_path = os.path.join(arrow_files_dir, arrow_file)
        try:
            table = pq.read_table(file_path)
            all_data.append(table)
        except Exception as e:
            print(f"Error loading {arrow_file}: {e}")  # Skip corrupted files

    # Concatenate all tables into a single table
    if all_data:
        combined_data = pa.concat_tables(all_data)
        
        # Convert to pandas DataFrame for easier viewing
        df = combined_data.to_pandas()

        # Write the DataFrame to stdout (which will be captured in the redirected output file)
        df.to_string(index=False)  # Disable index for cleaner output

        # Output the full DataFrame, ensuring all rows are printed
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df.to_string(index=False))
    else:
        print("No valid data found in the specified directory.")

# Call the function to load and view partial data
load_partial_arrow_files(arrow_files_dir)