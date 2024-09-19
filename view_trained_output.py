import os
from datasets import load_from_disk

# Define the path to the dataset folders
result_db_path = '/srv/scratch/z5369417/outputs/trained_result/cu/exp2/results_test_valid.db' 
valid_data_path = os.path.join(result_db_path, 'valid')
test_data_path = os.path.join(result_db_path, 'test')

# Load the datasets
try:
    valid_dataset = load_from_disk(valid_data_path)
    test_dataset = load_from_disk(test_data_path)
    
    # Define the output text file
    output_file_path = '/srv/scratch/z5369417/outputs/trained_result/cu/exp2/results_test_valid.db/data_after_training.txt'  

    with open(output_file_path, 'w') as file:
        file.write("VALID DATASET:\n")
        for idx, example in enumerate(valid_dataset):
            file.write(f"Example {idx}:\n{example}\n\n")
        
        file.write("\nTEST DATASET:\n")
        for idx, example in enumerate(test_dataset):
            file.write(f"Example {idx}:\n{example}\n\n")

    print(f"Data successfully written to {output_file_path}")

except Exception as e:
    print(f"An error occurred while loading the datasets: {e}")