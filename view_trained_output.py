import os
from datasets import load_from_disk

# Define the path to the dataset folders
result_db_path = '/srv/scratch/z5369417/outputs/trained_result/AKT/exp16/results_test.db' 
valid_data_path = os.path.join(result_db_path, 'valid')
test_data_path = os.path.join(result_db_path, 'test')

# Load the datasets
try:
    valid_dataset = load_from_disk(valid_data_path)
    test_dataset = load_from_disk(test_data_path)
    
    # Define the output text files
    #valid_output_file_path = '/srv/scratch/z5369417/outputs/trained_result/cu/exp9/results_test_valid.db/valid_after_training.txt' 
    test_output_file_path = '/srv/scratch/z5369417/outputs/trained_result/cu/exp9/results_test_valid.db/test_after_training.txt' 

    # Write the 'valid' dataset to a text file
    #with open(valid_output_file_path, 'w') as valid_file:
    #    valid_file.write("VALID DATASET:\n")
    #    for idx, example in enumerate(valid_dataset):
    #        valid_file.write(f"Example {idx}:\n{example}\n\n")
    
    # Write the 'test' dataset to a text file
    with open(test_output_file_path, 'w') as test_file:
        test_file.write("TEST DATASET:\n")
        for idx, example in enumerate(test_dataset):
            test_file.write(f"Example {idx}:\n{example}\n\n")

    #print(f"Data successfully written to {valid_output_file_path} and {test_output_file_path}")
    print(f"Data successfully written to {test_output_file_path}")

except Exception as e:
    print(f"An error occurred while loading the datasets: {e}")