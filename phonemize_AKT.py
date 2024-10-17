"""
Author: Antonia Jian
Date (Last modified): 17/10/2024
Description:
This script phonemizes AKT (Australian English) dataset using a phoneme mapping 
chart. It processes the dataset using the Hugging Face 
'datasets' library and adds phonemic representations of the text data.

The script reads the phoneme mappings from the Excel file, uses them to convert transcriptions 
into phonemes, and outputs the phonemized dataset in a DatasetDict format.

The output dataset will have the following structure:

DatasetDict({
    train: Dataset({
        features: ['audio', 'text', 'speaker_id', 'age', 'phoneme'],
        num_rows: <number_of_rows>
    }),
})
- 'audio': Contains audio data as a NumPy array and sampling rate
- 'text': The corresponding transcription of the audio
- 'speaker_id': The ID of the speaker
- 'age': The age of the speaker
- 'phoneme': The phonemic representation of the transcription based on the custom mapping

For load_phoneme_mapping(file_path):
   Reads the phoneme mapping from an Excel file. Specifically, it accesses the 'transcription' sheet 
   and constructs a dictionary mapping each word to its corresponding phoneme sequence.

For phonemize_text(text, phoneme_dict):
   Converts a given text (transcription) into its phonemic representation based on the provided 
   phoneme mapping.

   Output:
   - phonemes: A list of phonemes corresponding to the words in the input text. Words not found in the 
     phoneme dictionary are represented as "[UNK]".

For phonemize_dataset(dataset, phoneme_dict):
   Applies phonemization to an entire dataset of text transcriptions using the phoneme mapping.
   
For main(akt_dataset_path, phoneme_mapping_file, output_path):
   This is the main function that loads the AKT dataset from Hugging Face, applies phonemization using 
   the Australian English phoneme mapping, and saves the phonemized dataset.
"""
import pandas as pd
import re
from datasets import load_dataset, Dataset, load_from_disk
from collections import defaultdict

# Load the phoneme mapping from the 'transcription' sheet in the Excel file
def load_phoneme_mapping(file_path):
    # Load the specific sheet
    df = pd.read_excel(file_path, sheet_name='transcription')  
    # Create a default dictionary for the phoneme mapping
    phoneme_dict = defaultdict(list)  
    
    # Iterate through each row of the dataframe, extracting word and phoneme
    for _, row in df.iterrows():
        # Normalize the word (lowercase, no extra spaces)
        word = row['word'].strip().lower()  
         # Split the phonemes (space-separated)
        phonemes = row['transcription'].strip().split()
        # Add the word and its phoneme mapping to the dictionary 
        phoneme_dict[word] = phonemes  
    return phoneme_dict

# Phonemize a given text using the Australian phoneme mapping
def phonemize_text(text, phoneme_dict, unknown_words):
    # Remove punctuation and lowercase the text for normalization, then split into words
    words = re.sub(r'[^\w\s]', '', text.lower()).split()  # Removes any non-alphabetical characters
    # Initialize an empty list to store phonemes
    phonemes = []  
    
    # Convert each word into its phonemic representation
    for word in words:
        if word in phoneme_dict:
            phonemes.extend(phoneme_dict[word])  # Add phonemes for the known word
        else:
            phonemes.append(f"[UNK]")  # If the word is not in the phoneme dictionary, mark it as unknown
            unknown_words.add(word)  # Track unknown words
    return phonemes

# Save unknown words to a text file
def save_unknown_words(unknown_words, file_path):
    with open(file_path, 'w') as f:
        for word in sorted(unknown_words):
            f.write(f"{word}\n")

# Apply phonemization to the entire dataset
def phonemize_dataset(dataset, phoneme_dict, unknown_words):
     # Define the function to be applied to each batch of text entries
    def apply_phonemization(batch):
        # Apply the phonemize_text function to each text entry in the batch
        batch['phoneme'] = [phonemize_text(text, phoneme_dict, unknown_words) for text in batch['text']]
        return batch
    
    # Apply the phonemization function to the entire dataset in batches
    phonemized_dataset = dataset.map(apply_phonemization, batched=True)
    return phonemized_dataset

# Main function to handle dataset loading, phonemization, and saving the output
def main(akt_dataset_path, phoneme_mapping_file, output_path, unknown_words_file):
     # Load the AKT dataset from Hugging Face
    dataset = load_from_disk(akt_dataset_path)

    # Load the Australian English word-to-phoneme mapping from the transcription sheet
    phoneme_dict = load_phoneme_mapping(phoneme_mapping_file)

     # Set to track unknown words
    unknown_words = set()
    
    # Phonemize the 'train' split of the dataset
    phonemized_dataset = phonemize_dataset(dataset['train'], phoneme_dict, unknown_words)
    
    # Save the phonemized dataset to disk
    phonemized_dataset.save_to_disk(output_path)

    # Save the unknown words to a file
    save_unknown_words(unknown_words, unknown_words_file)

phoneme_mapping_file = '/srv/scratch/z5369417/AKT_data_processing/AusKidTalk_transcription.xlsx'  # Path to the Excel file
akt_dataset_path = '/srv/scratch/z5369417/created_dataset_0808/AKT_dataset'  # Path to the AKT dataset
output_path = '/srv/scratch/z5369417/outputs/phonemization_AKT'  # Path to save the phonemized dataset
unknown_words_file = '/srv/scratch/z5369417/outputs/phonemization_AKT/unknown_words.txt'  # Path to save the unknown words
main(akt_dataset_path, phoneme_mapping_file, output_path, unknown_words_file)