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
from datasets import load_dataset, Dataset
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
        phonemes = row['phoneme'].strip().split()
        # Add the word and its phoneme mapping to the dictionary 
        phoneme_dict[word] = phonemes  
    return phoneme_dict

# Phonemize a given text using the Australian phoneme mapping
def phonemize_text(text, phoneme_dict):
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
    return phonemes

# Apply phonemization to the entire dataset
def phonemize_dataset(dataset, phoneme_dict):
    