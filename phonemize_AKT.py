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
import os
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
        phonemes = row['transcription'].strip()
        # Add the word and its phoneme mapping to the dictionary 
        phoneme_dict[word] = phonemes  
    return phoneme_dict


# Load HCE phoneme list from the provided Excel chart (columns contain phonemes)
def load_hce_phonemes(file_path):
     hce_phonemes_df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)  # No header in the sheet
    # Extract the first column (Column A) containing the phonemes
     phonemes = hce_phonemes_df.iloc[:, 0].dropna().tolist()  # Drop any NaN values

     return phonemes

# Phonemize a given text using the Australian phoneme mapping
def phonemize_text(text, phoneme_dict, hce_phonemes, unknown_words):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()  # Normalize text and split into words
    phonemes_list = []
    
    for word in words:
        # Debug: Track word processing
        print(f"Processing word: {word}")

        if word in phoneme_dict:
            transcription = phoneme_dict[word]
            pattern = '|'.join([re.escape(p) for p in hce_phonemes])
            separated_phonemes = re.findall(pattern, transcription)
            
            if separated_phonemes:
                phonemes_list.append(' '.join(separated_phonemes))  # Join phonemes with spaces
                # Debug: Show phonemized word
                print(f"Phonemized word: {word} -> {' '.join(separated_phonemes)}")
            else:
                phonemes_list.append("UNK")  # Phonemization failed
                unknown_words.add(word)  # Add to unknown words
                # Debug: Failed phonemization
                print(f"Failed to phonemize word: {word} (Added to unknown words)")
        else:
            phonemes_list.append("UNK")
            unknown_words.add(word)  # Track unknown words
            # Debug: Word not found in phoneme_dict
            print(f"Word not found in phoneme_dict: {word} (Added to unknown words)")
    
    return ' '.join(phonemes_list)

# Process text to handle compound words before phonemization
def process_compound_words(text, phoneme_dict, hce_phonemes, unknown_words):
    # Debug: Start processing compound word
    print(f"Processing compound word: {text}")
    
    if "_o_clock" in text:
        components = text.split('_o_clock')
        components.append("o'clock")  # Add "o'clock" as the second part
        print(f"Split '{text}' into: {components}")

        # Handle apostrophes in "o'clock"
        if "o'clock" not in phoneme_dict and "o’clock" in phoneme_dict:
            components[-1] = "o’clock"  # Use curly apostrophe if straight one is missing
            print(f"Using curly apostrophe for 'o'clock'")
    else:
        components = text.split('_')

    phonemized_components = []
    for component in components:
        component = component.strip()  # Clean up component
        print(f"Processing component: {component}")

        if component in phoneme_dict:
            transcription = phoneme_dict[component]
            pattern = '|'.join([re.escape(p) for p in hce_phonemes])
            separated_phonemes = re.findall(pattern, transcription)
            
            if separated_phonemes:
                phonemized_components.append(' '.join(separated_phonemes))
                print(f"Phonemized component: {component} -> {' '.join(separated_phonemes)}")
            else:
                print(f"Failed to phonemize component: {component}")
                return None  # Component not found, mark as unknown
        else:
            print(f"Component not found in phoneme_dict: {component}")
            unknown_words.add(text)  # Add compound word to unknown if a part fails
            return None  # Component not found, mark as unknown
    
    return ' '.join(phonemized_components)  # Join phonemes of all components

# Phonemize the dataset and process unknown words
def phonemize_dataset(dataset, phoneme_dict, hce_phonemes, unknown_words):
    def apply_phonemization(batch):
        phonemes_akt = []
        filtered_batch = {key: [] for key in batch.keys()}  # Initialize filtered batch

        for i, text in enumerate(batch['text']):
            # Check for compound words first
            phonemized_text = process_compound_words(text, phoneme_dict, hce_phonemes, unknown_words)
            
            if phonemized_text is not None:
                # If the compound word was successfully phonemized, use that result
                phonemes_akt.append(phonemized_text)
                for key in batch.keys():
                    filtered_batch[key].append(batch[key][i])  # Keep only rows that are phonemized
            else:
                # If the word is 'UNK', it will be added to unknown and removed
                phonemized_single_word = phonemize_text(text, phoneme_dict, hce_phonemes, unknown_words)
                if phonemized_single_word != "UNK":  # If not UNK, keep the row
                    phonemes_akt.append(phonemized_single_word)
                    for key in batch.keys():
                        filtered_batch[key].append(batch[key][i])
        
        filtered_batch['phoneme_akt'] = phonemes_akt
        return filtered_batch

    phonemized_dataset = dataset.map(apply_phonemization, batched=True) 
    return phonemized_dataset

# Save unknown words to a text file
def save_unknown_words(unknown_words, file_path):
    with open(file_path, 'w') as f:
        for word in sorted(unknown_words):
            f.write(f"{word}\n")

# Main function to handle dataset loading, phonemization, and saving the output
def main(akt_dataset_path, phoneme_mapping_file, hce_phonemes_file, output_path, unknown_words_file):
     # Load the AKT dataset from Hugging Face
    dataset = load_from_disk(akt_dataset_path)

    # Load the Australian English word-to-phoneme mapping from the transcription sheet
    phoneme_dict = load_phoneme_mapping(phoneme_mapping_file)

    # Load the HCE phonemes to help with splitting the phoneme strings
    hce_phonemes = load_hce_phonemes(hce_phonemes_file)

     # Set to track unknown words
    unknown_words = set()
    
    # Phonemize the 'train' split of the dataset
    phonemized_dataset = phonemize_dataset(dataset['train'], phoneme_dict, hce_phonemes, unknown_words)
    
    # Create directory structure as per the required format
    train_output_path = os.path.join(output_path, 'train')
    os.makedirs(train_output_path, exist_ok=True)

    # Save the phonemized dataset to disk
    phonemized_dataset.save_to_disk(train_output_path)

    # Save the unknown words to a file
    save_unknown_words(unknown_words, os.path.join(output_path, 'unknown_words.txt'))

phoneme_mapping_file = '/srv/scratch/z5369417/AKT_data_processing/AusKidTalk_transcription.xlsx'  # Path to the Excel file
hce_phonemes_file = '/srv/scratch/z5369417/AKT_data_processing/HCE_phonemes.xlsx'  # Path to the HCE phonemes file
akt_dataset_path = '/srv/scratch/z5369417/created_dataset_0808/AKT_dataset'  # Path to the AKT dataset
output_path = '/srv/scratch/z5369417/outputs/phonemization_AKT'  # Path to save the phonemized dataset
unknown_words_file = '/srv/scratch/z5369417/outputs/phonemization_AKT/unknown_words.txt'  # Path to save the unknown words
main(akt_dataset_path, phoneme_mapping_file, hce_phonemes_file, output_path, unknown_words_file)