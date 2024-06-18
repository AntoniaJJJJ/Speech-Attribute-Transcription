import os
from datasets import Dataset, DatasetDict, Audio, Features, Value
from tqdm import tqdm

# Define the path to dataset
dataset_path = "/srv/scratch/speechdata/children/TD/CU/data"

# Initialize lists to store data
audio_files = []
transcriptions = []

# Walk through the directory structure
for root, dirs, files in tqdm(os.walk(dataset_path)):
    for file_name in files:
        if file_name.endswith(".raw"):
            # Get the base name without the extension
            base_name = os.path.splitext(file_name)[0]
            
            # Define paths to the audio and text files
            audio_path = os.path.join(root, file_name)
            text_path = os.path.join(root, base_name + ".txt")
            
            # Check if the transcription file exists
            if os.path.exists(text_path):
                # Read the transcription
                with open(text_path, "r") as f:
                    transcription = f.read().strip()
                
                # Append the data
                audio_files.append(audio_path)
                transcriptions.append(transcription)

# Define the dataset features
features = Features({
    "audio": Audio(sampling_rate=16000),
    "transcription": Value("string")
})

# Create a DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict({"audio": audio_files, "transcription": transcriptions}, features=features)
})

# Verify the dataset
print(dataset["train"][0])