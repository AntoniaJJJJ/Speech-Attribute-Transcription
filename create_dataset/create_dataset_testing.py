import os
from datasets import Dataset, DatasetDict, Audio, Features, Value
from tqdm import tqdm

# Define the path to your dataset
dataset_path = "/srv/scratch/speechdata/children/TD/CU/data"

# Initialize lists to store data
data = {"audio": [], "transcription": []}

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
                try:
                    # Try reading the transcription with utf-8 encoding
                    with open(text_path, "r", encoding="utf-8") as f:
                        transcription = f.read().strip()
                except UnicodeDecodeError:
                    # If utf-8 fails, try reading with latin-1 encoding
                    with open(text_path, "r", encoding="latin-1") as f:
                        transcription = f.read().strip()
                
                # Append the data
                data["audio"].append(audio_path)
                data["transcription"].append(transcription)

# Define the dataset features
features = Features({
    "audio": Audio(sampling_rate=16000),
    "transcription": Value("string")
})

# Create a DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict({"audio": data["audio"], "transcription": data["transcription"]}, features=features)
})

# Verify the dataset
print(dataset["train"][0])
