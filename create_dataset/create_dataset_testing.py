import os
import soundfile as sf
from datasets import DatasetDict, Audio, Features, Value
from tqdm import tqdm

# Define the path to your dataset
dataset_path = "/srv/scratch/speechdata/children/TD/CU/data"

# Initialize lists to store data
data = {"audio": [], "transcription": []}

# Specify the sample rate of the .raw files
SAMPLE_RATE = 16000  # Change this to your actual sample rate

# Function to read .raw file and convert to a .wav format-like structure
def read_raw_file(file_path, sample_rate):
    with open(file_path, 'rb') as f:
        data = f.read()
    # Assuming 16-bit PCM encoding, little-endian
    return sf.read(io.BytesIO(data), samplerate=sample_rate, format='RAW', subtype='PCM_16', endian='LITTLE')

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
                with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                    transcription = f.read().strip()
                
                # Append the data
                data["audio"].append(audio_path)
                data["transcription"].append(transcription)

# Define the dataset features with custom decoding function
def decode_raw_audio(example):
    audio_path = example["audio"]
    array, sr = read_raw_file(audio_path, SAMPLE_RATE)
    return {"audio": {"array": array, "sampling_rate": sr}}

features = Features({
    "audio": Audio(decode=decode_raw_audio),
    "transcription": Value("string")
})

# Create a DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict(data, features=features)
})

# Verify the dataset
print(dataset["train"][0])