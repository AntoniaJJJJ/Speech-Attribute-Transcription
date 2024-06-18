import os
import io
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio, Features, Value
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
                
                # Read the raw audio file and convert it to a .wav-like array
                array, sr = read_raw_file(audio_path, SAMPLE_RATE)
                
                # Append the data
                data["audio"].append({"path": audio_path, "array": array, "sampling_rate": sr})
                data["transcription"].append(transcription)

# Create a DatasetDict
features = Features({
    "audio": Audio(sampling_rate=SAMPLE_RATE),
    "transcription": Value("string")
})

dataset = DatasetDict({
    "train": Dataset.from_dict({"audio": data["audio"], "transcription": data["transcription"]})
})

# Verify the dataset
print(dataset["train"][0])

# Create a DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_dict({"audio": data["audio"], "transcription": data["transcription"]}, features=features)
})

# Verify the dataset
print(dataset["train"][0])