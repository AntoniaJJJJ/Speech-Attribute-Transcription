"""
Author: Antonia Jian
Date (Last Modified): 30/09/2025
Description:
This script processes AusKidTalk Task 3 data and creates a Hugging Face DatasetDict.

- Train split: children without SSD (SSD == 0)
- Test split: children with SSD (SSD == 1)
- For Task 3, we do NOT filter on hand-correction (all children are included if data exists).
- Audio is downsampled to 16kHz and segmented according to word-level annotations.

Output format:
DatasetDict({
    train: Dataset({
        features: ['audio', 'text', 'speaker_id', 'age', 'gender'],
        num_rows: <n>
    }),
    test: Dataset({
        features: ['audio', 'text', 'speaker_id', 'age', 'gender'],
        num_rows: <n>
    })
})
"""

import os
import pandas as pd
import numpy as np
import librosa
from datasets import DatasetDict, Dataset, Audio
from datasets import concatenate_datasets

# === Load demographic data ===
def load_demographic_data(demographic_csv):
    df = pd.read_csv(demographic_csv)
    df = df.loc[:, ~df.columns.duplicated()]
    df['SpeakerID'] = pd.to_numeric(df['SpeakerID'], errors='coerce')
    df = df.dropna(subset=['SpeakerID'])
    return df[['SpeakerID', 'Gender', 'Age_yrs']].set_index('SpeakerID').T.to_dict()

# === Read word-level annotation CSV ===
def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    return [{"start_time": row['tmin'], "end_time": row['tmax'], "word": row['text']} for _, row in df.iterrows()]

# === Split audio into segments ===
def split_audio(wav_path, segments):
    audio_data, original_sr = librosa.load(wav_path, sr=44100)  # load original at 44.1kHz
    audio_data_16k = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)

    out = []
    for seg in segments:
        start_sample = int(seg["start_time"] * 16000)
        end_sample = int(seg["end_time"] * 16000)
        segment_audio = audio_data_16k[start_sample:end_sample]
        out.append({
            "audio": {
                "path": wav_path,
                "array": np.array(segment_audio, dtype=np.float32),
                "sampling_rate": 16000
            },
            "text": seg["word"]
        })
    return out

# === Build dataset for one speaker ===
def create_dataset(csv_path, wav_path, speaker_id, speaker_data):
    segments = read_csv(csv_path)
    try:
        speaker_id = int(speaker_id)
    except ValueError:
        speaker_id = None

    speaker_info = speaker_data.get(speaker_id, {}) if speaker_id is not None else {}
    age = speaker_info.get("Age_yrs", None)
    gender = str(speaker_info.get("Gender", "Unknown"))

    audio_segments = split_audio(wav_path, segments)

    data = {
        "audio": [seg["audio"] for seg in audio_segments],
        "text": [seg["text"] for seg in audio_segments],
        "speaker_id": [speaker_id] * len(audio_segments),
        "age": [age] * len(audio_segments),
        "gender": [gender] * len(audio_segments),
    }
    dataset = Dataset.from_dict(data)
    return dataset.cast_column("audio", Audio())

# === Main function ===
def create_dataset_dict_AKT_task3(data_dir, demographic_csv, annotation_file, output_dir):
    speaker_data = load_demographic_data(demographic_csv)
    annotation_df = pd.read_excel(annotation_file)

    # Note: For Task 3, we do NOT filter on T3_Handcorrection_Completed
    wav_files = {os.path.splitext(f)[0].replace('_task3', ''): os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if f.endswith('_task3.wav')}
    csv_files = {os.path.splitext(f)[0].replace('_task3_child', ''): os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if f.endswith('_task3_child.csv') and not f.endswith('_log.csv')}

    common = set(wav_files.keys()).intersection(csv_files.keys())

    train_datasets, test_datasets = [], []
    processed_ids, missing_ids = [], []

    for _, row in annotation_df.iterrows():
        sid = str(row['Child_ID'])
        if sid in common:
            wav_path, csv_path = wav_files[sid], csv_files[sid]
            ds = create_dataset(csv_path, wav_path, sid, speaker_data)

            if row['SSD'] == 0:
                train_datasets.append(ds)
            else:
                test_datasets.append(ds)

            processed_ids.append(sid)
        else:
            missing_ids.append(sid)

    dataset_dict = DatasetDict()
    if train_datasets:
        dataset_dict['train'] = concatenate_datasets(train_datasets)
    if test_datasets:
        dataset_dict['test'] = concatenate_datasets(test_datasets)

    dataset_dict.save_to_disk(output_dir)

    print(f"Saved dataset to {output_dir}")
    print(f"Processed IDs: {processed_ids}")
    print(f"Missing data IDs: {missing_ids}")

# === Example run ===
if __name__ == "__main__":
    data_directory = "/srv/scratch/z5369417/AKT_data_task3"
    demographic_csv = "/srv/scratch/z5369417/AKT_data_processing/AKT_demographic.csv"
    annotation_file = "/srv/scratch/z5369417/AKT_data_processing/AKT_id2diagnosis.xlsx"
    output_directory = "/srv/scratch/z5369417/created_dataset_0930/AKT_dataset_task3"
    create_dataset_dict_AKT_task3(data_directory, demographic_csv, annotation_file, output_directory)