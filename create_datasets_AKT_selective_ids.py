"""
Author: Antonia Jian
Date(Last modified): 01/08/2025
Description: 
This program is an extension of create_datasets_AKT_updated, with the following features:
    1. create flat huggingface Dataset (no train/test distinction)
    2. contain all segments from the selective IDs

"""

import os
import numpy as np
import pandas as pd
import librosa
from datasets import Dataset, Audio

def load_demographic_data(demographic_csv):
    df = pd.read_csv(demographic_csv)
    df = df.loc[:, ~df.columns.duplicated()]
    df['SpeakerID'] = pd.to_numeric(df['SpeakerID'], errors='coerce')
    df = df.dropna(subset=['SpeakerID'])
    return df[['SpeakerID', 'Gender', 'Age_yrs']].set_index('SpeakerID').T.to_dict()

def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    return [{"start_time": row['tmin'], "end_time": row['tmax'], "word": row['text']} for _, row in df.iterrows()]

def split_audio(wav_path, segments):
    audio_data, original_sr = librosa.load(wav_path, sr=44100)
    audio_data_16k = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)
    audio_data_list = []
    for segment in segments:
        start = int(segment["start_time"] * 16000)
        end = int(segment["end_time"] * 16000)
        audio_segment = audio_data_16k[start:end]
        audio_data_list.append({
            "audio": {
                "path": wav_path,
                "array": np.array(audio_segment, dtype=np.float32),
                "sampling_rate": 16000
            },
            "text": segment["word"]
        })
    return audio_data_list

def create_dataset(csv_path, wav_path, speaker_id, speaker_data):
    segments = read_csv(csv_path)
    audio_segments = split_audio(wav_path, segments)

    try:
        speaker_id_int = int(speaker_id)
    except ValueError:
        speaker_id_int = None

    speaker_info = speaker_data.get(speaker_id_int, {}) if speaker_id_int is not None else {}
    age = speaker_info.get("Age_yrs", None)
    gender = str(speaker_info.get("Gender", "Unknown"))

    data = {
        "audio": [seg["audio"] for seg in audio_segments],
        "text": [seg["text"] for seg in audio_segments],
        "speaker_id": [speaker_id_int] * len(audio_segments),
        "age": [age] * len(audio_segments),
        "gender": [gender] * len(audio_segments),
    }

    ds = Dataset.from_dict(data)
    return ds.cast_column("audio", Audio())

def process_to_flat_dataset(data_dir, demographic_csv, output_path, selected_ids):
    speaker_data = load_demographic_data(demographic_csv)
    datasets = []

    for sid in selected_ids:
        wav_file = os.path.join(data_dir, f"{sid}_task1.wav")
        csv_file = os.path.join(data_dir, f"{sid}_task1_kaldi.csv")
        if os.path.exists(wav_file) and os.path.exists(csv_file):
            print(f"Processing ID {sid}...")
            ds = create_dataset(csv_file, wav_file, sid, speaker_data)
            datasets.append(ds)
        else:
            print(f"Missing files for ID {sid}")

    if datasets:
        combined = datasets[0].concatenate(*datasets[1:]) if len(datasets) > 1 else datasets[0]
        combined.save_to_disk(output_path)
        print(f"Flat dataset saved to: {output_path}")
    else:
        print("No valid datasets were created.")

# Main Function
data_directory = "/srv/scratch/z5369417/AKT_data/"
demographic_csv = "/srv/scratch/z5369417/AKT_data_processing/AKT_demographic.csv"
output_path = "/srv/scratch/z5369417/created_dataset_AKT_selective_ids"
selected_ids = [786, 958, 1429]

process_to_flat_dataset(data_directory, demographic_csv, output_path, selected_ids)