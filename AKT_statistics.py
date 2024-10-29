import os
import pandas as pd
import numpy as np
import librosa
from collections import Counter

def load_demographic_data(demographic_csv):
    """Load demographic data and return a dictionary indexed by SpeakerID."""
    demographic_df = pd.read_csv(demographic_csv)
    demographic_df = demographic_df.loc[:, ~demographic_df.columns.duplicated()]
    demographic_df['SpeakerID'] = pd.to_numeric(demographic_df['SpeakerID'], errors='coerce')
    demographic_df = demographic_df.dropna(subset=['SpeakerID'])
    demographic_dict = demographic_df[['SpeakerID', 'Gender', 'Age_yrs']].set_index('SpeakerID').T.to_dict()
    return demographic_dict

def read_csv(csv_path):
    """Read the CSV file for a given speaker and extract intervals with word annotations."""
    df = pd.read_csv(csv_path)
    data = [{"start_time": row['tmin'], "end_time": row['tmax'], "word": row['text']} for _, row in df.iterrows()]
    return data

def split_audio(wav_path, segments):
    """Split the audio file into segments based on start and end times from the CSV."""
    audio_data, original_sr = librosa.load(wav_path, sr=44100)
    audio_data_16k = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)

    segment_durations = []
    for segment in segments:
        start_sample = int(segment["start_time"] * 16000)
        end_sample = int(segment["end_time"] * 16000)
        segment_audio = audio_data_16k[start_sample:end_sample]
        duration = len(segment_audio) / 16000
        segment_durations.append(duration)

    return segment_durations

def calculate_error_counts(speaker_id, annotation_df):
    """Calculate error counts for each segment based on 'Difference' columns in annotation data."""
    error_cols = [col for col in annotation_df.columns if 'Difference' in col]
    speaker_data = annotation_df[annotation_df['Child_ID'] == int(speaker_id)][error_cols]
    error_counts = speaker_data.notna().sum(axis=1).tolist()
    return error_counts

def calculate_statistics(data_dir, demographic_csv, annotation_file):
    demographic_data = load_demographic_data(demographic_csv)
    annotation_df = pd.read_excel(annotation_file)

    # Create dictionaries to hold paths of wav and csv files
    wav_files = {os.path.splitext(f)[0].replace('_task1', ''): os.path.join(data_dir, f) 
                 for f in os.listdir(data_dir) if f.endswith('_task1.wav')}
    csv_files = {os.path.splitext(f)[0].replace('_task1_kaldi', ''): os.path.join(data_dir, f) 
                 for f in os.listdir(data_dir) if f.endswith('_task1_kaldi.csv') and not f.endswith('_log.csv')}

    experiment_results = {}

    # Define experiment settings
    experiments = {
        "Exp14": {
            "remove_3_years": False, 
            "move_high_error_segments": False, 
            "exclude_high_error_segments": False, 
            "age_balanced_test_set": False
        },
        "Exp16": {
            "remove_3_years": True, 
            "move_high_error_segments": True, 
            "exclude_high_error_segments": False, 
            "age_balanced_test_set": False
        },
        "Exp17": {
            "remove_3_years": True, 
            "move_high_error_segments": False, 
            "exclude_high_error_segments": True, 
            "age_balanced_test_set": False
        },
        "Exp18": {
            "remove_3_years": True, 
            "move_high_error_segments": False, 
            "exclude_high_error_segments": False, 
            "age_balanced_test_set": True
        }
    }

    for exp_name, settings in experiments.items():
        total_train_duration, total_test_duration = 0, 0
        train_segments, test_segments = 0, 0
        train_error_count, test_error_count = 0, 0
        train_speakers, test_speakers = set(), set()

        for _, row in annotation_df.iterrows():
            speaker_id = str(row['Child_ID'])
            ssd_status = row['SSD']
            age = demographic_data.get(int(speaker_id), {}).get('Age_yrs', None)

            # Apply age-based filtering for experiments that exclude 3-year-olds
            if settings['remove_3_years'] and age == 3:
                print(f"Skipping 3-year-old speaker {speaker_id} in {exp_name}")
                continue

            # Verify if files for this speaker exist
            if speaker_id not in wav_files or speaker_id not in csv_files:
                print(f"Missing files for speaker {speaker_id}")
                continue

            wav_path = wav_files[speaker_id]
            csv_path = csv_files[speaker_id]
            segments = read_csv(csv_path)
            segment_durations = split_audio(wav_path, segments)
            error_counts = calculate_error_counts(speaker_id, annotation_df)
            high_error_segments = [i for i, count in enumerate(error_counts) if count >= 2]

            # Experiment-specific segment handling
            if settings['move_high_error_segments'] and ssd_status == 0:
                low_error_durations = [duration for i, duration in enumerate(segment_durations) if i not in high_error_segments]
                high_error_durations = [duration for i, duration in enumerate(segment_durations) if i in high_error_segments]
                
                total_train_duration += sum(low_error_durations)
                total_test_duration += sum(high_error_durations)
                train_segments += len(low_error_durations)
                test_segments += len(high_error_durations)
                
                train_error_count += sum([error_counts[i] for i in range(len(error_counts)) if i not in high_error_segments])
                test_error_count += sum([error_counts[i] for i in range(len(error_counts)) if i in high_error_segments])

            elif settings['exclude_high_error_segments'] and ssd_status == 0:
                low_error_durations = [duration for i, duration in enumerate(segment_durations) if i not in high_error_segments]
                
                total_train_duration += sum(low_error_durations)
                train_segments += len(low_error_durations)
                train_error_count += sum([error_counts[i] for i in range(len(error_counts)) if i not in high_error_segments])

            else:
                if ssd_status == 0:
                    total_train_duration += sum(segment_durations)
                    train_segments += len(segment_durations)
                    train_error_count += sum(error_counts)
                    train_speakers.add(speaker_id)
                else:
                    total_test_duration += sum(segment_durations)
                    test_segments += len(segment_durations)
                    test_error_count += sum(error_counts)
                    test_speakers.add(speaker_id)

        # Calculate and print demographics and age range
        def get_age_range(speakers):
            ages = [demographic_data[int(speaker)]['Age_yrs'] for speaker in speakers if demographic_data.get(int(speaker))]
            return (min(ages), max(ages)) if ages else (None, None)

        def get_gender_distribution(speakers):
            genders = [demographic_data[int(speaker)].get('Gender', 'Unknown') for speaker in speakers if demographic_data.get(int(speaker))]
            return Counter(genders)

        age_range_overall = get_age_range(train_speakers | test_speakers)
        age_range_train = get_age_range(train_speakers)
        age_range_test = get_age_range(test_speakers)

        gender_overall = get_gender_distribution(train_speakers | test_speakers)
        gender_train = get_gender_distribution(train_speakers)
        gender_test = get_gender_distribution(test_speakers)

        experiment_results[exp_name] = {
            "Total Dataset Size (number of speakers)": len(train_speakers | test_speakers),
            "Total Train Set Size (number of speakers)": len(train_speakers),
            "Total Test Set Size (number of speakers)": len(test_speakers),
            "Total Dataset Size (Segments)": train_segments + test_segments,
            "Total Train Set Size (Segments)": train_segments,
            "Total Test Set Size (Segments)": test_segments,
            "Total Segment Duration (Train)": total_train_duration,
            "Total Segment Duration (Test)": total_test_duration,
            "Average Error Count per Segment (Train)": train_error_count / train_segments if train_segments else 0,
            "Average Error Count per Segment (Test)": test_error_count / test_segments if test_segments else 0,
            "Age Range of Speakers (Overall)": age_range_overall,
            "Age Range of Speakers (Train)": age_range_train,
            "Age Range of Speakers (Test)": age_range_test,
            "Gender Distribution (Overall)": gender_overall,
            "Gender Distribution (Train)": gender_train,
            "Gender Distribution (Test)": gender_test
        }

    # Display the statistics for each experiment
    for exp_name, stats in experiment_results.items():
        print(f"--- Statistics for {exp_name} ---")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value}")
        print("\n")


    
# Define paths to demographic and annotation files and the raw data directory
demographic_csv = "/srv/scratch/z5369417/AKT_data_processing/AKT_demographic.csv"
annotation_file = "/srv/scratch/z5369417/AKT_data_processing/AKT_id2diagnosis.xlsx"
data_dir = "/srv/scratch/z5369417/AKT_data/"

# Run the function to calculate statistics for all experiments
calculate_statistics(data_dir, demographic_csv, annotation_file)