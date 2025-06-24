from datasets import load_from_disk, DatasetDict
from collections import defaultdict

# Load original AKT dataset
akt = load_from_disk('/srv/scratch/z5369417/outputs/phonemization_AKT')

# Define exclusion list
excluded_ids = {170, 264, 413, 440, 608, 786, 958, 982,
                1226, 1327, 1429, 1001, 1075, 466, 516, 899}

# Gather all speaker_ids in train split
speaker_ids_in_train = set(x['speaker_id'] for x in akt['train'])

# Determine which IDs exist and which don’t
found_ids = sorted(excluded_ids & speaker_ids_in_train)
not_found_ids = sorted(excluded_ids - speaker_ids_in_train)

print("Speaker IDs to exclude:")
print(f"Found and excluded: {found_ids}")
print(f"Not found in AKT train set: {not_found_ids}")

# Filter training data
akt_filtered_train = akt['train'].filter(lambda x: x['speaker_id'] not in excluded_ids)

# Repackage into new dataset
akt_filtered = DatasetDict({
    'train': akt_filtered_train,
    'test': akt['test']
})

# Save to new path
akt_filtered.save_to_disk('/srv/scratch/z5369417/outputs/phonemization_AKT_Mos')