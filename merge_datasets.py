from datasets import load_from_disk, concatenate_datasets, DatasetDict, Value
import datasets
import pandas as pd

# Load original datasets
cu = load_from_disk('/srv/scratch/z5369417/outputs/phonemization2_remove/cu_remove_adult_noise')
akt = load_from_disk('/srv/scratch/z5369417/outputs/phonemization_AKT')

# === Load CU ARPA → IPA mapping ===
cu_mapping = pd.read_csv('data/Phoneme2att_camb_att_Diph_v1.csv')
arpa_to_ipa = dict(zip(cu_mapping["Phoneme_arpa"], cu_mapping["Phoneme_ipa"]))

# === Function to convert ARPA string to IPA string ===
def convert_arpa_to_ipa(phoneme_str):
    return " ".join(arpa_to_ipa.get(p, p) for p in phoneme_str.split())

# === Apply conversion to CU ===
cu = cu.map(lambda x: {"phoneme_combined": convert_arpa_to_ipa(x["phoneme_cmu"])})

# Rename phoneme column to be consistent across both
akt = akt.rename_column('phoneme_akt', 'phoneme_combined')

# Harmonize data types
cu = cu.cast_column('speaker_id', Value('string'))
akt = akt.cast_column('speaker_id', Value('string'))

cu = cu.cast_column('age', Value('string'))
akt = akt.cast_column('age', Value('string'))

# Merge by split
train = concatenate_datasets([cu['train'], akt['train']])
validation = concatenate_datasets([cu['valid'], akt['test']])
test = concatenate_datasets([cu['test'], akt['test']])

# Save merged dataset
combined = DatasetDict({
    'train': train,
    'valid': validation,
    'test': test,
})
combined.save_to_disk('/srv/scratch/z5369417/outputs/phonemization_combined_CU_AKT_merged')