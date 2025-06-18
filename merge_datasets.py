from datasets import load_from_disk, concatenate_datasets, DatasetDict, Value
import datasets

# Load original datasets
cu = load_from_disk('/srv/scratch/z5369417/outputs/phonemization2_remove/cu_remove_adult_noise')
akt = load_from_disk('/srv/scratch/z5369417/outputs/phonemization_AKT')

# Rename phoneme column to be consistent across both
cu = cu.rename_column('phoneme_cmu', 'phoneme_combined')
akt = akt.rename_column('phoneme_akt', 'phoneme_combined')

# Harmonize data types
cu = cu.cast_column('speaker_id', Value('string'))
akt = akt.cast_column('speaker_id', Value('string'))

cu = cu.cast_column('age', Value('string'))
akt = akt.cast_column('age', Value('string'))

# Merge by split
train = concatenate_datasets([cu['train'], akt['train']])
validation = concatenate_datasets([cu['validation'], akt['validation']])
test = concatenate_datasets([cu['test'], akt['test']])

# Save merged dataset
combined = DatasetDict({
    'train': train,
    'validation': validation,
    'test': test,
})
combined.save_to_disk('/srv/scratch/z5369417/outputs/phonemization_combined_CU_AKT_merged')