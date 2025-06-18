from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load original datasets
cu = load_from_disk('/path/to/CU_dataset')
akt = load_from_disk('/path/to/AKT_dataset')

# Rename phoneme column to be consistent across both
cu = cu.rename_column('phoneme_cu', 'phoneme_combined')
akt = akt.rename_column('phoneme_akt', 'phoneme_combined')

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