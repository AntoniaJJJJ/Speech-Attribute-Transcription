import pandas as pd
from datasets import load_from_disk, DatasetDict

# 1) Paths
mapping_csv  = "/srv/scratch/z5369417/Speech-Attribute-Transcription/data/Phoneme2att_camb_att_Diph_v1.csv"
dataset_path = "/srv/scratch/z5369417/outputs/phonemization_speechocean_exp11_4"
output_path  = "/srv/scratch/z5369417/outputs/phonemization_speechocean_exp11_4_ipa"

# 2) Load Arpabet→IPA mapping
mapping_df = pd.read_csv(mapping_csv)
arpa2ipa   = mapping_df.set_index("Phoneme_arpa")["Phoneme_ipa"].to_dict()

# 3) Load your dataset
ds = load_from_disk(dataset_path)

# 4) Fields to convert
seq_fields = ["phoneme_speechocean", "actual_spoken_phonemes"]
#    (add any other space‐delimited phoneme fields here)

def to_ipa(batch):
    out = {}
    for fld in seq_fields:
        seqs = batch[fld]
        ipa_seqs = [
            " ".join(arpa2ipa.get(tok, tok) for tok in seq.split())
            for seq in seqs
        ]
        out[fld] = ipa_seqs
    return out

# 5) Apply mapping with a safe batch size
B = 500
if isinstance(ds, DatasetDict):
    for split in ds:
        ds[split] = ds[split].map(to_ipa, batched=True, batch_size=B)
else:
    ds = ds.map(to_ipa, batched=True, batch_size=B)

# 6) Save the IPA‐riched dataset
ds.save_to_disk(output_path)
print("Converted dataset saved to:", output_path)