import re
from datasets import load_dataset

# Load the original SpeechOcean dataset
ds = load_dataset("mispeech/speechocean762")

# Variables to track occurrences
found_ar = False
found_ir = False
found_ar_mispronounced = False
found_ir_mispronounced = False

# Function to remove stress markers from vowels (e.g., AR1 -> AR, IR0 -> IR)
def remove_stress(phoneme):
    return re.sub(r'([a-z]+)[0-2]$', r'\1', phoneme.lower())  # Convert to lowercase and remove final digit

# Function to check phonemes in both canonical and mispronunciation fields
def check_phonemes(sample):
    global found_ar, found_ir, found_ar_mispronounced, found_ir_mispronounced

    for word in sample["words"]:
        # Check canonical phonemes
        for phoneme in word["phones"]:
            phoneme = remove_stress(phoneme)
            if phoneme == "ar":
                found_ar = True
            if phoneme == "ir":
                found_ir = True

        # Check mispronunciation block
        if "mispronunciations" in word and word["mispronunciations"]:
            for misp in word["mispronunciations"]:
                mispronounced_phone = remove_stress(misp["pronounced-phone"])
                if mispronounced_phone == "ar":
                    found_ar_mispronounced = True
                if mispronounced_phone == "ir":
                    found_ir_mispronounced = True

    # Stop checking once all are found
    return found_ar and found_ir and found_ar_mispronounced and found_ir_mispronounced

# Search for ar and ir in the dataset
for split in ["train", "test"]:
    for sample in ds[split]:
        if check_phonemes(sample):
            break  # Stop early if all cases are found

# Print results
print("\n🔎 **Check Results:**")
print(f"✅ `ar` in canonical phonemes: {'Found' if found_ar else 'Not Found'}")
print(f"✅ `ir` in canonical phonemes: {'Found' if found_ir else 'Not Found'}")
print(f"✅ `ar` in mispronunciation block: {'Found' if found_ar_mispronounced else 'Not Found'}")
print(f"✅ `ir` in mispronunciation block: {'Found' if found_ir_mispronounced else 'Not Found'}")
print("✅ Check completed.")