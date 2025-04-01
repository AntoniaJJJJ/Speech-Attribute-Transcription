from datasets import load_dataset

# Load dataset and filter for age
ds = load_dataset("mispeech/speechocean762")
ds_filtered = ds.filter(lambda x: x["age"] <= 11)

# Store insertion candidates
insertion_candidates = []

def find_insertions(example):
    for word in example["words"]:
        canonical = word.get("phones", [])
        misps = word.get("mispronunciations", [])

        for misp in misps:
            idx = misp.get("index")
            pronounced = misp.get("pronounced-phone", "").strip().lower()
            canonical_phone = misp.get("canonical-phone", "").strip().lower() if "canonical-phone" in misp else None

            # Case 1: Index out of bounds — no canonical phoneme exists
            if idx is None or idx >= len(canonical):
                insertion_candidates.append({
                    "text": example["text"],
                    "index": idx,
                    "pronounced_phone": pronounced,
                    "canonical_phone": canonical_phone,
                    "canonical_phones": canonical,
                    "note": "Index out of bounds – likely insertion"
                })
                continue

            # Case 2: Canonical-phone field is blank or missing
            if canonical_phone in ["", " "] or canonical_phone is None:
                insertion_candidates.append({
                    "text": example["text"],
                    "index": idx,
                    "pronounced_phone": pronounced,
                    "canonical_phone": canonical_phone,
                    "canonical_phones": canonical,
                    "note": "Missing or blank canonical-phone – likely insertion"
                })
                continue

            # ✅ Otherwise: it's likely a substitution or distortion — ignore

ds_filtered["train"].map(find_insertions)

# Print summary of suspected insertions
print(f"Total suspected insertions: {len(insertion_candidates)}\n")
for ex in insertion_candidates[:5]:
    print(f"Text: {ex['text']}")
    print(f"Index: {ex['index']} | Canonical: '{ex['canonical_phone']}' | Pronounced: '{ex['pronounced_phone']}'")
    print(f"Note: {ex['note']}")
    print("-" * 50)
