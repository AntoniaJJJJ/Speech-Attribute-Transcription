from datasets import load_dataset

# Load dataset and filter young speakers
ds = load_dataset("mispeech/speechocean762")
ds_filtered = ds.filter(lambda x: x["age"] <= 11)

potential_insertions = []

def check_for_insertions(example):
    for word in example["words"]:
        canonical = word.get("phones", [])
        misps = word.get("mispronunciations", [])

        for misp in misps:
            idx = misp.get("index")
            pronounced = misp.get("pronounced-phone", "").strip().lower()
            canonical_phone = misp.get("canonical-phone", "").strip().lower()

            # Case A: Missing or invalid index
            if idx is None or idx >= len(canonical):
                potential_insertions.append({
                    "text": example["text"],
                    "canonical_phones": canonical,
                    "index": idx,
                    "pronounced_phone": pronounced,
                    "canonical_phone": canonical_phone,
                    "note": "Index missing or out of bounds – possible insertion"
                })
                continue

            # Case B: Canonical phone is blank or missing
            if canonical_phone == "" or canonical_phone == " ":
                potential_insertions.append({
                    "text": example["text"],
                    "canonical_phones": canonical,
                    "index": idx,
                    "pronounced_phone": pronounced,
                    "canonical_phone": canonical_phone,
                    "note": "Blank or missing canonical-phone – possible insertion"
                })
                continue

            # Case C: Pronounced phone doesn't resemble canonical or neighbors
            nearby = canonical[max(0, idx - 1):idx + 2]
            if pronounced not in [c.lower() for c in nearby]:
                potential_insertions.append({
                    "text": example["text"],
                    "canonical_phones": canonical,
                    "index": idx,
                    "pronounced_phone": pronounced,
                    "canonical_phone": canonical_phone,
                    "note": "Pronounced phone does not match neighborhood – maybe insertion?"
                })

ds_filtered["train"].map(check_for_insertions)

# Summary
print(f"Total suspicious cases: {len(potential_insertions)}")
for ex in potential_insertions[:5]:
    print(f"Text: {ex['text']}")
    print(f"Index: {ex['index']} | Canonical: '{ex['canonical_phone']}' | Pronounced: {ex['pronounced_phone']} | Note: {ex['note']}")
    print("-" * 40)
