import json
from collections import Counter

json_path = "kaggle_dataset_info/nslt_300.json"
missing_path = "kaggle_dataset_info/missing.txt"  # it's just a text file

# load dataset
with open(json_path, "r") as f:
    data = json.load(f)

# load missing IDs
with open(missing_path, "r") as f:
    missing_ids = set(line.strip() for line in f if line.strip())

# filter dataset
filtered_data = {
    vid: info for vid, info in data.items()
    if vid not in missing_ids
}

# count splits
split_counter = Counter()
for vid, info in filtered_data.items():
    split_counter[info["subset"]] += 1

print("Original videos:", len(data))
print("Missing videos:", len(missing_ids))
print("Remaining videos:", len(filtered_data))

print("\nSplit distribution (after removing missing):")
for k, v in split_counter.items():
    print(f"{k}: {v}")