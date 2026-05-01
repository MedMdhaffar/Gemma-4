import json

# Load dataset
with open("WLASL_v0.3.json", "r") as f:
    data = json.load(f)

# Load missing IDs
with open("missing.txt", "r") as f:
    missing_ids = set(line.strip() for line in f)

filtered_data = []
class_counts = {}

# Step 1: build CLEAN dataset
for entry in data:
    gloss = entry["gloss"]

    valid_instances = [
        inst for inst in entry["instances"]
        if str(inst["video_id"]) not in missing_ids
    ]

    # keep only classes that still have videos
    if len(valid_instances) > 0:
        filtered_data.append({
            "gloss": gloss,
            "instances": valid_instances
        })

        class_counts[gloss] = len(valid_instances)

# Step 2: compute correct stats from filtered dataset
total_videos = sum(class_counts.values())
num_classes = len(class_counts)
avg_per_class = total_videos / num_classes if num_classes else 0

# Save results
stats = {
    "num_classes_filtered": num_classes,
    "total_videos_filtered": total_videos,
    "avg_videos_per_class_filtered": avg_per_class,
    "class_counts_filtered": class_counts
}

with open("wlasl_stats_filtered.json", "w") as f:
    json.dump(stats, f, indent=4)

print("✔ Correct filtered stats saved")
print(f"Classes: {num_classes}")
print(f"Total videos: {total_videos}")