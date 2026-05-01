import os
import cv2
import numpy as np
from tqdm import tqdm

video_dir = "videos"
output_dir = "frames_without_resize_crop"
num_frames = 32

os.makedirs(output_dir, exist_ok=True)

def extract_frames(video_path, save_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    selected_set = set(indices)

    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i in selected_set:
            # ❌ NO resize
            # ❌ NO crop
            frames.append(frame)

        if len(frames) == num_frames:
            break

    cap.release()

    # padding (keep original frame size)
    if len(frames) < num_frames:
        last = frames[-1] if frames else None
        while len(frames) < num_frames:
            frames.append(last)

    os.makedirs(save_folder, exist_ok=True)

    for idx, frame in enumerate(frames):
        filename = os.path.join(save_folder, f"frame_{idx:03d}.jpg")
        cv2.imwrite(filename, frame)

    return True


# Process all videos
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

success = 0
failed = 0

for video_file in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join(video_dir, video_file)
    video_id = os.path.splitext(video_file)[0]
    save_folder = os.path.join(output_dir, video_id)

    ok = extract_frames(video_path, save_folder)

    if ok:
        success += 1
    else:
        failed += 1
        tqdm.write(f"⚠️ Failed: {video_file}")

print("\n✅ Done extracting frames")
print(f"✔ Success: {success}")
print(f"❌ Failed: {failed}")