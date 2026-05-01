import os
import cv2
from tqdm import tqdm

video_dir = "videos"

files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

max_duration = 0
longest_video = None
corrupted_videos = []

for file in tqdm(files, desc="Processing videos"):
    path = os.path.join(video_dir, file)
    
    cap = cv2.VideoCapture(path)
    
    # Check 1: cannot open file
    if not cap.isOpened():
        corrupted_videos.append(file)
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Check 2: invalid metadata
    if fps <= 0 or frame_count <= 0:
        corrupted_videos.append(file)
        cap.release()
        continue
    
    # Check 3: try reading first frame
    ret, frame = cap.read()
    if not ret:
        corrupted_videos.append(file)
        cap.release()
        continue
    
    # Compute duration
    duration = frame_count / fps
    
    if duration > max_duration:
        max_duration = duration
        longest_video = file
    
    cap.release()

print(f"\nLongest video: {longest_video}")
print(f"Duration: {max_duration:.2f} seconds")

print("\nCorrupted videos:")
for v in corrupted_videos:
    print(v)

print(f"\nTotal corrupted: {len(corrupted_videos)}")