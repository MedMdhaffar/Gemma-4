import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------- INIT MODELS ONCE (IMPORTANT for speed) --------
base_options_face = python.BaseOptions(model_asset_path='./mediaPipe_models/face_landmarker_v2_with_blendshapes.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=base_options_face,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

base_options_hand = python.BaseOptions(model_asset_path='./mediaPipe_models/hand_landmarker.task')
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


# -------- FUNCTION --------
def extract_landmarks(image_path, output_path):
    image = mp.Image.create_from_file(image_path)

    # Detect
    face_result = face_detector.detect(image)
    hand_result = hand_detector.detect(image)

    NUM_HANDS = 2
    NUM_HAND_LANDMARKS = 21
    NUM_FACE_LANDMARKS = 478

    # ---- HANDS ----
    hand_data = np.zeros((NUM_HANDS, NUM_HAND_LANDMARKS, 2), dtype=np.float32)

    if hand_result.hand_landmarks:
        for i, hand in enumerate(hand_result.hand_landmarks):
            if i >= NUM_HANDS:
                break
            hand_data[i] = np.array([[lm.x, lm.y] for lm in hand])

    # ---- FACE ----
    face_data = np.zeros((1, NUM_FACE_LANDMARKS, 2), dtype=np.float32)

    if face_result.face_landmarks:
        face = face_result.face_landmarks[0]
        face_array = np.array([[lm.x, lm.y] for lm in face])
        face_data[0, :face_array.shape[0]] = face_array

    # ---- FLATTEN ----
    hand_flat = hand_data.reshape(-1, 2)
    face_flat = face_data.reshape(-1, 2)

    all_landmarks = np.concatenate([hand_flat, face_flat], axis=0)

    # ---- SAVE ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_landmarks)


from tqdm import tqdm

input_root = "./frames_without_resize_crop"
output_root = "./npy"

video_ids = os.listdir(input_root)

for vid in tqdm(video_ids, desc="Processing videos"):
    vid_path = os.path.join(input_root, vid)

    if not os.path.isdir(vid_path):
        continue

    frame_files = sorted(os.listdir(vid_path))

    for frame in frame_files:
        if not frame.endswith(".jpg"):
            continue

        input_path = os.path.join(vid_path, frame)

        # Change extension to .npy
        frame_name = frame.replace(".jpg", ".npy")
        output_path = os.path.join(output_root, vid, frame_name)

        try:
            extract_landmarks(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")