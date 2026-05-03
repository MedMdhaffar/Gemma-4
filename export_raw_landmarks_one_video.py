import argparse
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


NUM_HAND_LANDMARKS = 21
NUM_HANDS = 2
NUM_POSE_LANDMARKS = 6

MOUTH_LANDMARKS = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80,
    81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
    185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
    314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
]
LEFT_EYE_LANDMARKS = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_LANDMARKS = [
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 398, 384, 385, 386, 387, 388, 466,
]
EYE_LANDMARKS = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS
UPPER_BODY_POSE_LANDMARKS = [11, 12, 13, 14, 15, 16]

NUM_MOUTH_LANDMARKS = len(MOUTH_LANDMARKS)
NUM_EYE_LANDMARKS = len(EYE_LANDMARKS)
NUM_POINTS = (
    (NUM_HANDS * NUM_HAND_LANDMARKS)
    + NUM_POSE_LANDMARKS
    + NUM_MOUTH_LANDMARKS
    + NUM_EYE_LANDMARKS
)
FEATURE_DIM = 3


def natural_frame_key(path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else path.stem


def build_detectors(model_dir):
    model_dir = Path(model_dir)

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_dir / "hand_landmarker.task")
        ),
        num_hands=2,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_dir / "face_landmarker_v2_with_blendshapes.task")
        ),
        num_faces=1,
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_dir / "pose_landmarker_full.task")
        ),
        num_poses=1,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    return hand_detector, face_detector, pose_detector


def landmarks_to_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def handedness_name(hand_result, index):
    try:
        categories = hand_result.handedness[index]
        if categories:
            return categories[0].category_name.lower()
    except (AttributeError, IndexError):
        pass
    return None


def assign_hands(hand_result):
    hands = np.zeros((2, NUM_HAND_LANDMARKS, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((2, NUM_HAND_LANDMARKS), dtype=np.float32)
    unassigned = []

    if not hand_result.hand_landmarks:
        return hands, mask

    for i, hand in enumerate(hand_result.hand_landmarks[:2]):
        hand_array = landmarks_to_array(hand)
        label = handedness_name(hand_result, i)

        if label == "left":
            slot = 0
        elif label == "right":
            slot = 1
        else:
            unassigned.append(hand_array)
            continue

        hands[slot] = hand_array
        mask[slot] = 1.0

    for hand_array in unassigned:
        mean_x = float(hand_array[:, 0].mean())
        preferred_slot = 0 if mean_x < 0.5 else 1
        slot = preferred_slot if mask[preferred_slot, 0] == 0 else 1 - preferred_slot
        hands[slot] = hand_array
        mask[slot] = 1.0

    return hands, mask


def extract_face_subset(face_result, indices):
    points = np.zeros((len(indices), FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(indices),), dtype=np.float32)

    if not face_result.face_landmarks:
        return points, mask

    face = face_result.face_landmarks[0]
    for out_idx, face_idx in enumerate(indices):
        if face_idx < len(face):
            lm = face[face_idx]
            points[out_idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            mask[out_idx] = 1.0

    return points, mask


def extract_upper_body_pose(pose_result):
    pose = np.zeros((NUM_POSE_LANDMARKS, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((NUM_POSE_LANDMARKS,), dtype=np.float32)

    if not pose_result.pose_landmarks:
        return pose, mask

    landmarks = pose_result.pose_landmarks[0]
    for out_idx, pose_idx in enumerate(UPPER_BODY_POSE_LANDMARKS):
        if pose_idx < len(landmarks):
            lm = landmarks[pose_idx]
            pose[out_idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            mask[out_idx] = 1.0

    return pose, mask


def extract_frame_raw(image_path, hand_detector, face_detector, pose_detector):
    image = mp.Image.create_from_file(str(image_path))

    hand_result = hand_detector.detect(image)
    face_result = face_detector.detect(image)
    pose_result = pose_detector.detect(image)

    hands, hand_mask = assign_hands(hand_result)
    pose, pose_mask = extract_upper_body_pose(pose_result)
    mouth, mouth_mask = extract_face_subset(face_result, MOUTH_LANDMARKS)
    eyes, eyes_mask = extract_face_subset(face_result, EYE_LANDMARKS)

    points = np.concatenate([hands[0], hands[1], pose, mouth, eyes], axis=0)
    mask = np.concatenate(
        [hand_mask[0], hand_mask[1], pose_mask, mouth_mask, eyes_mask],
        axis=0,
    )

    return points, mask


def main():
    parser = argparse.ArgumentParser(
        description="Export raw MediaPipe coordinates for one video folder."
    )
    parser.add_argument("--video-id", default="02003")
    parser.add_argument("--frames-root", default="frames_without_resize_crop")
    parser.add_argument("--model-dir", default="mediaPipe_models")
    parser.add_argument("--output-root", default="raw_landmark_debug")
    args = parser.parse_args()

    frame_dir = Path(args.frames_root) / args.video_id
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame folder not found: {frame_dir}")

    frame_paths = sorted(
        list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")),
        key=natural_frame_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg/.png frames found in: {frame_dir}")

    hand_detector, face_detector, pose_detector = build_detectors(args.model_dir)

    features = np.zeros((len(frame_paths), NUM_POINTS, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((len(frame_paths), NUM_POINTS), dtype=np.float32)

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=args.video_id)):
        frame_features, frame_mask = extract_frame_raw(
            frame_path, hand_detector, face_detector, pose_detector
        )
        features[frame_idx] = frame_features
        masks[frame_idx] = frame_mask

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{args.video_id}_raw.npy"
    mask_path = output_dir / f"{args.video_id}_raw_mask.npy"

    np.save(feature_path, features)
    np.save(mask_path, masks)

    print(f"Saved raw landmarks: {feature_path}")
    print(f"Saved raw mask: {mask_path}")
    print(f"Raw shape: {features.shape}")
    print("Landmark order: left_hand, right_hand, upper_body_pose, mouth, eyes")


if __name__ == "__main__":
    main()
