import argparse
import csv
import json
import os
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


NUM_FRAMES = 32
NUM_HAND_LANDMARKS = 21
NUM_HANDS = 2
NUM_POSE_LANDMARKS = 6

# A compact lip/mouth subset from MediaPipe Face Mesh. Full face landmarks add a
# lot of noise for a first model; mouth motion is usually the useful face signal.
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

# MediaPipe Pose indices: left/right shoulders, elbows, wrists.
UPPER_BODY_POSE_LANDMARKS = [11, 12, 13, 14, 15, 16]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

NUM_MOUTH_LANDMARKS = len(MOUTH_LANDMARKS)
NUM_EYE_LANDMARKS = len(EYE_LANDMARKS)
NUM_POINTS = (
    (NUM_HANDS * NUM_HAND_LANDMARKS)
    + NUM_POSE_LANDMARKS
    + NUM_MOUTH_LANDMARKS
    + NUM_EYE_LANDMARKS
)
FEATURE_DIM = 3
EPSILON = 1e-6


def natural_frame_key(path):
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


def load_gloss_lookup(wlasl_json_path):
    """Return class-index -> gloss from WLASL_v0.3.json when available."""
    if not wlasl_json_path or not Path(wlasl_json_path).exists():
        return {}

    with open(wlasl_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    return {idx: item["gloss"] for idx, item in enumerate(rows)}


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


def handedness_name(hand_result, index):
    try:
        categories = hand_result.handedness[index]
        if categories:
            return categories[0].category_name.lower()
    except (AttributeError, IndexError):
        pass
    return None


def landmarks_to_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def assign_hands(hand_result):
    hands = np.zeros((2, NUM_HAND_LANDMARKS, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((2, NUM_HAND_LANDMARKS), dtype=np.float32)
    unassigned = []

    if not hand_result.hand_landmarks:
        return hands, mask

    for i, hand_landmarks in enumerate(hand_result.hand_landmarks[:2]):
        hand_array = landmarks_to_array(hand_landmarks)
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

    # Fallback for cases where handedness is unavailable or duplicated: use x
    # position in the image. Smaller x is image-left, larger x is image-right.
    for hand_array in unassigned:
        mean_x = float(hand_array[:, 0].mean())
        preferred_slot = 0 if mean_x < 0.5 else 1
        slot = preferred_slot if mask[preferred_slot, 0] == 0 else 1 - preferred_slot
        hands[slot] = hand_array
        mask[slot] = 1.0

    return hands, mask


def extract_mouth(face_result):
    mouth = np.zeros((NUM_MOUTH_LANDMARKS, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((NUM_MOUTH_LANDMARKS,), dtype=np.float32)

    if not face_result.face_landmarks:
        return mouth, mask

    face = face_result.face_landmarks[0]
    for out_idx, face_idx in enumerate(MOUTH_LANDMARKS):
        if face_idx < len(face):
            lm = face[face_idx]
            mouth[out_idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            mask[out_idx] = 1.0

    return mouth, mask


def extract_eyes(face_result):
    eyes = np.zeros((NUM_EYE_LANDMARKS, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((NUM_EYE_LANDMARKS,), dtype=np.float32)

    if not face_result.face_landmarks:
        return eyes, mask

    face = face_result.face_landmarks[0]
    for out_idx, face_idx in enumerate(EYE_LANDMARKS):
        if face_idx < len(face):
            lm = face[face_idx]
            eyes[out_idx] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
            mask[out_idx] = 1.0

    return eyes, mask


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


def normalize_frame(points, mask):
    """Center each frame and scale it so signer/video size matters less."""
    normalized = np.zeros_like(points, dtype=np.float32)
    visible = mask > 0

    if not np.any(visible):
        return normalized

    left_wrist_visible = bool(mask[0] > 0)
    right_wrist_visible = bool(mask[NUM_HAND_LANDMARKS] > 0)
    left_shoulder_idx = NUM_HANDS * NUM_HAND_LANDMARKS
    right_shoulder_idx = left_shoulder_idx + 1
    shoulders_visible = bool(mask[left_shoulder_idx] > 0 and mask[right_shoulder_idx] > 0)

    if shoulders_visible:
        left_shoulder = points[left_shoulder_idx]
        right_shoulder = points[right_shoulder_idx]
        origin = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    elif left_wrist_visible and right_wrist_visible:
        left_wrist = points[0]
        right_wrist = points[NUM_HAND_LANDMARKS]
        origin = (left_wrist + right_wrist) / 2.0
        scale = np.linalg.norm(left_wrist[:2] - right_wrist[:2])
    elif left_wrist_visible:
        origin = points[0]
        visible_xy = points[visible, :2]
        scale = np.ptp(visible_xy, axis=0).max()
    elif right_wrist_visible:
        origin = points[NUM_HAND_LANDMARKS]
        visible_xy = points[visible, :2]
        scale = np.ptp(visible_xy, axis=0).max()
    else:
        visible_points = points[visible]
        origin = visible_points.mean(axis=0)
        scale = np.ptp(visible_points[:, :2], axis=0).max()

    if scale < EPSILON:
        scale = 1.0

    normalized[visible] = (points[visible] - origin) / scale
    return normalized


def extract_frame_features(image_path, hand_detector, face_detector, pose_detector):
    image = mp.Image.create_from_file(str(image_path))

    hand_result = hand_detector.detect(image)
    face_result = face_detector.detect(image)
    pose_result = pose_detector.detect(image)

    hands, hand_mask = assign_hands(hand_result)
    pose, pose_mask = extract_upper_body_pose(pose_result)
    mouth, mouth_mask = extract_mouth(face_result)
    eyes, eyes_mask = extract_eyes(face_result)

    points = np.concatenate(
        [
            hands[0],
            hands[1],
            pose,
            mouth,
            eyes,
        ],
        axis=0,
    )
    mask = np.concatenate(
        [
            hand_mask[0],
            hand_mask[1],
            pose_mask,
            mouth_mask,
            eyes_mask,
        ],
        axis=0,
    )

    return normalize_frame(points, mask), mask


def sample_or_pad_frames(frame_paths, target_frames):
    if not frame_paths:
        return []

    if len(frame_paths) == target_frames:
        return frame_paths

    if len(frame_paths) > target_frames:
        indices = np.linspace(0, len(frame_paths) - 1, target_frames).round().astype(int)
        return [frame_paths[i] for i in indices]

    padded = list(frame_paths)
    while len(padded) < target_frames:
        padded.append(frame_paths[-1])
    return padded


def process_video(
    video_id,
    frame_dir,
    output_dir,
    hand_detector,
    face_detector,
    pose_detector,
    target_frames,
    strict_frames,
):
    image_paths = sorted(
        list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")),
        key=natural_frame_key,
    )
    sampled_paths = sample_or_pad_frames(image_paths, target_frames)
    if strict_frames and len(image_paths) != target_frames:
        raise ValueError(f"expected {target_frames} frames, found {len(image_paths)}")

    features = np.zeros((target_frames, NUM_POINTS, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((target_frames, NUM_POINTS), dtype=np.float32)

    for frame_idx, image_path in enumerate(sampled_paths):
        frame_features, frame_mask = extract_frame_features(
            image_path, hand_detector, face_detector, pose_detector
        )
        features[frame_idx] = frame_features
        masks[frame_idx] = frame_mask

    output_dir.mkdir(parents=True, exist_ok=True)
    stgcn_features = features.transpose(2, 0, 1)[:, :, :, np.newaxis]
    np.save(output_dir / f"{video_id}.npy", stgcn_features)
    np.save(output_dir / f"{video_id}_mask.npy", masks)

    return {
        "frames_found": len(image_paths),
        "frames_used": len(sampled_paths),
        "visible_ratio": float(masks.mean()) if masks.size else 0.0,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_stgcn_graph_metadata(num_frames):
    left_hand_start = 0
    right_hand_start = left_hand_start + NUM_HAND_LANDMARKS
    pose_start = right_hand_start + NUM_HAND_LANDMARKS
    mouth_start = pose_start + NUM_POSE_LANDMARKS
    eyes_start = mouth_start + NUM_MOUTH_LANDMARKS

    edges = []
    edges.extend((left_hand_start + a, left_hand_start + b) for a, b in HAND_CONNECTIONS)
    edges.extend((right_hand_start + a, right_hand_start + b) for a, b in HAND_CONNECTIONS)

    # pose order: left shoulder, right shoulder, left elbow, right elbow,
    # left wrist, right wrist
    edges.extend(
        [
            (pose_start + 0, pose_start + 1),
            (pose_start + 0, pose_start + 2),
            (pose_start + 2, pose_start + 4),
            (pose_start + 1, pose_start + 3),
            (pose_start + 3, pose_start + 5),
        ]
    )

    # Compact contour chains. These are not perfect anatomical meshes, but they
    # give ST-GCN local neighborhoods for the face signals.
    edges.extend(
        (mouth_start + i, mouth_start + i + 1)
        for i in range(NUM_MOUTH_LANDMARKS - 1)
    )
    edges.append((mouth_start + NUM_MOUTH_LANDMARKS - 1, mouth_start))

    left_eye_start = eyes_start
    right_eye_start = eyes_start + len(LEFT_EYE_LANDMARKS)
    edges.extend(
        (left_eye_start + i, left_eye_start + i + 1)
        for i in range(len(LEFT_EYE_LANDMARKS) - 1)
    )
    edges.append((left_eye_start + len(LEFT_EYE_LANDMARKS) - 1, left_eye_start))
    edges.extend(
        (right_eye_start + i, right_eye_start + i + 1)
        for i in range(len(RIGHT_EYE_LANDMARKS) - 1)
    )
    edges.append((right_eye_start + len(RIGHT_EYE_LANDMARKS) - 1, right_eye_start))

    return {
        "format": "C,T,V,M",
        "shape": [FEATURE_DIM, num_frames, NUM_POINTS, 1],
        "channels": ["x", "y", "z"],
        "num_frames": num_frames,
        "num_vertices": NUM_POINTS,
        "num_people": 1,
        "vertex_ranges": {
            "left_hand": [left_hand_start, left_hand_start + NUM_HAND_LANDMARKS - 1],
            "right_hand": [right_hand_start, right_hand_start + NUM_HAND_LANDMARKS - 1],
            "upper_body_pose": [pose_start, pose_start + NUM_POSE_LANDMARKS - 1],
            "mouth": [mouth_start, mouth_start + NUM_MOUTH_LANDMARKS - 1],
            "eyes": [eyes_start, eyes_start + NUM_EYE_LANDMARKS - 1],
        },
        "source_landmark_indices": {
            "upper_body_pose": UPPER_BODY_POSE_LANDMARKS,
            "mouth": MOUTH_LANDMARKS,
            "eyes": EYE_LANDMARKS,
        },
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build normalized 32-frame landmark tensors for WLASL NSLT-2000."
    )
    parser.add_argument("--metadata", default="kaggle_dataset_info/nslt_2000.json")
    parser.add_argument("--wlasl-json", default="kaggle_dataset_info/WLASL_v0.3.json")
    parser.add_argument("--frames-root", default="frames_without_resize_crop")
    parser.add_argument("--model-dir", default="mediaPipe_models")
    parser.add_argument("--output-root", default="preprocessed_nslt2000")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small debug run, e.g. --limit 20.",
    )
    parser.add_argument(
        "--strict-frames",
        action="store_true",
        help="Skip videos that do not contain exactly --num-frames frames.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    data_dir = output_root / "landmarks"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    gloss_lookup = load_gloss_lookup(args.wlasl_json)
    hand_detector, face_detector, pose_detector = build_detectors(args.model_dir)

    rows = []
    skipped = []

    items = list(metadata.items())
    if args.limit is not None:
        items = items[: args.limit]

    for video_id, item in tqdm(items, desc="Preprocessing NSLT-2000"):
        frame_dir = frames_root / video_id
        if not frame_dir.exists():
            skipped.append({"video_id": video_id, "reason": "missing_frame_dir"})
            continue

        label = int(item["action"][0])
        subset = item["subset"]
        try:
            stats = process_video(
                video_id=video_id,
                frame_dir=frame_dir,
                output_dir=data_dir,
                hand_detector=hand_detector,
                face_detector=face_detector,
                pose_detector=pose_detector,
                target_frames=args.num_frames,
                strict_frames=args.strict_frames,
            )
        except ValueError as exc:
            skipped.append({"video_id": video_id, "reason": str(exc)})
            continue

        rows.append(
            {
                "video_id": video_id,
                "subset": subset,
                "label": label,
                "gloss": gloss_lookup.get(label, ""),
                "feature_path": str(data_dir / f"{video_id}.npy"),
                "mask_path": str(data_dir / f"{video_id}_mask.npy"),
                **stats,
            }
        )

    fieldnames = [
        "video_id",
        "subset",
        "label",
        "gloss",
        "feature_path",
        "mask_path",
        "frames_found",
        "frames_used",
        "visible_ratio",
    ]
    write_csv(output_root / "manifest.csv", rows, fieldnames)

    for subset in ["train", "val", "test"]:
        subset_rows = [row for row in rows if row["subset"] == subset]
        write_csv(output_root / f"{subset}.csv", subset_rows, fieldnames)

    write_csv(output_root / "skipped.csv", skipped, ["video_id", "reason"])
    write_json(output_root / "stgcn_graph.json", build_stgcn_graph_metadata(args.num_frames))

    label_rows = [
        {"label": label, "gloss": gloss}
        for label, gloss in sorted(gloss_lookup.items())
        if 0 <= label < 2000
    ]
    write_csv(output_root / "label_map.csv", label_rows, ["label", "gloss"])

    print(f"Saved {len(rows)} processed videos to: {output_root}")
    print(f"Skipped {len(skipped)} videos. See: {output_root / 'skipped.csv'}")
    print(f"Feature tensor shape per video: ({FEATURE_DIM}, {args.num_frames}, {NUM_POINTS}, 1)")
    print(f"ST-GCN graph metadata: {output_root / 'stgcn_graph.json'}")


if __name__ == "__main__":
    main()
