import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

DEFAULT_EDGES = {
    "left_hand": [(a, b) for a, b in HAND_CONNECTIONS],
    "right_hand": [(21 + a, 21 + b) for a, b in HAND_CONNECTIONS],
    "upper_body_pose": [(42, 43), (42, 44), (44, 46), (43, 45), (45, 47)],
}

COLORS = {
    "left_hand": (80, 220, 80),
    "right_hand": (80, 180, 255),
    "upper_body_pose": (255, 170, 80),
    "face": (255, 80, 220),
}


def natural_frame_key(path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else path.stem


def load_sequence(npy_path):
    arr = np.load(npy_path)

    # New ST-GCN format: C, T, V, M
    if arr.ndim == 4:
        if arr.shape[0] not in (2, 3):
            raise ValueError(f"Expected C first in ST-GCN tensor, got shape {arr.shape}")
        arr = arr[:, :, :, 0].transpose(1, 2, 0)
        return arr.astype(np.float32), "stgcn_normalized"

    # Video sequence format: T, V, C
    if arr.ndim == 3:
        return arr.astype(np.float32), "absolute_or_sequence"

    # Single-frame format from the original npy_generator.py: V, C
    if arr.ndim == 2:
        return arr[np.newaxis, ...].astype(np.float32), "single_frame_absolute"

    raise ValueError(f"Unsupported .npy shape: {arr.shape}")


def load_mask(mask_path, sequence_shape):
    if not mask_path:
        return None
    mask = np.load(mask_path).astype(np.float32)
    if mask.shape != sequence_shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match expected {sequence_shape[:2]}"
        )
    return mask


def load_graph_edges(graph_path):
    if not graph_path:
        return DEFAULT_EDGES

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    edges = {"graph": [tuple(edge) for edge in graph.get("edges", [])]}
    return edges if edges["graph"] else DEFAULT_EDGES


def point_to_pixel(point, width, height, mode, normalized_scale):
    x = float(point[0])
    y = float(point[1])

    if mode == "absolute":
        px = int(round(x * width))
        py = int(round(y * height))
    else:
        scale = min(width, height) * normalized_scale
        px = int(round((width / 2.0) + (x * scale)))
        py = int(round((height / 2.0) + (y * scale)))

    return px, py


def is_missing(point, mask_value=None):
    if mask_value is not None:
        return mask_value <= 0
    return np.allclose(point[:2], 0.0)


def draw_edges(frame, points, mask, edges_by_group, mode, normalized_scale):
    height, width = frame.shape[:2]

    for group, edges in edges_by_group.items():
        color = COLORS.get(group, (240, 240, 240))
        for a, b in edges:
            if a >= len(points) or b >= len(points):
                continue
            ma = None if mask is None else mask[a]
            mb = None if mask is None else mask[b]
            if is_missing(points[a], ma) or is_missing(points[b], mb):
                continue
            pa = point_to_pixel(points[a], width, height, mode, normalized_scale)
            pb = point_to_pixel(points[b], width, height, mode, normalized_scale)
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)


def draw_points(frame, points, mask, mode, normalized_scale):
    height, width = frame.shape[:2]

    for idx, point in enumerate(points):
        mask_value = None if mask is None else mask[idx]
        if is_missing(point, mask_value):
            continue

        if idx < 21:
            color = COLORS["left_hand"]
            radius = 4
        elif idx < 42:
            color = COLORS["right_hand"]
            radius = 4
        elif idx < 48:
            color = COLORS["upper_body_pose"]
            radius = 5
        else:
            color = COLORS["face"]
            radius = 2

        px, py = point_to_pixel(point, width, height, mode, normalized_scale)
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)


def draw_label(frame, text):
    cv2.rectangle(frame, (8, 8), (8 + 11 * len(text), 38), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Overlay a landmark .npy sequence on a 32-frame video folder."
    )
    parser.add_argument("--frames-dir", required=True, help="Folder containing frame_*.jpg")
    parser.add_argument("--npy", required=True, help="Landmark .npy file to visualize")
    parser.add_argument("--mask", default=None, help="Optional *_mask.npy file")
    parser.add_argument("--graph", default=None, help="Optional stgcn_graph.json file")
    parser.add_argument("--output", default="landmark_preview.mp4")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=["auto", "absolute", "normalized"],
        default="auto",
        help="Use absolute for raw 0..1 MediaPipe coords; normalized for ST-GCN tensors.",
    )
    parser.add_argument(
        "--normalized-scale",
        type=float,
        default=0.28,
        help="Skeleton size for normalized ST-GCN previews.",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")),
        key=natural_frame_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg/.png frames found in {frames_dir}")

    sequence, detected_kind = load_sequence(args.npy)
    mask = load_mask(args.mask, sequence.shape) if args.mask else None
    edges = load_graph_edges(args.graph)

    if args.mode == "auto":
        mode = "normalized" if detected_kind == "stgcn_normalized" else "absolute"
    else:
        mode = args.mode

    if len(sequence) == 1:
        frame_paths = frame_paths[:1]
    elif len(frame_paths) != len(sequence):
        raise ValueError(
            f"Frame count ({len(frame_paths)}) does not match .npy timesteps ({len(sequence)})."
        )

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Could not read frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}")

    for t, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")

        frame_mask = None if mask is None else mask[t]
        draw_edges(frame, sequence[t], frame_mask, edges, mode, args.normalized_scale)
        draw_points(frame, sequence[t], frame_mask, mode, args.normalized_scale)
        draw_label(frame, f"{Path(args.npy).name} | frame {t:02d} | {mode}")
        writer.write(frame)

    writer.release()
    print(f"Saved landmark preview to {output_path.resolve()}")


if __name__ == "__main__":
    main()
