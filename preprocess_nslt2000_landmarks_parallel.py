import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from preprocess_nslt2000_landmarks import (
    FEATURE_DIM,
    NUM_POINTS,
    build_detectors,
    build_stgcn_graph_metadata,
    load_gloss_lookup,
    natural_frame_key,
    process_video,
    write_csv,
    write_json,
)


_HAND_DETECTOR = None
_FACE_DETECTOR = None
_POSE_DETECTOR = None
_FRAMES_ROOT = None
_DATA_DIR = None
_TARGET_FRAMES = None
_STRICT_FRAMES = None
_SKIP_EXISTING = None


def init_worker(model_dir, frames_root, data_dir, target_frames, strict_frames, skip_existing):
    global _HAND_DETECTOR
    global _FACE_DETECTOR
    global _POSE_DETECTOR
    global _FRAMES_ROOT
    global _DATA_DIR
    global _TARGET_FRAMES
    global _STRICT_FRAMES
    global _SKIP_EXISTING

    _HAND_DETECTOR, _FACE_DETECTOR, _POSE_DETECTOR = build_detectors(model_dir)
    _FRAMES_ROOT = Path(frames_root)
    _DATA_DIR = Path(data_dir)
    _TARGET_FRAMES = target_frames
    _STRICT_FRAMES = strict_frames
    _SKIP_EXISTING = skip_existing


def count_video_frames(frame_dir):
    return len(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))


def existing_video_stats(video_id, frame_dir, data_dir):
    mask_path = data_dir / f"{video_id}_mask.npy"
    mask = np.load(mask_path)
    frames_found = count_video_frames(frame_dir)

    return {
        "frames_found": frames_found,
        "frames_used": int(mask.shape[0]),
        "visible_ratio": float(mask.mean()) if mask.size else 0.0,
        "status": "existing",
    }


def worker_process_video(task):
    video_id, item = task
    frame_dir = _FRAMES_ROOT / video_id
    feature_path = _DATA_DIR / f"{video_id}.npy"
    mask_path = _DATA_DIR / f"{video_id}_mask.npy"

    if not frame_dir.exists():
        return {
            "video_id": video_id,
            "ok": False,
            "reason": "missing_frame_dir",
        }

    if _SKIP_EXISTING and feature_path.exists() and mask_path.exists():
        try:
            stats = existing_video_stats(video_id, frame_dir, _DATA_DIR)
        except Exception as exc:
            return {
                "video_id": video_id,
                "ok": False,
                "reason": f"bad_existing_output: {exc}",
            }

        return {
            "video_id": video_id,
            "ok": True,
            "subset": item["subset"],
            "label": int(item["action"][0]),
            **stats,
        }

    try:
        stats = process_video(
            video_id=video_id,
            frame_dir=frame_dir,
            output_dir=_DATA_DIR,
            hand_detector=_HAND_DETECTOR,
            face_detector=_FACE_DETECTOR,
            pose_detector=_POSE_DETECTOR,
            target_frames=_TARGET_FRAMES,
            strict_frames=_STRICT_FRAMES,
        )
    except Exception as exc:
        return {
            "video_id": video_id,
            "ok": False,
            "reason": str(exc),
        }

    return {
        "video_id": video_id,
        "ok": True,
        "subset": item["subset"],
        "label": int(item["action"][0]),
        "status": "processed",
        **stats,
    }


def append_manifest_row(result, data_dir, gloss_lookup):
    video_id = result["video_id"]
    return {
        "video_id": video_id,
        "subset": result["subset"],
        "label": result["label"],
        "gloss": gloss_lookup.get(result["label"], ""),
        "feature_path": str(data_dir / f"{video_id}.npy"),
        "mask_path": str(data_dir / f"{video_id}_mask.npy"),
        "frames_found": result["frames_found"],
        "frames_used": result["frames_used"],
        "visible_ratio": result["visible_ratio"],
        "status": result["status"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel preprocessing for normalized ST-GCN tensors on WLASL NSLT-2000."
    )
    parser.add_argument("--metadata", default="kaggle_dataset_info/nslt_2000.json")
    parser.add_argument("--wlasl-json", default="kaggle_dataset_info/WLASL_v0.3.json")
    parser.add_argument("--frames-root", default="frames_without_resize_crop")
    parser.add_argument("--model-dir", default="mediaPipe_models")
    parser.add_argument("--output-root", default="preprocessed_nslt2000")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strict-frames",
        action="store_true",
        help="Skip videos that do not contain exactly --num-frames frames.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess videos even when both .npy and *_mask.npy already exist.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    data_dir = output_root / "landmarks"
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    gloss_lookup = load_gloss_lookup(args.wlasl_json)
    items = list(metadata.items())
    if args.limit is not None:
        items = items[: args.limit]

    rows = []
    skipped = []
    skip_existing = not args.no_skip_existing

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(
            args.model_dir,
            str(frames_root),
            str(data_dir),
            args.num_frames,
            args.strict_frames,
            skip_existing,
        ),
    ) as executor:
        futures = [executor.submit(worker_process_video, item) for item in items]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Preprocessing NSLT-2000 ({args.workers} workers)",
        ):
            result = future.result()
            if result["ok"]:
                rows.append(append_manifest_row(result, data_dir, gloss_lookup))
            else:
                skipped.append(
                    {
                        "video_id": result["video_id"],
                        "reason": result["reason"],
                    }
                )

    rows.sort(key=lambda row: row["video_id"])
    skipped.sort(key=lambda row: row["video_id"])

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
        "status",
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

    existing_count = sum(1 for row in rows if row["status"] == "existing")
    processed_count = sum(1 for row in rows if row["status"] == "processed")

    print(f"Output root: {output_root}")
    print(f"Processed this run: {processed_count}")
    print(f"Skipped existing: {existing_count}")
    print(f"Failed/skipped: {len(skipped)}")
    print(f"Feature tensor shape per video: ({FEATURE_DIM}, {args.num_frames}, {NUM_POINTS}, 1)")
    print(f"Manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
