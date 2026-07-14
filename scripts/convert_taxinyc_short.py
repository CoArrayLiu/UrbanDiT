#!/usr/bin/env python3
"""Convert UniST's TaxiNYC_short.json to UrbanDiT's continuous NPY format.

The UniST release stores each scalar flow channel as overlapping 12-frame
windows. Within every train/validation/test split, the first half of the
samples is one channel and the second half is the other channel. This script
restores each split to a time series and writes the two channels separately.

By convention used by the source TaxiNYC data, the first group is treated as
inflow and the second as outflow. Pass --swap-channels if that convention is
known to be reversed in a different copy of the JSON file.
"""

from __future__ import annotations

import argparse
import json
import mmap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SPLITS = ("train", "val", "test")
JSON_KEYS = {"train": "X_train", "val": "X_val", "test": "X_test"}
WINDOW_LENGTH = 12
HEIGHT = 10
WIDTH = 20
SLOTS_PER_DAY = 48


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "UniST" / "dataset" / "TaxiNYC_short.json",
        help="Path to TaxiNYC_short.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root.parent / "dataset" / "train_data" / "npy_file",
        help="Destination read by UrbanDiT when it is launched from src/",
    )
    parser.add_argument(
        "--swap-channels",
        action="store_true",
        help="Treat the first sample group as outflow instead of inflow",
    )
    return parser.parse_args()


def load_timestamps(mm: mmap.mmap) -> Dict[str, np.ndarray]:
    key = b'"timestamps":'
    offset = mm.find(key)
    if offset < 0:
        raise ValueError("JSON does not contain a timestamps object")

    # timestamps is the final, small object in this JSON (about 600 KiB).
    payload = json.loads(b"{" + mm[offset:])
    timestamps = payload["timestamps"]
    result: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        array = np.asarray(timestamps[split], dtype=np.int64)
        if array.ndim != 3 or array.shape[1:] != (WINDOW_LENGTH, 2):
            raise ValueError(
                f"Unexpected timestamp shape for {split}: {array.shape}; "
                f"expected [samples, {WINDOW_LENGTH}, 2]"
            )
        result[split] = array
    return result


def extract_primary_windows(
    mm: mmap.mmap, json_key: str, sample_count: int
) -> np.ndarray:
    """Extract X_<split>[0] while skipping UniST's large periodic context."""
    marker = f'"{json_key}":'.encode()
    key_offset = mm.find(marker)
    if key_offset < 0:
        raise ValueError(f"JSON does not contain {json_key}")

    outer_start = mm.find(b"[", key_offset + len(marker))
    primary_start = mm.find(b"[", outer_start + 1)
    if outer_start < 0 or primary_start < 0:
        raise ValueError(f"Malformed array for {json_key}")

    # X_<split> is [primary_windows, periodic_context]. The primary tensor is
    # [sample, time, height, width], so its final value is followed by four
    # closing brackets. The periodic 5-D tensor then starts with five brackets.
    separator = mm.find(b"]]]], [[[[[", primary_start)
    if separator < 0:
        raise ValueError(f"Cannot locate primary/periodic separator for {json_key}")
    primary_end = separator + 4

    # Replacing brackets with spaces lets NumPy's C parser read all numbers
    # directly, without materialising millions of Python float objects.
    bracket_translation = bytes.maketrans(b"[]", b"  ")
    numeric_text = mm[primary_start:primary_end].translate(bracket_translation)
    flat = np.fromstring(numeric_text, sep=",", dtype=np.float32)
    expected = sample_count * WINDOW_LENGTH * HEIGHT * WIDTH
    if flat.size != expected:
        raise ValueError(
            f"Unexpected value count for {json_key}: {flat.size}; expected {expected}"
        )
    return flat.reshape(sample_count, WINDOW_LENGTH, HEIGHT, WIDTH)


def restore_windows(windows: np.ndarray, label: str) -> np.ndarray:
    if windows.shape[0] == 0:
        raise ValueError(f"No windows found for {label}")
    if windows.shape[0] > 1 and not np.array_equal(
        windows[:-1, 1:], windows[1:, :-1]
    ):
        raise ValueError(f"Overlapping windows are inconsistent in {label}")
    return np.concatenate((windows[0], windows[1:, -1]), axis=0)


def restore_timestamp_windows(windows: np.ndarray, label: str) -> np.ndarray:
    if windows.shape[0] > 1 and not np.array_equal(
        windows[:-1, 1:], windows[1:, :-1]
    ):
        raise ValueError(f"Overlapping timestamps are inconsistent in {label}")
    return np.concatenate((windows[0], windows[1:, -1]), axis=0)


def find_time_gaps(timestamps: np.ndarray) -> List[Tuple[int, List[int], List[int]]]:
    absolute_slots = timestamps[:, 0] * SLOTS_PER_DAY + timestamps[:, 1]
    deltas = (absolute_slots[1:] - absolute_slots[:-1]) % (7 * SLOTS_PER_DAY)
    indices = np.flatnonzero(deltas != 1) + 1
    return [
        (int(i), timestamps[i - 1].tolist(), timestamps[i].tolist()) for i in indices
    ]


def save_outputs(
    output_dir: Path,
    inflow: np.ndarray,
    outflow: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "TaxiNYCIn_48.npy": inflow.astype(np.float32, copy=False),
        "TaxiNYCIn_48_ts.npy": timestamps.astype(np.int64, copy=False),
        "TaxiNYCOut_48.npy": outflow.astype(np.float32, copy=False),
        "TaxiNYCOut_48_ts.npy": timestamps.astype(np.int64, copy=False),
    }
    for filename, array in outputs.items():
        path = output_dir / filename
        np.save(path, array, allow_pickle=False)
        print(f"wrote {path}  shape={array.shape} dtype={array.dtype}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    restored: Dict[str, List[np.ndarray]] = {"first": [], "second": [], "ts": []}
    with input_path.open("rb") as file_obj:
        with mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            timestamps = load_timestamps(mm)
            for split in SPLITS:
                ts_windows = timestamps[split]
                total_samples = ts_windows.shape[0]
                if total_samples % 2:
                    raise ValueError(
                        f"{split} has an odd sample count ({total_samples}); "
                        "cannot split it into inflow/outflow groups"
                    )
                channel_samples = total_samples // 2

                first_ts = ts_windows[:channel_samples]
                second_ts = ts_windows[channel_samples:]
                if not np.array_equal(first_ts, second_ts):
                    raise ValueError(
                        f"The two channel timestamp groups differ in {split}"
                    )

                windows = extract_primary_windows(
                    mm, JSON_KEYS[split], total_samples
                )
                first = restore_windows(
                    windows[:channel_samples], f"{split}/first-channel"
                )
                second = restore_windows(
                    windows[channel_samples:], f"{split}/second-channel"
                )
                split_ts = restore_timestamp_windows(first_ts, f"{split}/timestamps")

                restored["first"].append(first)
                restored["second"].append(second)
                restored["ts"].append(split_ts)
                print(
                    f"restored {split}: samples/channel={channel_samples}, "
                    f"frames/channel={first.shape[0]}"
                )

    first_channel = np.concatenate(restored["first"], axis=0)
    second_channel = np.concatenate(restored["second"], axis=0)
    continuous_ts = np.concatenate(restored["ts"], axis=0)

    if first_channel.shape != second_channel.shape:
        raise ValueError("Inflow and outflow shapes differ")
    if first_channel.shape[0] != continuous_ts.shape[0]:
        raise ValueError("Data and timestamp lengths differ")

    gaps = find_time_gaps(continuous_ts)
    if gaps:
        print("time gaps retained from source JSON (no values were invented):")
        for index, previous, current in gaps:
            print(f"  frame {index}: {previous} -> {current}")

    if args.swap_channels:
        inflow, outflow = second_channel, first_channel
    else:
        inflow, outflow = first_channel, second_channel

    save_outputs(args.output_dir.resolve(), inflow, outflow, continuous_ts)


if __name__ == "__main__":
    main()
