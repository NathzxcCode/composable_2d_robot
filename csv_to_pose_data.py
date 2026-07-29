"""
Convert a tab/comma-separated encoder+distance CSV to pose_data.json.

CSV format (no header):
    encoder_0   encoder_1   distance_cm

Usage:
    python csv_to_pose_data.py input.csv output.json
    python csv_to_pose_data.py input.csv             # writes pose_data.json

Optional overrides (all have defaults matching the two-limb robot):
    --limb-lengths   17.5 17.5
    --sensor-offsets 8.7 8.7
"""

import argparse
import csv
import json
import sys


DEFAULTS = {
    "limb_lengths":    [17.5, 17.5],
    "sensor_offsets":  [8.7,  8.7],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file",          help="Input CSV file")
    p.add_argument("output", nargs="?", default="pose_data.json",
                   help="Output JSON file (default: pose_data.json)")
    p.add_argument("--limb-lengths",   type=float, nargs="+",
                   default=DEFAULTS["limb_lengths"],
                   help="Limb length per limb (cm)")
    p.add_argument("--sensor-offsets", type=float, nargs="+",
                   default=DEFAULTS["sensor_offsets"],
                   help="Sensor x-offset per limb (cm)")
    return p.parse_args()


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Accept tab or comma separation
            parts = line.replace(",", "\t").split()
            if len(parts) < 3:
                continue
            # Skip header rows
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return rows


def build_pose_data(rows, limb_lengths, sensor_offsets):
    n_limbs = len(limb_lengths)
    assert len(sensor_offsets) == n_limbs, "sensor-offsets count must match limb count"
    # CSV columns: encoder_0, encoder_1, ..., encoder_(n-1), distance
    assert len(rows[0]) == n_limbs + 1, \
        f"Expected {n_limbs + 1} columns per row, got {len(rows[0])}"

    poses = []
    for row in rows:
        encoders = row[:n_limbs]
        distance = row[n_limbs]

        limbs = []
        for i, (angle, length, offset) in enumerate(
                zip(encoders, limb_lengths, sensor_offsets)):
            limbs.append({
                "id":            i,
                "depth":         i,
                "local_angle":   angle,
                "limb_length":   length,
                "sensor_offset": {"x": offset, "y": 0},
            })

        # One connection per adjacent limb pair (parent -> child)
        connections = []
        for i in range(1, n_limbs):
            connections.append({
                "parent_id": i - 1,
                "child_id":  i,
                "depth":     i,
                "distance":  distance,
            })

        poses.append({"limbs": limbs, "connections": connections})

    return poses


def main():
    args = parse_args()

    rows = load_csv(args.csv_file)
    if not rows:
        sys.exit("No data rows found in CSV.")

    poses = build_pose_data(rows, args.limb_lengths, args.sensor_offsets)

    with open(args.output, "w") as f:
        json.dump(poses, f, indent=2)

    print(f"Wrote {len(poses)} poses to {args.output}")


if __name__ == "__main__":
    main()
