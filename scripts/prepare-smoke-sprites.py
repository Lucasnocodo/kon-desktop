"""Package one generated, registered 4x3 smoke sheet without moving artwork.

Every complete source cell receives the same resize. Alpha bounds are measured
for diagnostics only: no bbox crop, fringe removal, anchor fitting, or per-frame
scale is applied. Inspect the reported edge/padding notes visually before use.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/fox-smoke/source/generated-grid.png"
OUTPUT = ROOT / "assets/fox-smoke"
COLUMNS, ROWS, FRAME_SIZE = 4, 3, 512


def alpha_bounds(alpha: np.ndarray, threshold: int = 0) -> list[int] | None:
    """Return Pillow-style exclusive bounds, without changing any pixels."""
    ys, xs = np.nonzero(alpha > threshold)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def inspect_alpha(frame: Image.Image, index: int) -> dict:
    alpha = np.asarray(frame.getchannel("A"))
    bounds = alpha_bounds(alpha)
    if bounds is None:
        raise ValueError(f"Frame {index:02d} is empty")
    transparent_fraction = float(np.mean(alpha == 0))
    opaque_fraction = float(np.mean(alpha == 255))
    if transparent_fraction < .01:
        raise ValueError(
            f"Frame {index:02d} lacks a transparent background "
            f"({transparent_fraction:.2%} fully transparent pixels)"
        )

    # A fully opaque canvas or a nearly opaque rectangle with tiny alpha holes
    # must not be accepted as transparent smoke. Artwork is never keyed out.
    corner_size = max(1, round(min(frame.size) * .05))
    corner_patches = [alpha[:corner_size, :corner_size],
                      alpha[:corner_size, -corner_size:],
                      alpha[-corner_size:, :corner_size],
                      alpha[-corner_size:, -corner_size:]]
    corners_opaque = all(float(np.mean(patch >= 250)) >= .98
                         for patch in corner_patches)
    if corners_opaque and opaque_fraction > .90:
        raise ValueError(f"Frame {index:02d} has a suspicious opaque background")

    padding = [bounds[0], bounds[1], frame.width - bounds[2],
               frame.height - bounds[3]]
    visible_bounds = alpha_bounds(alpha, 16)
    perimeter = np.concatenate((alpha[0], alpha[-1], alpha[1:-1, 0],
                                alpha[1:-1, -1]))
    edge_alpha_max = int(perimeter.max())
    notes = []
    if edge_alpha_max:
        notes.append(
            f"Alpha reaches a cell edge (maximum {edge_alpha_max}); inspect "
            "for clipped wisps or generation spill. All edge pixels preserved."
        )
    if min(padding) < 8:
        notes.append(
            f"Less than 8 px clear padding on at least one side: {padding}. "
            "No crop, realignment, or fringe deletion was applied."
        )
    if visible_bounds is None:
        notes.append("Only faint alpha (<= 16) remains; inspect this fade frame.")
    return {
        "alpha_bounds": bounds,
        "visible_alpha_bounds": visible_bounds,
        "alpha_min": int(alpha.min()),
        "alpha_max": int(alpha.max()),
        "transparent_fraction": round(transparent_fraction, 6),
        "partial_alpha_fraction": round(float(np.mean((alpha > 0) & (alpha < 255))), 6),
        "opaque_fraction": round(opaque_fraction, 6),
        "padding_ltrb": padding,
        "edge_alpha_max": edge_alpha_max,
        "padding_notes": notes,
    }


def prepare(source_path: Path, output_path: Path) -> dict:
    with Image.open(source_path) as opened:
        if opened.mode != "RGBA":
            raise ValueError(f"Source must contain real RGBA transparency; mode is {opened.mode}")
        source = opened.copy()
    if source.width * ROWS != source.height * COLUMNS:
        raise ValueError(
            f"Source {source.width}x{source.height} violates the 4:3 sheet ratio; "
            "regenerate square registered cells instead of distorting them"
        )
    if source.width % COLUMNS or source.height % ROWS:
        raise ValueError("Source dimensions must divide evenly into exactly 4x3 cells")
    cell_width, cell_height = source.width // COLUMNS, source.height // ROWS
    records, frames, encoded_frames = [], [], []
    # Validate all cells before writing, so a bad cell cannot leave a partial set.
    for index in range(COLUMNS * ROWS):
        column, row = index % COLUMNS, index // COLUMNS
        box = (column * cell_width, row * cell_height,
               (column + 1) * cell_width, (row + 1) * cell_height)
        cell = source.crop(box)
        source_alpha = inspect_alpha(cell, index + 1)
        frame = cell.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS)
        frame_alpha = inspect_alpha(frame, index + 1)
        encoded = io.BytesIO()
        frame.save(encoded, format="PNG")
        data = encoded.getvalue()
        record = {
            "frame": index + 1,
            "file": f"frames/{index + 1:02d}.png",
            "source_cell": list(box),
            "source_alpha": source_alpha,
            **frame_alpha,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        records.append(record)
        frames.append(frame)
        encoded_frames.append(data)

    output_path.mkdir(parents=True, exist_ok=True)
    frames_path = output_path / "frames"
    frames_path.mkdir(exist_ok=True)
    sheet = Image.new("RGBA", (FRAME_SIZE * COLUMNS, FRAME_SIZE * ROWS))
    for index, (frame, data) in enumerate(zip(frames, encoded_frames)):
        (frames_path / f"{index + 1:02d}.png").write_bytes(data)
        # paste preserves the original RGBA bytes, including their transparency.
        sheet.paste(frame, ((index % COLUMNS) * FRAME_SIZE,
                            (index // COLUMNS) * FRAME_SIZE))
    sheet_path = output_path / "fox-smoke-sheet.png"
    sheet.save(sheet_path)
    manifest = {
        "frames": COLUMNS * ROWS,
        "columns": COLUMNS,
        "rows": ROWS,
        "frame_size": FRAME_SIZE,
        "pivot_uv": [.50, .62],
        "source": Path(os.path.relpath(source_path.resolve(), ROOT)).as_posix(),
        "source_dims": list(source.size),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_cell_size": [cell_width, cell_height],
        "source_generation": {"single_pass": True, "passes": 1,
                              "description": "All 12 frames generated together in one registered sheet"},
        "preserved_registration": True,
        "extraction": "Exact equal 4x3 cells; full-cell uniform LANCZOS resize",
        "shared_scale": FRAME_SIZE / cell_width,
        "alpha_validation": "Real RGBA, nonempty cells, transparent background; padding retained for visual inspection",
        "frame_records": records,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(frames)} registered transparent frames: {sheet_path}")
    for record in records:
        for note in record["source_alpha"]["padding_notes"]:
            print(f"Source frame {record['frame']:02d}: {note}")
        for note in record["padding_notes"]:
            print(f"Output frame {record['frame']:02d}: {note}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot prepare smoke sprites: {error}\n")


if __name__ == "__main__":
    main()
