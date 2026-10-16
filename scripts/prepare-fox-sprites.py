"""Extract the generated 4x3 grid, retaining one scale and a shared nose anchor.

This packages generated artwork; it does not draw or synthesize poses. Seams
follow transparent gutters so a nose crossing a nominal cell edge is retained.
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/fox-dash/source/generated-grid.png"
OUTPUT = ROOT / "assets/fox-dash"
FRAME_SIZE = 512


def main():
    source = Image.open(SOURCE).convert("RGBA")
    pieces, anchors = [], []
    for row in range(3):
        band = source.crop((0, round(row * source.height / 3), source.width,
                            round((row + 1) * source.height / 3)))
        alpha = np.asarray(band)[:, :, 3]
        seams = [0]
        for column in range(1, 4):
            nominal = round(column * source.width / 4)
            low, high = nominal - 45, nominal + 45
            empty = np.flatnonzero((alpha[:, low:high] > 16).sum(axis=0) == 0) + low
            if len(empty) == 0:
                raise ValueError(f"No clear gutter at row {row + 1}, column {column}")
            seams.append(int(empty[len(empty) // 2]))
        seams.append(source.width)
        for column in range(4):
            slot = band.crop((seams[column], 0, seams[column + 1], band.height))
            bounds = slot.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox()
            if bounds is None:
                raise ValueError("Empty animation frame")
            piece = slot.crop(bounds)
            pixels = np.asarray(piece)
            # Black nose at the right tip, excluding the jaw and forehead.
            mask = ((pixels[:, :, :3].max(axis=2) < 80) & (pixels[:, :, 3] > 200))
            mask[:, :piece.width - 24] = False
            mask[:int(piece.height * .45)] = False
            mask[int(piece.height * .70):] = False
            nose_y, nose_x = np.nonzero(mask)
            if not len(nose_y):
                raise ValueError("Right-facing nose anchor not found")
            pieces.append(piece)
            anchors.append((float(nose_x.mean()), float(nose_y.mean())))

    # One common scale, never individual per-frame fitting.
    scale = min(FRAME_SIZE * .84 / max(p.width for p in pieces),
                FRAME_SIZE * .80 / max(p.height for p in pieces))
    nose = (FRAME_SIZE * .90, FRAME_SIZE * .54)
    sheet = Image.new("RGBA", (FRAME_SIZE * 4, FRAME_SIZE * 3))
    records = []
    frames_dir = OUTPUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, (piece, anchor) in enumerate(zip(pieces, anchors)):
        resized = piece.resize((round(piece.width * scale), round(piece.height * scale)),
                               Image.Resampling.LANCZOS)
        offset = (round(nose[0] - anchor[0] * scale), round(nose[1] - anchor[1] * scale))
        if (min(offset) < 0 or offset[0] + resized.width > FRAME_SIZE
                or offset[1] + resized.height > FRAME_SIZE):
            raise ValueError(f"Frame {index + 1} would be clipped")
        frame = Image.new("RGBA", (FRAME_SIZE, FRAME_SIZE))
        frame.alpha_composite(resized, offset)
        frame.save(frames_dir / f"{index + 1:02d}.png")
        sheet.alpha_composite(frame, ((index % 4) * FRAME_SIZE, (index // 4) * FRAME_SIZE))
        records.append({"frame": index + 1, "source_size": list(piece.size),
                        "source_nose": anchor, "offset": list(offset)})
    sheet.save(OUTPUT / "fox-dash-sheet.png")
    manifest = {"frames": 12, "columns": 4, "rows": 3, "frame_size": FRAME_SIZE,
                "direction": "right", "shared_scale": scale,
                "nose_uv": [.90, .54], "mouth_uv": [.62, .70],
                "closed_lip_spray_uv": [.78, .61],
                "closed_mouth_uv": [.69, .62], "extraction": records}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(records)} transparent frames and {OUTPUT / 'fox-dash-sheet.png'}")


if __name__ == "__main__":
    main()
