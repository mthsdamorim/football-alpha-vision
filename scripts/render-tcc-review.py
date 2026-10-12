"""Render source-backed stills for human review of the frozen GSR references."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def render(packet: Path) -> int:
    review = json.loads((packet / "review.json").read_text())["frames"]
    rows = (
        json.loads(line)
        for line in (packet / "reference.jsonl").read_text().splitlines()
    )
    output = packet / "review-stills"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    count = 0
    by_clip: dict[str, list[Path]] = {}
    for row in rows:
        key = f"{row['clip']}:{row['source_frame']}"
        if key not in review:
            continue
        with Image.open(row["image_path"]) as original:
            image = original.convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for group in row["pitch_lines"]:
            for name, points in group.items():
                if len(points) < 2:
                    continue
                pixels = [
                    (round(p["x"] * width), round(p["y"] * height)) for p in points
                ]
                draw.line(pixels, fill="#ff00ff", width=3)
                draw.text(
                    pixels[0],
                    name,
                    fill="#ffffff",
                    stroke_width=2,
                    stroke_fill="#770077",
                )
        for person in row["people"]:
            box = person["bbox_image"]
            x, y = box["x"], box["y"]
            draw.rectangle(
                (x, y, x + box["w"], y + box["h"]), outline="#00e5ff", width=3
            )
            draw.text(
                (x, max(0, y - 14)),
                f"{person['role']} ID {person['track_id']}",
                fill="#ffffff",
                stroke_width=2,
                stroke_fill="#004455",
            )
            support = person["support_image"]
            sx, sy = support["x"], support["y"]
            color = (
                "#ff3333"
                if support["visibility"] == "image_edge_truncated"
                else "#ffff00"
            )
            draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), fill=color)
        draw.text(
            (16, 12),
            f"{key} | {row['metric_reference_status']}",
            fill="#ffffff",
            stroke_width=2,
            stroke_fill="#000000",
        )
        draw.text(
            (16, 30),
            "support: yellow=pending review, red=image-edge truncated",
            fill="#ffffff",
            stroke_width=2,
            stroke_fill="#000000",
        )
        still = output / f"{row['clip']}-{row['source_frame']:06}.jpg"
        image.save(still, quality=94, subsampling=0)
        by_clip.setdefault(row["clip"], []).append(still)
        count += 1
    if count != len(review):
        raise ValueError(f"rendered {count} of {len(review)} review frames")
    for clip, stills in by_clip.items():
        if len(stills) != 10:
            raise ValueError(f"{clip}: expected ten review stills")
        sheet = Image.new("RGB", (1920, 2700))
        for index, still in enumerate(stills):
            with Image.open(still) as image:
                thumbnail = image.resize((960, 540), Image.Resampling.LANCZOS)
            sheet.paste(thumbnail, ((index % 2) * 960, (index // 2) * 540))
        sheet.save(output / f"contact-{clip}.jpg", quality=92)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    print(f"Rendered {render(args.packet)} review stills")
