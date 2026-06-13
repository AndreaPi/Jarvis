"""Export a visual QA report for canonical digit-strip windows."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = ROOT / "data" / "digit_dataset"
DEFAULT_OUTPUT_ROOT = ROOT.parent / "output" / "strip-dataset-qa"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy_images(rows: list[dict[str, str]], dataset_root: Path, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = dataset_root / row["canonical_window_path"]
        target = image_dir / f"{Path(row['canonical_window_path']).stem}.png"
        shutil.copy2(source, target)
        row["qa_image_path"] = f"images/{target.name}"


def _draw_contact_sheet(rows: list[dict[str, str]], dataset_root: Path, output_path: Path) -> None:
    label_width = 430
    image_width = 520
    row_height = 190
    padding = 18
    sheet_width = label_width + image_width + padding * 3
    sheet_height = padding + max(1, len(rows)) * row_height
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        top = padding + index * row_height
        left_text = padding
        text_lines = [
            f"{index + 1}. {row['filename']}",
            f"split: {row['split']}",
            f"reading: {row['reading']}",
            row["canonical_window_path"],
        ]
        for line_index, line in enumerate(text_lines):
            draw.text((left_text, top + line_index * 18), line, fill=(20, 30, 45), font=font)

        source = dataset_root / row["canonical_window_path"]
        with Image.open(source) as image:
            image = image.convert("RGB")
            scale = min(image_width / image.width, 150 / image.height)
            resized = image.resize((round(image.width * scale), round(image.height * scale)))
            image_x = label_width + padding * 2
            image_y = top + (150 - resized.height) // 2
            sheet.paste(resized, (image_x, image_y))
            draw.rectangle(
                [image_x, image_y, image_x + resized.width - 1, image_y + resized.height - 1],
                outline=(210, 220, 232),
            )

        draw.line(
            [(padding, top + row_height - 12), (sheet_width - padding, top + row_height - 12)],
            fill=(230, 235, 242),
        )

    sheet.save(output_path)


def _write_html(rows: list[dict[str, str]], output_path: Path, generated_at: str) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['split'])}</td>"
            f"<td>{html.escape(row['filename'])}</td>"
            f"<td>{html.escape(row['reading'])}</td>"
            f"<td>{html.escape(row['canonical_window_path'])}</td>"
            f"<td><img src=\"{html.escape(row['qa_image_path'])}\" alt=\"{html.escape(row['filename'])}\"></td>"
            "</tr>"
        )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Strip Reader Canonical Window QA</title>
  <style>
    :root {{
      color: #172033;
      background: #f7f1e6;
      font-family: Georgia, "Times New Roman", serif;
    }}
    body {{ margin: 24px; }}
    h1 {{ font-size: 42px; margin: 0 0 12px; }}
    p {{ font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fffdf8; }}
    th, td {{ border: 1px solid #d7dfeb; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #edf3f8; }}
    td {{ font-size: 16px; }}
    img {{ max-width: 760px; width: 100%; image-rendering: auto; }}
    .path {{ font-family: Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>Strip Reader Canonical Window QA</h1>
  <p>Generated {html.escape(generated_at)}. Validate that each strip is upright, readable left-to-right, tightly enough framed, and labeled with the expected four-digit reading.</p>
  <p>Rows: {len(rows)}. <a href="strip-canonical-windows-contact-sheet.png">Open PNG contact sheet</a></p>
  <table>
    <thead>
      <tr>
        <th>Split</th>
        <th>Filename</th>
        <th>Reading</th>
        <th>Canonical Path</th>
        <th>Canonical Image</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=_timestamp())
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "manifests" / "canonical_windows.csv"
    rows = _load_rows(manifest_path)

    output_dir = (args.output_root / args.timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_images(rows, dataset_root, output_dir / "images")

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    contact_sheet_path = output_dir / "strip-canonical-windows-contact-sheet.png"
    html_path = output_dir / "strip-canonical-windows-qa.html"
    _draw_contact_sheet(rows, dataset_root, contact_sheet_path)
    _write_html(rows, html_path, generated_at)

    print(f"Rows: {len(rows)}")
    print(f"HTML: {html_path}")
    print(f"PNG: {contact_sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
