from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CELL_COUNT = 4
THUMB_STRIP_WIDTH = 420
THUMB_CELL_WIDTH = 92
CONTACT_MARGIN = 24
CONTACT_ROW_HEIGHT = 190


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export a QA page and PNG contact sheet for runtime failure digit cells."
  )
  parser.add_argument(
    "--dataset-root",
    default="data/runtime_failure_dataset",
    help="Runtime failure dataset root produced by export_runtime_digit_failure_set.py."
  )
  parser.add_argument(
    "--output-root",
    default="../output/runtime-failure-dataset-qa",
    help="Directory where timestamped QA artifacts are written."
  )
  parser.add_argument(
    "--timestamp",
    default="",
    help="Optional timestamp folder name. Defaults to current UTC time."
  )
  parser.add_argument(
    "--selection-log-json",
    default="../output/digit-classifier-recovery/baseline-ui-run.json",
    help=(
      "Optional UI benchmark JSON containing selectionLogs. "
      "When present, matching runtime candidates are tagged as UI-selected."
    )
  )
  parser.add_argument(
    "--selected-only",
    action="store_true",
    help="Render only the UI-selected failure candidates from the selection log."
  )
  parser.add_argument(
    "--annotation-csv",
    default="../docs/ocr-runtime-failure-selected-taxonomy.csv",
    help="Optional CSV with manual failure buckets keyed by filename and candidate."
  )
  return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def load_selected_sources(selection_log_path: Path) -> dict[str, str]:
  if not selection_log_path.exists():
    return {}
  payload = json.loads(selection_log_path.read_text(encoding="utf-8"))
  selected_sources: dict[str, str] = {}
  for log in payload.get("selectionLogs", []):
    if not isinstance(log, dict):
      continue
    image = str(log.get("image") or "").strip()
    selected = log.get("selected") if isinstance(log.get("selected"), dict) else {}
    source_label = str(selected.get("sourceLabel") or "").strip()
    if image and source_label:
      selected_sources[image] = source_label
  return selected_sources


def load_annotations(annotation_path: Path) -> dict[tuple[str, str], dict[str, str]]:
  if not annotation_path.exists():
    return {}
  annotations: dict[tuple[str, str], dict[str, str]] = {}
  for row in read_manifest(annotation_path):
    filename = str(row.get("filename") or "").strip()
    candidate = str(row.get("candidate") or "").strip()
    if filename and candidate:
      annotations[(filename, candidate)] = row
  return annotations


def copy_asset(source: Path, assets_dir: Path, name: str) -> str:
  target = assets_dir / name
  target.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, target)
  return f"assets/{target.name}"


def collect_cell_paths(dataset_root: Path, strip_stem: str) -> list[Path]:
  cells_root = dataset_root / "sections_labeled" / "train"
  paths = sorted(cells_root.glob(f"*/{strip_stem}__c*.png"))
  return sorted(paths, key=lambda path: path.stem.rsplit("__c", 1)[-1])


def resize_width(image: Image.Image, width: int) -> Image.Image:
  if image.width == width:
    return image.copy()
  scale = width / max(1, image.width)
  height = max(1, int(round(image.height * scale)))
  return image.resize((width, height), Image.Resampling.BILINEAR)


def render_contact_sheet(
  rows: list[dict[str, object]],
  output_path: Path
) -> None:
  width = CONTACT_MARGIN * 2 + THUMB_STRIP_WIDTH + CELL_COUNT * THUMB_CELL_WIDTH + 220
  height = CONTACT_MARGIN * 2 + max(1, len(rows)) * CONTACT_ROW_HEIGHT
  canvas = Image.new("RGB", (width, height), "#f6f1e8")
  draw = ImageDraw.Draw(canvas)
  font = ImageFont.load_default()

  y = CONTACT_MARGIN
  for index, row in enumerate(rows, start=1):
    filename = str(row["filename"])
    candidate = str(row["candidate"])
    expected = str(row["expected"])
    predicted = str(row["predicted"])
    strip_path = Path(str(row["strip_path"]))
    cell_paths = [Path(str(path)) for path in row["cell_paths"]]
    selected_text = " | UI SELECTED" if row.get("is_selected") else ""

    draw.text(
      (CONTACT_MARGIN, y),
      f"{index}. {filename} | {candidate}{selected_text} | expected {expected} | predicted {predicted}",
      fill="#7c2d12" if row.get("is_selected") else "#111827",
      font=font
    )
    image_y = y + 24
    if strip_path.exists():
      with Image.open(strip_path) as source:
        strip = resize_width(source.convert("RGB"), THUMB_STRIP_WIDTH)
      canvas.paste(strip, (CONTACT_MARGIN, image_y))

    cell_x = CONTACT_MARGIN + THUMB_STRIP_WIDTH + 24
    for cell_index, cell_path in enumerate(cell_paths):
      if not cell_path.exists():
        continue
      with Image.open(cell_path) as source:
        cell = resize_width(source.convert("RGB"), THUMB_CELL_WIDTH)
      canvas.paste(cell, (cell_x, image_y))
      draw.rectangle(
        (cell_x, image_y, cell_x + THUMB_CELL_WIDTH, image_y + cell.height),
        outline="#f97316",
        width=2
      )
      draw.text((cell_x, image_y + cell.height + 4), f"cell {cell_index + 1}", fill="#374151", font=font)
      cell_x += THUMB_CELL_WIDTH + 8

    y += CONTACT_ROW_HEIGHT

  output_path.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(output_path)


def write_html(
  rows: list[dict[str, object]],
  summary: dict[str, object],
  html_path: Path,
  contact_sheet_name: str,
  selected_only: bool
) -> None:
  cards: list[str] = []
  bucket_counts: dict[str, int] = {}
  for row in rows:
    bucket = str(row.get("failure_bucket") or "unannotated")
    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
  bucket_summary = " ".join(
    f"<code>{html.escape(bucket)}: {count}</code>"
    for bucket, count in sorted(bucket_counts.items())
  )
  for index, row in enumerate(rows, start=1):
    filename = html.escape(str(row["filename"]))
    candidate = html.escape(str(row["candidate"]))
    expected = html.escape(str(row["expected"]))
    predicted = html.escape(str(row["predicted"]))
    confidence = html.escape(str(row.get("avg_confidence", "")))
    failure_bucket = html.escape(str(row.get("failure_bucket", "")))
    notes = html.escape(str(row.get("failure_notes", "")))
    strip_href = html.escape(str(row["strip_href"]))
    is_selected = bool(row.get("is_selected"))
    selected_badge = '<span class="selected-badge">UI SELECTED</span>' if is_selected else ''
    card_class = "card selected-card" if is_selected else "card"
    cells = []
    for cell_index, href in enumerate(row["cell_hrefs"], start=1):
      cells.append(
        f"""
        <figure>
          <img src="{html.escape(str(href))}" alt="{filename} {candidate} cell {cell_index}">
          <figcaption>cell {cell_index}</figcaption>
        </figure>
        """
      )
    cards.append(
      f"""
      <article class="{card_class}">
        <h2>{index}. {filename} {selected_badge}</h2>
        <p>
          <span>candidate <strong>{candidate}</strong></span>
          <span>expected <strong>{expected}</strong></span>
          <span>predicted <strong>{predicted}</strong></span>
          <span>avg confidence <strong>{confidence}</strong></span>
          {f'<span>failure bucket <strong>{failure_bucket}</strong></span>' if failure_bucket else ''}
        </p>
        {f'<p class="notes">{notes}</p>' if notes else ''}
        <div class="images">
          <figure class="strip">
            <img src="{strip_href}" alt="{filename} {candidate} strip">
            <figcaption>runtime strip</figcaption>
          </figure>
          <div class="cells">
            {''.join(cells)}
          </div>
        </div>
      </article>
      """
    )

  payload = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Jarvis Runtime Failure Digit QA</title>
  <style>
    :root {{
      color: #172033;
      background: #f6f1e8;
      font-family: Georgia, 'Times New Roman', serif;
    }}
    body {{
      margin: 24px;
    }}
    a {{
      color: #075985;
    }}
    .summary, .card {{
      border: 1px solid #cbd5e1;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.78);
      padding: 18px;
      margin: 18px 0;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    .summary code, .card span {{
      display: inline-block;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 4px 10px;
      margin: 3px;
      background: #f8fafc;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    .selected-badge {{
      border-color: #f97316 !important;
      background: #ffedd5 !important;
      color: #9a3412;
      font-size: 14px;
      vertical-align: middle;
    }}
    .selected-card {{
      border: 3px solid #f97316;
      background: #fff7ed;
    }}
    .notes {{
      color: #334155;
      font-style: italic;
    }}
    .images {{
      display: grid;
      grid-template-columns: minmax(360px, 1.2fr) minmax(360px, 1fr);
      gap: 18px;
      align-items: start;
    }}
    figure {{
      margin: 0;
    }}
    figcaption {{
      margin-top: 4px;
      color: #64748b;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
    }}
    img {{
      max-width: 100%;
      image-rendering: auto;
      border-radius: 8px;
      background: #102033;
    }}
    .cells {{
      display: grid;
      grid-template-columns: repeat(4, minmax(70px, 1fr));
      gap: 8px;
    }}
    .cells img {{
      border: 2px solid #f97316;
    }}
  </style>
</head>
<body>
  <h1>Jarvis Runtime Failure Digit QA</h1>
  <section class="summary">
    <p>Validate that these train-only runtime failure cells are useful hard examples before fine-tuning.</p>
    <p>Annotate the cards tagged <strong>UI SELECTED</strong> first; those are the candidates the normal UI actually chose for that source image.</p>
    <p>
      <code>failure candidates: {len(rows)}</code>
      <code>UI-selected failure candidates: {sum(1 for row in rows if row.get('is_selected'))}</code>
      <code>selected-only view: {html.escape(str(selected_only).lower())}</code>
      <code>exported cells: {html.escape(str(summary.get('exported_cells', '')))}</code>
      <code>images total: {html.escape(str(summary.get('images_total', '')))}</code>
    </p>
    <p>{bucket_summary}</p>
    <p><a href="{html.escape(contact_sheet_name)}">Open PNG contact sheet</a></p>
  </section>
  {''.join(cards)}
</body>
</html>
"""
  html_path.write_text(payload, encoding="utf-8")


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve_path(base_dir, args.dataset_root)
  output_root = resolve_path(base_dir, args.output_root)
  selection_log_path = resolve_path(base_dir, args.selection_log_json) if args.selection_log_json else None
  annotation_path = resolve_path(base_dir, args.annotation_csv) if args.annotation_csv else None
  timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  output_dir = output_root / timestamp
  assets_dir = output_dir / "assets"
  selected_sources = load_selected_sources(selection_log_path) if selection_log_path else {}
  annotations = load_annotations(annotation_path) if annotation_path else {}

  manifest_path = dataset_root / "manifests" / "runtime_failure_candidates.csv"
  summary_path = dataset_root / "manifests" / "summary.json"
  if not manifest_path.exists():
    raise FileNotFoundError(f"Runtime failure manifest not found: {manifest_path}")
  summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
  manifest_rows = read_manifest(manifest_path)

  qa_rows: list[dict[str, object]] = []
  for row in manifest_rows:
    if row.get("status") != "failure":
      continue
    filename = row.get("filename", "")
    candidate = row.get("candidate", "")
    if not filename or not candidate:
      continue
    is_selected = selected_sources.get(filename) == candidate
    annotation = annotations.get((filename, candidate), {})
    strip_stem = f"{Path(filename).stem}__{candidate}"
    strip_path = dataset_root / "strips" / f"{strip_stem}.png"
    cell_paths = collect_cell_paths(dataset_root, strip_stem)
    if len(cell_paths) != CELL_COUNT or not strip_path.exists():
      continue
    strip_href = copy_asset(strip_path, assets_dir, f"{strip_stem}__strip.png")
    cell_hrefs = [
      copy_asset(path, assets_dir, f"{strip_stem}__{path.parent.name}__{path.name}")
      for path in cell_paths
    ]
    qa_rows.append({
      **row,
      "is_selected": is_selected,
      "failure_bucket": annotation.get("failure_bucket", ""),
      "failure_notes": annotation.get("notes", ""),
      "strip_path": strip_path,
      "cell_paths": cell_paths,
      "strip_href": strip_href,
      "cell_hrefs": cell_hrefs
    })

  if args.selected_only:
    qa_rows = [row for row in qa_rows if row.get("is_selected")]

  output_dir.mkdir(parents=True, exist_ok=True)
  contact_sheet_path = output_dir / "runtime-failure-dataset-contact-sheet.png"
  html_path = output_dir / "runtime-failure-dataset-qa.html"
  render_contact_sheet(qa_rows, contact_sheet_path)
  write_html(
    rows=qa_rows,
    summary=summary,
    html_path=html_path,
    contact_sheet_name=contact_sheet_path.name,
    selected_only=args.selected_only
  )

  selected_rows = sum(1 for row in qa_rows if row.get("is_selected"))
  bucket_counts: dict[str, int] = {}
  for row in qa_rows:
    bucket = str(row.get("failure_bucket") or "unannotated")
    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
  report = {
    "html": str(html_path),
    "contact_sheet": str(contact_sheet_path),
    "rows": len(qa_rows),
    "selected_rows": selected_rows,
    "selected_only": args.selected_only,
    "bucket_counts": bucket_counts,
    "selection_log_json": str(selection_log_path) if selection_log_path else None,
    "annotation_csv": str(annotation_path) if annotation_path else None
  }
  (output_dir / "runtime-failure-dataset-summary.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8"
  )
  print(json.dumps({
    **report,
    "summary_json": str(output_dir / "runtime-failure-dataset-summary.json")
  }, indent=2))


if __name__ == "__main__":
  main()
