"""Shared, standard-library source-exclusion parsing for data and UI evaluation."""

from __future__ import annotations

import csv
import io
from pathlib import Path

SOURCE_EXCLUSION_HEADERS = [
  "filename",
  "scope",
  "reason",
  "retention",
  "notes",
]
FULL_IMAGE_DIGIT_SCOPE = "full_image_digit_detection"

def parse_source_exclusions(text: str, path: Path) -> dict[str, dict[str, str]]:
  rows = list(csv.DictReader(io.StringIO(text)))
  exclusions: dict[str, dict[str, str]] = {}
  for row in rows:
    normalized = {
      header: (row.get(header) or "").strip()
      for header in SOURCE_EXCLUSION_HEADERS
    }
    filename = normalized["filename"]
    if not filename:
      raise ValueError(f"Source exclusion row is missing filename: {path}")
    if filename in exclusions:
      raise ValueError(f"Duplicate source exclusion for {filename}: {path}")
    if normalized["scope"] != FULL_IMAGE_DIGIT_SCOPE:
      raise ValueError(
        f"Unsupported source exclusion scope for {filename}: "
        f"{normalized['scope']!r}"
      )
    if not normalized["reason"]:
      raise ValueError(f"Source exclusion is missing a reason for {filename}")
    if normalized["retention"] != "legacy_stress":
      raise ValueError(
        f"Source exclusion retention for {filename} must be 'legacy_stress'"
      )
    exclusions[filename] = normalized
  return exclusions


def read_source_exclusions(path: Path) -> dict[str, dict[str, str]]:
  return parse_source_exclusions(path.read_text(encoding="utf-8") if path.exists() else "", path)
