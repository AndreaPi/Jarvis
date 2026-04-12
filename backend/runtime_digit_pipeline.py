from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

CELL_OVERLAP = 0.03
MIN_CANDIDATE_WIDTH = 120
MIN_CANDIDATE_HEIGHT = 28
MIN_CANDIDATE_ASPECT = 0.12
MAX_CANDIDATE_ASPECT = 18.0
MIN_STRIP_ASPECT = 1.45
MAX_STRIP_ASPECT = 8.2
DESKEW_MAX_ANGLE = 8
DESKEW_STEP = 2
TIGHTEN_INK_RATIO = 0.18
NORMALIZE_WIDTH = 520
HARD_STRIP_MIN_FACTOR = 0.96
HARD_STRIP_MAX_FACTOR = 1.06


@dataclass(frozen=True)
class CropRect:
  left: int
  top: int
  right: int
  bottom: int

  @property
  def width(self) -> int:
    return self.right - self.left

  @property
  def height(self) -> int:
    return self.bottom - self.top


@dataclass(frozen=True)
class NormalizedStrip:
  image: Image.Image
  deskew_angle: int
  major_axis_rotation: int


def clamp(value: float, minimum: float, maximum: float) -> float:
  return min(maximum, max(minimum, value))


def normalize_angle(angle: float) -> int:
  return int(((angle % 360) + 360) % 360)


def resolve_crop_rect(image: Image.Image, rect: dict[str, float]) -> CropRect:
  x = clamp(rect["x"], 0, image.width - 1)
  y = clamp(rect["y"], 0, image.height - 1)
  width = clamp(rect["width"], 1, image.width - x)
  height = clamp(rect["height"], 1, image.height - y)
  left = int(round(x))
  top = int(round(y))
  right = int(round(x + width))
  bottom = int(round(y + height))
  right = max(left + 1, min(image.width, right))
  bottom = max(top + 1, min(image.height, bottom))
  return CropRect(left=left, top=top, right=right, bottom=bottom)


def crop_image(image: Image.Image, rect: dict[str, float]) -> Image.Image:
  crop_rect = resolve_crop_rect(image, rect)
  return image.crop((crop_rect.left, crop_rect.top, crop_rect.right, crop_rect.bottom))


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
  normalized = normalize_angle(angle)
  if normalized == 0:
    return image
  if normalized == 90:
    return image.transpose(Image.Transpose.ROTATE_270)
  if normalized == 180:
    return image.transpose(Image.Transpose.ROTATE_180)
  if normalized == 270:
    return image.transpose(Image.Transpose.ROTATE_90)
  return image.rotate(normalized, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)


def resize_to_width(image: Image.Image, target_width: int) -> Image.Image:
  if image.width == target_width:
    return image
  scale = target_width / max(1, image.width)
  new_height = max(1, int(round(image.height * scale)))
  return image.resize((target_width, new_height), Image.Resampling.BILINEAR)


def to_grayscale_array(image: Image.Image) -> np.ndarray:
  return np.asarray(image.convert("L"), dtype=np.float32)


def has_valid_candidate_geometry(image: Image.Image) -> bool:
  width = image.width
  height = image.height
  aspect = width / max(1, height)
  return (
    width >= MIN_CANDIDATE_WIDTH
    and height >= MIN_CANDIDATE_HEIGHT
    and MIN_CANDIDATE_ASPECT <= aspect <= MAX_CANDIDATE_ASPECT
  )


def expand_interval_to_min_length(
  start: int,
  end: int,
  min_length: int,
  limit: int
) -> tuple[int, int]:
  current = end - start
  if current >= min_length:
    return start, end
  min_length = min(min_length, limit)
  deficit = min_length - current
  grow_before = deficit // 2
  grow_after = deficit - grow_before
  start = max(0, start - grow_before)
  end = min(limit, end + grow_after)
  current = end - start
  if current >= min_length:
    return start, end
  if start == 0:
    end = min(limit, min_length)
  elif end == limit:
    start = max(0, limit - min_length)
  return start, end


def tighten_crop_by_ink(
  image: Image.Image,
  min_area_ratio: float = 0.15,
  min_minor_axis_ratio: float = 0.70
) -> Image.Image:
  gray = to_grayscale_array(image)
  dark = 255.0 - gray
  cols = dark.sum(axis=0)
  rows = dark.sum(axis=1)
  mean_cols = float(cols.mean())
  mean_rows = float(rows.mean())
  max_cols = float(cols.max())
  max_rows = float(rows.max())
  col_threshold = mean_cols + (max_cols - mean_cols) * 0.25
  row_threshold = mean_rows + (max_rows - mean_rows) * 0.25

  col_indices = np.flatnonzero(cols > col_threshold)
  row_indices = np.flatnonzero(rows > row_threshold)
  if not len(col_indices) or not len(row_indices):
    return image

  left = int(col_indices[0])
  right = int(col_indices[-1])
  top = int(row_indices[0])
  bottom = int(row_indices[-1])
  if right <= left or bottom <= top:
    return image

  padding_x = int(round((right - left) * 0.08))
  padding_y = int(round((bottom - top) * 0.15))
  left = int(clamp(left - padding_x, 0, image.width - 1))
  right = int(clamp(right + padding_x, 1, image.width))
  top = int(clamp(top - padding_y, 0, image.height - 1))
  bottom = int(clamp(bottom + padding_y, 1, image.height))
  crop_width = right - left
  crop_height = bottom - top
  min_target = int(round(min(image.width, image.height) * min_minor_axis_ratio))
  if crop_width <= crop_height and crop_width < min_target:
    left, right = expand_interval_to_min_length(left, right, min_target, image.width)
    crop_width = right - left
  elif crop_height < min_target:
    top, bottom = expand_interval_to_min_length(top, bottom, min_target, image.height)
    crop_height = bottom - top
  area_ratio = (crop_width * crop_height) / max(1, image.width * image.height)
  if area_ratio < min_area_ratio or area_ratio > 0.95:
    return image

  return image.crop((left, top, right, bottom))


def score_deskew_candidate(image: Image.Image, angle: int) -> tuple[Image.Image, float]:
  rotated = image if angle == 0 else rotate_image(image, angle)
  tightened = tighten_crop_by_ink(rotated, TIGHTEN_INK_RATIO)
  scoring_image = tightened
  if scoring_image.height > scoring_image.width:
    scoring_image = rotate_image(scoring_image, 90)
  rotated_scoring = rotated
  if rotated_scoring.height > rotated_scoring.width:
    rotated_scoring = rotate_image(rotated_scoring, 90)
  aspect = scoring_image.width / max(1, scoring_image.height)
  area_ratio = (tightened.width * tightened.height) / max(1, rotated.width * rotated.height)
  height_ratio = scoring_image.height / max(1, rotated_scoring.height)
  score = aspect - max(0.0, 0.32 - area_ratio) * 6.0 - max(0.0, 0.58 - height_ratio) * 9.0
  return tightened, score


def build_ink_projection(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
  gray = to_grayscale_array(image)
  ink = 255.0 - gray
  columns = ink.sum(axis=0)
  rows = ink.sum(axis=1)
  return columns, rows


def find_max_ink_window_start(values: np.ndarray, window_size: int) -> int:
  if values.size == 0:
    return 0
  size = min(values.size, max(1, int(round(window_size))))
  window_sum = float(values[:size].sum())
  best_sum = window_sum
  best_start = 0
  for index in range(size, values.size):
    window_sum += float(values[index] - values[index - size])
    start = index - size + 1
    if window_sum > best_sum:
      best_sum = window_sum
      best_start = start
  return best_start


def normalize_roi_strip(image: Image.Image) -> NormalizedStrip | None:
  candidate_bases = (
    [(rotate_image(image, 90), 90)]
    if image.height > image.width
    else [(image, 0)]
  )

  base_image, base_rotation = candidate_bases[0]
  best_image, best_score = score_deskew_candidate(base_image, 0)
  best_angle = 0
  best_base_rotation = base_rotation
  best_valid_image: Image.Image | None = None
  best_valid_score: float | None = None
  best_valid_angle = 0
  best_valid_base_rotation = base_rotation

  def has_valid_post_rotation_geometry(candidate_image: Image.Image) -> bool:
    candidate = candidate_image
    if candidate.height > candidate.width:
      candidate = rotate_image(candidate, 90)
    return has_valid_candidate_geometry(candidate)

  if has_valid_post_rotation_geometry(best_image):
    best_valid_image = best_image
    best_valid_score = best_score
    best_valid_angle = best_angle
    best_valid_base_rotation = best_base_rotation

  for base_image, base_rotation in candidate_bases:
    base_angles = [0]
    for delta in range(DESKEW_STEP, DESKEW_MAX_ANGLE + 1, DESKEW_STEP):
      base_angles.extend((delta, -delta))
    for angle in base_angles:
      if base_rotation == best_base_rotation and angle == 0:
        continue
      candidate_image, candidate_score = score_deskew_candidate(base_image, angle)
      if candidate_score > best_score:
        best_image = candidate_image
        best_score = candidate_score
        best_angle = angle
        best_base_rotation = base_rotation
      if has_valid_post_rotation_geometry(candidate_image):
        if best_valid_score is None or candidate_score > best_valid_score:
          best_valid_image = candidate_image
          best_valid_score = candidate_score
          best_valid_angle = angle
          best_valid_base_rotation = base_rotation

  normalized = best_valid_image if best_valid_image is not None else best_image
  best_angle = best_valid_angle if best_valid_image is not None else best_angle
  base_rotation = best_valid_base_rotation if best_valid_image is not None else best_base_rotation
  major_axis_rotation = base_rotation
  if normalized.height > normalized.width:
    normalized = rotate_image(normalized, 90)
    major_axis_rotation = normalize_angle(major_axis_rotation + 90)
  if not has_valid_candidate_geometry(normalized):
    return None

  aspect = normalized.width / max(1, normalized.height)
  if aspect < MIN_STRIP_ASPECT:
    target_height = max(
      MIN_CANDIDATE_HEIGHT,
      min(normalized.height, int(round(normalized.width / MIN_STRIP_ASPECT)))
    )
    if target_height < normalized.height:
      _, rows = build_ink_projection(normalized)
      start_y = find_max_ink_window_start(rows, target_height)
      normalized = crop_image(normalized, {
        "x": 0,
        "y": start_y,
        "width": normalized.width,
        "height": target_height
      })
  elif aspect > MAX_STRIP_ASPECT:
    target_width = max(
      MIN_CANDIDATE_WIDTH,
      min(normalized.width, int(round(normalized.height * MAX_STRIP_ASPECT)))
    )
    if target_width < normalized.width:
      columns, _ = build_ink_projection(normalized)
      start_x = find_max_ink_window_start(columns, target_width)
      normalized = crop_image(normalized, {
        "x": start_x,
        "y": 0,
        "width": target_width,
        "height": normalized.height
      })

  normalized = resize_to_width(normalized, NORMALIZE_WIDTH)
  if normalized.height > normalized.width:
    normalized = rotate_image(normalized, 90)
    major_axis_rotation = normalize_angle(major_axis_rotation + 90)
  if not has_valid_candidate_geometry(normalized):
    return None

  aspect = normalized.width / max(1, normalized.height)
  hard_min = MIN_STRIP_ASPECT * HARD_STRIP_MIN_FACTOR
  hard_max = MAX_STRIP_ASPECT * HARD_STRIP_MAX_FACTOR
  if aspect < hard_min or aspect > hard_max:
    return None

  return NormalizedStrip(
    image=normalized,
    deskew_angle=normalize_angle(best_angle),
    major_axis_rotation=major_axis_rotation
  )


def build_cell_rects(
  image: Image.Image,
  count: int,
  overlap_ratio: float = CELL_OVERLAP,
  x_offset_px: float = 0.0,
  per_section_x_offsets: list[float] | None = None
) -> list[CropRect]:
  rects: list[CropRect] = []
  cell_width = image.width / count
  overlap = cell_width * overlap_ratio
  for index in range(count):
    offset = x_offset_px
    if per_section_x_offsets is not None and index < len(per_section_x_offsets):
      offset = per_section_x_offsets[index]
    rects.append(resolve_crop_rect(image, {
      "x": cell_width * index - overlap + offset,
      "y": 0,
      "width": cell_width + overlap * 2,
      "height": image.height
    }))
  return rects


def split_into_cells(
  image: Image.Image,
  count: int,
  overlap_ratio: float = CELL_OVERLAP,
  x_offset_px: float = 0.0,
  per_section_x_offsets: list[float] | None = None
) -> list[Image.Image]:
  return [
    image.crop((rect.left, rect.top, rect.right, rect.bottom))
    for rect in build_cell_rects(image, count, overlap_ratio, x_offset_px, per_section_x_offsets)
  ]
