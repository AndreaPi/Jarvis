# Manual ROI labels

Use for initial labels, supplied-label validation, or corrections to the
selected batch. The ROI covers the complete four-digit black register window.

When labels or corrections are needed, lead the request with the clickable
[Open Make Sense](https://www.makesense.ai/) link, localized to the user's
language. Prefer the link over automatically opening an external browser.
Provide clickable absolute local links to every target canonical image and the
exact expected output filenames in the same message, including on corrections.

Ask for one exported YOLO TXT per requested image in `assets/`:

- Filename: canonical stem plus `.txt`, e.g. `assets/meter_20260521.txt` for
  `assets/meter_20260521.JPEG`.
- Exactly one row: `0 x_center y_center width height`, normalized to the full
  image, with class `0` and a positive, finite, in-bounds box.

Reuse valid matching exports already provided. Missing or invalid labels block
the affected batch sync; list exact missing/corrective filenames. Ignore and
report unrelated extra exports without importing or deleting them. If an export
uses pre-normalization or ambiguous names, inspect it to diagnose the mismatch
and use the recorded source mapping or the user's explicit mapping. Ask only
when that mapping is not established; never infer it from file order.

The user's supplied export is evidence of manual review on the canonical input
image, including when it corrects a previous box. Retain the original export
coordinates and source-image identity for comparison after rebuilding.

Once the batch is complete, return to the skill's manifest-sync and rebuild
steps, preserving existing ROI splits. Apply the main skill's consistency and
agent visual checks to the generated overlays. A faithful import needs no second
human overlay confirmation. Request review only for changed source/box geometry,
unresolved import discrepancies, or visual ambiguity; agent-proposed corrections
must not be treated as human-reviewed exports.
