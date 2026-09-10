# Make Sense digit-box review

Use for a new review request or correction when no matching reviewed export is
already available. Inspect target crops under
`output/full-image-digit-review/previews/crops/` at full resolution;
`digit-box-contact-sheet.jpg` is an overview only.

Lead each initial or corrective request with a prominent clickable
[Open Make Sense](https://www.makesense.ai/) link, localized to the user's
language. Prefer the link over automatically opening an external browser.
In the same message provide absolute local links to every target image, the
class list, and matching annotation files, plus the exact expected ZIP or
directory export path. Do not rely on the user finding an earlier message.

Ask the user to:

1. Upload only target images from `output/full-image-digit-review/images/`.
2. Choose **Object Detection** and load the class list from
   `output/full-image-digit-review/annotations/labels.txt`.
3. Import only matching `<meter-stem>.txt` files from that annotations directory;
   do not import `labels.txt` as annotations.
4. Keep exactly four boxes per image, each covering a complete digit-wheel
   aperture rather than just the dark glyph.
5. Verify classes `0..9` form the trusted four-digit reading in reading order,
   correct boxes/classes, retain canonical filenames, and export YOLO annotations.

The export is required evidence of human review even if bootstrap boxes look
correct. Once it is supplied, validate/import the intended subset through
`backend/import_full_image_digit_annotations.py`, then rebuild derived labels
and previews as described in the main skill. Human approval of regenerated
previews is not required a second time when the export is imported faithfully
and the main skill's consistency checks and agent visual QA pass. Import success
alone is insufficient. Ask for renewed review only for unresolved discrepancies,
changed source/box geometry or orientation, or visual ambiguity.
