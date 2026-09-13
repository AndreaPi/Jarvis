---
name: jarvis-meter-to-digit-box-sync
description: "Prepare, import, or correct human-reviewed full-image digit boxes for Jarvis meter photos with approved readings and ROIs."
---

# Jarvis Meter To Digit Box Sync

Execute this workflow from the Jarvis repository root. Use `backend/.venv` for
every Python step.

This skill requires verified upstream artifacts, not a fresh invocation of
`jarvis-meter-to-roi-sync`. It does not itself ingest raw photos, decide trusted
readings, approve register ROIs, train a detector, or promote a model.

Run mutating sequences fail-fast. Record `git status --short` first and preserve
unrelated Git and DVC changes. Do not rebuild while
`train_full_image_digit_detector.py` is running.

## Workflow

Choose the entry point from the user's request and the current artifact state:

- New approved photos: verify upstream data, seed missing boxes, and prepare review.
- Supplied Make Sense export: verify batch, upstream data, and orientation, then
  import without asking for the same export again.
- Existing-box corrections: use the requested canonical targets and current
  annotations; regenerate only as needed and review the changed boxes.
- Resumed approval: reuse explicit approval and passing checks for unchanged
  artifacts; continue at the first unmet step. A manifest row alone does not
  prove manual-export provenance, approved orientation, or successful visual QA.

The user's Make Sense export establishes human review of the digit boxes on
the annotated image. A verified equivalent import and agent visual QA satisfy
the final digit-box gate without a second human preview confirmation. Bootstrap
boxes alone never establish human review.

1. Establish the approved batch.
   - Carry forward the upstream batch DVC upload authorization and its QA
     evidence under [AGENTS.md](../../../AGENTS.md#artifact-retention). Check
     its filenames, artifact categories, and resolved destination. Consent
     covering this batch's digit derivatives is reusable without another
     upload question; consent limited to photos and ROI does not cover them.
   - Use canonical filenames from the user's request, a verified export, or the
     upstream handoff. Preserve their actual extension/case.
   - Do not silently add every unannotated file in the worktree to the batch.
   - For every target, verify:
     - the canonical photo exists in `assets/`;
     - `assets/meter_readings.csv` contains one trusted four-digit reading;
     - `backend/data/roi_boxes_manifest.json` contains the approved register
       ROI;
     - the matching image and one-box label exist in the persistent ROI split;
     - the photo and ROI image dataset DVC targets were published after the
       upstream ROI review gate passed: user-supplied manual labels, verified
       equivalent import, agent visual QA, and any required exceptional review.
   - If a prerequisite is missing, complete only the missing upstream work via
     `jarvis-meter-to-roi-sync` when already authorized. Otherwise report the
     concrete prerequisite and finish independent checks. Do not restart raw
     ingestion merely because this is a correction or resumed task.
   - Do not request another ROI overlay approval when the upstream gate already
     passed. ROI validation is not digit-box approval. Keep approvals tied to the same
     unchanged photo, reading, ROI, orientation, and annotation artifacts;
     request renewed review only for changed or unapproved artifacts.

2. Verify canonical reading orientation.
   - Confirm every target has one row in
     `backend/data/digit_dataset/manifests/canonical_windows.csv`, with the same
     split and reading as the approved upstream data.
   - If the row or matching QA is missing/stale, or orientation needs correction,
     read [orientation preparation](references/orientation-preparation.md).
   - Require explicit human confirmation of left-to-right reading order. Reuse
     existing approval only when its inputs and canonical strip are unchanged;
     otherwise show the target QA previews and request that review.
   - Record each explicit confirmation, rejection, or unclear result in
     `backend/data/digit_dataset/manifests/orientation_reviews.csv`, following
     the review-log rules in [orientation preparation](references/orientation-preparation.md).
     Preserve earlier failures and avoid counting a resumed approval twice.

3. Snapshot the full-image dataset boundary.
   - Read, do not infer, the existing:
     - `backend/data/full_image_digit_dataset/manifests/annotations.csv`;
     - `backend/data/full_image_digit_dataset/manifests/cv_folds.csv`;
     - `backend/data/full_image_digit_dataset/manifests/source_exclusions.csv`;
     - `backend/data/full_image_digit_dataset/manifests/summary.json`.
   - Record the target filenames, existing review states, exclusions, and CV
     assignments so the post-build comparison can detect unintended changes.
   - Preserve `source_exclusions.csv`. Do not exclude a new image because it is
     difficult without an explicit user decision.

4. Prepare missing annotations and review artifacts.
   - Run the builder when target rows or review artifacts are missing/stale:

     ```bash
     backend/.venv/bin/python backend/build_full_image_digit_dataset.py
     ```

   - Never pass `--rebuild-cv-folds` during ingestion.
   - For newly seeded targets, confirm exactly four rows in `annotations.csv`, at
     positions `0..3`, with digits/classes equal to the trusted reading,
     `annotation_source=bootstrap-roi-split`, and
     `review_status=pending`.
   - Confirm existing reviewed rows, existing fold assignments, and exclusions
     are unchanged. A new train source may receive one new persistent CV fold;
     a test source must not receive one.

5. Obtain human-reviewed boxes when missing or requiring correction.
   - Read [Make Sense review](references/makesense-review.md) when preparing a
     request or correction. Provide the exact target images, class list, matching
     annotation files, and expected export path with a clickable Make Sense link.
   - A Make Sense export is required even when bootstrap boxes look correct.
     Reuse an already supplied matching export; for unchanged imported reviewed
     boxes, retain their review provenance instead of requesting another export.
   - Await only the missing export or corrective input. Complete independent
     authorized checks and report prepared work while review is pending.

6. Import the reviewed subset.
   - Retain the supplied export (or its exact contents), its hash, and the
     annotated image's hash, dimensions, and EXIF orientation in batch QA
     evidence. Record the approved reading-direction rotation for comparison
     after import and on resumed work.
   - For a normal new-image batch, run:

     ```bash
     backend/.venv/bin/python backend/import_full_image_digit_annotations.py \
       /path/to/makesense-export.zip \
       --allow-partial
     backend/.venv/bin/python backend/build_full_image_digit_dataset.py
     ```

   - Omit `--allow-partial` only when the export intentionally contains every
     active review-package image.
   - Verify all coordinates are finite. Let the importer reject unknown images, out-of-bounds coordinates, any
     count other than four boxes, and class sequences that do not match the
     trusted reading. Do not bypass these checks.
   - The import must update `annotations.csv` as the source of truth. Derived
     YOLO labels, `summary.json`, fold assignment, and previews come from the
     following builder run; do not edit generated label files directly.

7. Enforce the final gate.
   - Confirm every target now has exactly four rows with:
     - positions `0..3`;
     - `annotation_source=human-makesense`;
     - `review_status=reviewed`;
     - classes equal to the trusted reading.
   - Confirm the target's derived YOLO label exists in its active split,
     exclusions are unchanged, old CV assignments are unchanged, and each new
     train target retains its newly seeded fold.
   - Run `npm run test:backend` once for the changed batch and complete the
     manifest/split checks before declaring validation complete. Reuse a passing
     run for unchanged code and artifacts; confirmation alone does not trigger
     another build or test run. Corrections require renewed affected checks.
   - Verify the reviewed image's identity, dimensions, EXIF orientation, and
     approved reading-direction rotation are unchanged. Sort exported boxes
     using that rotation, then compare all four classes and coordinates with
     `annotations.csv` and derived YOLO labels, allowing only serialization
     rounding (eight decimals in the manifest, six in YOLO labels).
   - Inspect the regenerated target crop previews at full resolution under
     `output/full-image-digit-review/previews/crops/`; each box must cover its
     complete digit-wheel aperture and the sequence must match the trusted reading.
   - When manual-export provenance, numerical checks, and agent visual QA agree,
     mark the target validated and continue authorized work without asking for
     another human confirmation. Show or link the preview for transparency and
     distinguish the user's manual review from the agent's import/visual checks.
   - Correct import defects that can be restored to the supplied export and
     rerun affected checks. Request human review only for unresolved discrepancies,
     changed source/box geometry or orientation, or visual ambiguity. A corrected
     manual export supplies review of that correction; an agent-proposed box
     change must not be accepted as human-reviewed without user review.
   - Declare the whole full-image dataset training-ready only when
     `summary.json` reports no active `pending` annotations and the affected
     targets pass this gate. Unresolved validation/review blocks training-ready
     claims and publication or commit of affected annotations. If unrelated
     pending rows remain, report the target as validated but the dataset as not
     training-ready. Passing this gate does not itself authorize training or
     upload, but an existing batch authorization covering the requested upload
     remains valid; do not ask again just because the workflow changed skills.

8. Report completion or the precise pending gate.
   - Report the target filenames and readings, imported row count, review
     status, regenerated labels/previews, new fold assignments, validation
     result, user approval state, and unrelated pre-existing changes.
   - Report any condition that prevents the training-ready claim.
   - Training, checkpoint evaluation/promotion, CV-fold rebuilding, release
     publishing, and Git commit/push require authorization in the user's
     request. If a downstream action is already authorized, continue when its
     prerequisites and applicable gates pass; do not require a separate turn.
     This skill alone does not authorize those actions.

## Invariants

- `annotations.csv` is the canonical human-review layer; generated labels are
  disposable derivatives.
- Bootstrap runs may seed missing rows but must never overwrite reviewed rows.
- `cv_folds.csv` is persistent. Do not recompute it during routine ingestion.
- `source_exclusions.csv` defines active scope while retaining legacy-stress
  artifacts.
- Validation and test images are never augmentation sources.
- Human Make Sense review, verified equivalent import, and agent visual QA are
  mandatory. Renew human review only for unresolved discrepancies or changed
  source/annotation geometry or orientation.
- Routine additive ingestion must preserve the synthetic digit dataset and its
  DVC pointer unchanged.
