# Canonical orientation preparation

Read only when the target canonical row/QA is missing or stale, or orientation
needs correction. Use checked, fail-fast commands from the repository root with
`backend/.venv`. Preserve unrelated changes, synthetic outputs, and their DVC
pointer; record their state before regeneration and compare afterward.

For missing or stale extracted windows, refresh the preparation pipeline:

```bash
backend/.venv/bin/python backend/extract_digit_windows.py
backend/.venv/bin/python backend/split_digit_windows.py --clean
backend/.venv/bin/python backend/label_digit_sections.py --clean
backend/.venv/bin/python backend/validate_digit_dataset.py
npm run qa:strip-dataset
```

Never use `extract_digit_windows.py --clean` for additive ingestion: it removes
the entire digit-dataset root, including synthetic artifacts and DVC pointers.
Reserve it for an explicitly requested full regeneration that rebuilds all
affected derived datasets and restores tracking. The two later `--clean` flags
clear only their respective derived output directories.

If only QA is missing/stale, run `npm run qa:strip-dataset` without rebuilding
valid data. Show target canonical-strip previews and require explicit human
confirmation that left-to-right order matches the trusted reading. An existing
canonical manifest row is insufficient evidence of approval. Reuse approval
when the reviewed strip, source photo, reading, ROI, and orientation inputs are
unchanged.

## Orientation review log

Keep `backend/data/digit_dataset/manifests/orientation_reviews.csv` as retained
Git metadata. Record actual human responses with the source and shown canonical
strip SHA-256 hashes, expected reading, applied rotation, `attempt` (`initial`,
`corrected`, or `unknown`), and `decision` (`confirmed`, `rejected`, or `unclear`).
Use `reported_order` only when established by the user's reply; leave it empty
when unknown. Keep the user-response evidence, the known agent model context
(otherwise `unknown`), and UTC recording time; recording time is not the time
of a retrospectively recovered review.

Reuse the row for the same filename/source hash/strip hash/reading/rotation
on resume. A corrected strip gets a new row and retains the earlier rejection.
If the user changes a verdict for the same strip, record the correction and its
evidence without treating it as another independent trial. Silence or a pending
reply is not a rejection. Backfill only responses with traceable image/strip
identity; do not infer historical success from a manifest row alone.

Report confirmations, rejections, and unclear outcomes, counting first attempts
per source separately from corrections. This measures the complete preparation
and review workflow, not model accuracy alone. A streak of confirmations does
not automatically remove the human gate; revisit it with the user after varied
cases have accumulated. Preserve this log during dataset regeneration.

## Corrections and publication

For an incorrect direction, correct
`backend/data/digit_dataset/manifests/direction_overrides.csv`, regenerate from
`split_digit_windows.py` onward, validate, and repeat affected QA and review.

When preparation changed the standard digit datasets and orientation is
approved, refresh only the changed targets from this set. Before the push,
reuse the batch upload authorization checked in the main skill. If these
derivatives or the destination are not covered, finish local preparation and
ask for the missing scope once, following
[AGENTS.md](../../../../AGENTS.md#artifact-retention). A prior authorization
covering these derivatives applies even when they were generated after consent:

```bash
backend/.venv/bin/python -m dvc add \
  backend/data/digit_dataset/windows \
  backend/data/digit_dataset/windows_canonical \
  backend/data/digit_dataset/sections \
  backend/data/digit_dataset/sections_labeled
scripts/dvc-push-safe.sh \
  backend/data/digit_dataset/windows.dvc \
  backend/data/digit_dataset/windows_canonical.dvc \
  backend/data/digit_dataset/sections.dvc \
  backend/data/digit_dataset/sections_labeled.dvc
backend/.venv/bin/python -m dvc status \
  backend/data/digit_dataset/windows.dvc \
  backend/data/digit_dataset/windows_canonical.dvc \
  backend/data/digit_dataset/sections.dvc \
  backend/data/digit_dataset/sections_labeled.dvc
```

Trim the example target list to the actual changed outputs; exclude unrelated
dirty targets. If DVC needs an alternate cache, use
`DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache`. Use only `scripts/dvc-push-safe.sh` with
a configured off-machine remote; never raw `dvc push`. Report publication
failures and remaining targets without claiming that backup is complete.
