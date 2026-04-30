# Jarvis

Jarvis is a lightweight personal assistant web app. The first module helps you read a water meter photo, review the detected value, and draft an email in Gmail.

## Documentation

- Docs index: [`docs/README.md`](./docs/README.md)
- OCR app logic flow: [`docs/app-logic.md`](./docs/app-logic.md)
- Backend API guide: [`docs/backend-api.md`](./docs/backend-api.md)
- OCR tuning playbook: [`docs/ocr-tuning-playbook.md`](./docs/ocr-tuning-playbook.md)

## Features
- Upload a meter photo and preview it.
- OCR from a neural-ROI crop with conservative acceptance (unsupported OCR guesses are rejected to manual input).
- Auto-fill an email draft with the current date in Italian format.
- Open a Gmail draft or use a mailto fallback.
- Run a built-in OCR test set table with `Detected`, `Absolute Error`, and `Failure Reason` columns plus MAE/exact-match/no-read summary stats.

## Local Development
1. Ensure Python 3 and Node.js are installed.
2. Run the dev server:

```bash
npm run serve
```

Then open `http://localhost:8000`.

If you also want to run Playwright checks, install JS dependencies once:

```bash
npm install
```

### GitHub Push Setup

This repository should push through SSH as `AndreaPi`. Configure the repo-local
remote and SSH key before pushing from a machine with multiple GitHub identities:

```bash
git remote set-url origin git@github.com:AndreaPi/Jarvis.git
git config core.sshCommand "ssh -i ~/.ssh/id_ed25519_andreapi -o IdentitiesOnly=yes"
git config user.name "AndreaPi"
git config user.email "8233615+AndreaPi@users.noreply.github.com"
```

The `core.sshCommand` setting is intentionally local to this checkout so Git uses
the AndreaPi key for this repo without changing other repositories on the same
machine.

### Optional Neural ROI Backend (recommended)
You can run a Python backend that detects the meter digit window using a fine-tuned pretrained model.

1. Open a second terminal and set up backend dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use `backend/.venv` for any Python workflow that touches images or CV dependencies such as `ultralytics`, `opencv`, or `Pillow`.

For CPU-only environments (for example Vercel), install:

```bash
pip install -r requirements-cpu.txt
```


2. Train/fine-tune a model (copies best checkpoint to `backend/models/roi.pt`):

```bash
python train_roi.py \
  --data data/roi_dataset.yaml \
  --base-model yolov8n.pt \
  --rotation-angles 90,180,270,360 \
  --heavy-augment
```

The API default ROI checkpoint is pinned to `backend/models/roi-rotaug-e30-640.pt`.
To run with a newly trained checkpoint, set `ROI_MODEL_PATH` explicitly before starting the backend.
`train_roi.py` now enforces heavy augmentation + rotation expansion by default; weaker runs require explicit `--allow-no-augment-policy`.

Optional: train the per-cell digit classifier checkpoint:

```bash
python train_digit_classifier.py --device cpu
```

## Artifact Retention

Treat the following as Tier 1 artifacts that must not be lost:

- raw meter photos in `assets/`
- `assets/meter_readings.csv`
- `backend/data/roi_dataset/images/**`
- `backend/data/roi_dataset/labels/**`
- `backend/data/roi_dataset/splits.json`
- `backend/data/digit_dataset/manifests/**`
- `backend/data/digit_dataset/sections_synthetic/manifests/**`
- promoted checkpoints in `backend/models/*.pt`

The repo now uses DVC for the large Tier 1 binaries:

```bash
uv pip install --python backend/.venv/bin/python "dvc[s3]"
```

Currently tracked by DVC:

- each raw meter photo in `assets/` via per-file `*.dvc` pointers
- `backend/data/roi_dataset/images` via `backend/data/roi_dataset/images.dvc`
- `backend/data/digit_dataset/windows` via `backend/data/digit_dataset/windows.dvc`
- `backend/data/digit_dataset/windows_canonical` via `backend/data/digit_dataset/windows_canonical.dvc`
- `backend/data/digit_dataset/sections` via `backend/data/digit_dataset/sections.dvc`
- `backend/data/digit_dataset/sections_labeled` via `backend/data/digit_dataset/sections_labeled.dvc`
- `backend/data/digit_dataset/sections_synthetic/train` via `backend/data/digit_dataset/sections_synthetic/train.dvc`
- promoted model weights in `backend/models/*.pt` via per-file `*.dvc` pointers

After dataset ingestion or model promotion:

```bash
source backend/.venv/bin/activate
dvc add backend/data/roi_dataset/images
dvc add backend/data/digit_dataset/windows
dvc add backend/data/digit_dataset/windows_canonical
dvc add backend/data/digit_dataset/sections
dvc add backend/data/digit_dataset/sections_labeled
dvc add backend/data/digit_dataset/sections_synthetic/train
dvc add backend/models/*.pt
find assets -maxdepth 1 -type f ! -name 'meter_readings.csv' ! -name '*:Zone.Identifier' -print0 | xargs -0 dvc add
scripts/dvc-push-safe.sh
```

Do not run raw `dvc push` directly in this repo. `scripts/dvc-push-safe.sh` activates `backend/.venv`, checks that a default DVC remote is configured, and refuses to push if the remote is a plain local path instead of a cloud/object-store URL.

Configure an off-machine DVC remote once before using `scripts/dvc-push-safe.sh`.

Backblaze B2 is a good default choice for this repo because the current artifact footprint is tiny and fits comfortably within B2's free storage tier. DVC talks to B2 through its S3-compatible endpoint:

```bash
source backend/.venv/bin/activate
dvc remote add -d b2 s3://<bucket-name>/jarvis-dvc
dvc remote modify b2 endpointurl https://s3.<region>.backblazeb2.com
dvc remote modify --local b2 access_key_id <key-id>
dvc remote modify --local b2 secret_access_key <application-key>
```

`dvc[s3]` must be installed in `backend/.venv` before this works. The access key and secret should stay in `.dvc/config.local`, not in committed repo config.

Then create a backup archive when you want a releaseable snapshot:

```bash
scripts/package-tier1-artifacts.sh
```

For GitHub-hosted retention, set the `DVC_REMOTE_URL`, `DVC_REMOTE_ACCESS_KEY_ID`, and `DVC_REMOTE_SECRET_ACCESS_KEY` repository secrets (plus optional `DVC_REMOTE_SESSION_TOKEN` if your remote uses temporary credentials), then use the manual `Publish Artifacts` workflow to `dvc pull`, package, and upload a release snapshot.

For dataset expansion/QA before retraining:

```bash
python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9
python validate_digit_dataset.py
```

3. Start the API:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

In the Codex/DevTools environment, a backend started inside the sandbox may answer shell `curl` but still be unreachable from the browser. If the page still gets `ERR_CONNECTION_REFUSED` or `Failed to fetch` for `127.0.0.1:8001`, restart the backend outside the sandbox with escalated permissions and verify from the page context.

By default, the frontend calls `http://127.0.0.1:8001/roi/detect` and requires neural ROI detection before OCR.
Digit decoding is neural-classifier-only and calls `http://127.0.0.1:8001/digit/predict-cells`.
Check backend readiness with:

```bash
curl -s http://127.0.0.1:8001/health
```

### E2E Tests

Run Playwright checks for neural-ROI failure handling and OCR selection guard regressions:

```bash
npm run test:e2e
```

Generate a per-image ROI checkpoint comparison report (`roi-rotaug-e30-640.pt` vs `roi.pt`) with stage `5/6` debug snapshots:

```bash
npm run benchmark:roi-diff
```

This benchmark requires all three local model files to be present:

- `backend/models/roi-rotaug-e30-640.pt`
- `backend/models/roi.pt`
- `backend/models/digit_classifier.pt`

Report artifacts are written under `output/roi-checkpoint-diff/<timestamp>/`.
Per-image diff tables include selected OCR metadata (`sourceLabel`, `method`, `preprocessMode`) and stage `6` exports use the last `6. OCR input candidate` frame from each debug session (the winning decode strip variant).

CI runs these tests on every pull request and on pushes to `master`.

## File Overview
- `index.html`: UI layout.
- `styles.css`: Styling.
- `app.js`: Thin entrypoint that imports `src/main.js`.
- `src/main.js`: UI orchestration and event wiring.
- `src/ocr/`: OCR pipeline and neural ROI integration.
- `src/testset/`: Manual OCR test-set runner.
- `backend/`: Optional FastAPI service for neural ROI and digit-classifier inference/training.
- `AGENTS.md`: Repo-wide contributor guide.
- `backend/AGENTS.md`: Backend-specific runtime and training guidance.
- `src/ocr/AGENTS.md`: OCR-specific behavior, benchmarks, and tuning guidance.
- `assets/`: Static assets and example uploads.

## Notes
- OCR now relies on neural ROI detection; if the backend is unavailable or ROI fails, the app asks for manual reading input.
- Digit decoding uses the backend neural classifier endpoint (`/digit/predict-cells`) and is enabled by default.
- Edge-derived ROI strip candidates are enabled by default and can be toggled with `OCR_CONFIG.roiDeterministic.useEdgeCandidates`.
- The selection layer prioritizes edge-derived strips, but the primary classifier pass now also includes top base-strip candidates when they are available; a narrow base fallback rerun is still available only when base candidates were not already evaluated and edge support remains weak. Low-confidence edge-only reads can still be rejected at the final gate.
- Use the UI `Run test set` action plus `npm run test:e2e` for OCR regressions before and after tuning.
- The Gmail flow opens a draft; you always review and send manually.

## Asset Naming (Meter Images)
- Use the EXIF `DateTimeOriginal` value as the source of truth for the acquisition date.
- Rename files to `meter_yyyymmdd` (zero-padded) and keep the original extension.
- If multiple images share the same date, keep one as-is and add numeric suffixes to the rest (e.g., `_1`, `_2`).
- If EXIF is missing, prefer a known date from the filename or capture notes and document it.
