# Backend Guidelines

## Scope
- This file covers `backend/` runtime, API, dataset, and training guidance.

## Setup and Run
- Create the backend virtualenv with `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- For any Python task that depends on computer-vision packages or image tooling (for example `ultralytics`, `opencv`, or `Pillow`), use `backend/.venv` rather than the system Python.
- Run the API with `cd backend && source .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8001 --reload`.
- In this Codex environment, if the service must be consumed by the DevTools browser, starting `uvicorn` inside the sandbox may not be reachable from the page. Prefer restarting it with escalated permissions when browser fetches to `127.0.0.1:8001` fail.

## API and Runtime Expectations
- Frontend/backend default ports are `127.0.0.1:8000` and `127.0.0.1:8001`.
- Health check: `GET /health` should report `ready: true`, `roi_ready: true`, `digit_ready: true`, `strip_digit_ready: true`, `strip_digit_23xx_ready: true`, and the expected model paths when all checkpoints are present.
- Default ROI endpoint: `http://127.0.0.1:8001/roi/detect`
- Default digit endpoint: `http://127.0.0.1:8001/digit/predict-cells`
- Default strip-reader shadow endpoint: `http://127.0.0.1:8001/digit/predict-strip`
- Default constrained `23xx` strip-reader shadow endpoint: `http://127.0.0.1:8001/digit/predict-strip-23xx`
- Backend default ROI model is pinned to `backend/models/roi-rotaug-e30-640.pt` (override with `ROI_MODEL_PATH`).
- Digit classifier path is `backend/models/digit_classifier.pt`.
- Strip digit reader path is `backend/models/digit_strip_reader.pt`.
- Constrained `23xx` strip digit reader path is `backend/models/digit_strip_reader_23xx.pt`.

## Dataset and Training Commands
- `cd backend && source .venv/bin/activate && python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --rotation-angles 90,180,270,360 --heavy-augment`: Fine-tune the ROI detector.
- `cd backend && source .venv/bin/activate && python extract_digit_windows.py --clean`: Rebuild split-wise digit windows from ROI labels.
- `cd backend && source .venv/bin/activate && python split_digit_windows.py --clean`: Canonicalize and split digit windows into 4 equispaced sections.
- `cd backend && source .venv/bin/activate && python label_digit_sections.py --clean`: Build labeled section datasets.
- `cd backend && source .venv/bin/activate && python validate_digit_dataset.py`: Validate the current windows/canonical/sections digit dataset.
- `cd backend && source .venv/bin/activate && python generate_synthetic_digit_dataset.py --clean --direct-per-real 6 --compose-window-count 180`: Generate synthetic train-only digit sections.
- `cd backend && source .venv/bin/activate && python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`: Refresh the targeted capture checklist.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu`: Train the real-only digit classifier.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu --synthetic-root data/digit_dataset/sections_synthetic --synthetic-target-ratio 2.0`: Train on mixed real + synthetic data while keeping val/test real-only.
- `cd backend && source .venv/bin/activate && python train_strip_digit_reader.py --device cpu`: Train the fixed four-head whole-strip reader from `data/digit_dataset/windows_canonical`.
- `cd backend && source .venv/bin/activate && python train_strip_digit_reader_23xx.py --device cpu`: Train the guarded house-specific `23xx` shadow reader from `data/digit_dataset/windows_canonical`.

## House-Specific Strip Reader Shortcut
- The constrained `23xx` strip reader is implemented as a shadow-only experiment. It uses a binary guard for whether the second digit is `3` plus two suffix digit heads; it only emits `23xx` when the guard confidence reaches `0.98`.
- This is valid only while the local water meter is expected to remain in the `2300`-`2399` range; review the assumption at least yearly and whenever readings approach `2390`.
- Keep benchmark comparison against the unconstrained four-head reader and current primary OCR path. The first checkpoint is diagnostic-only: cross-validation looked conservative (`0` guard false positives, `19` guard false negatives), but runtime QA still found accepted wrong predictions. Lowering the guard from `0.98` to `0.80` accepted more wrong values, so threshold tuning is not enough.

## Backend Policy
- `train_roi.py` should keep heavy augmentation and rotation expansion `90,180,270,360`; weaker runs require explicit override with `--allow-no-augment-policy`.
- Treat promoted checkpoints under `backend/models/*.pt` as must-retain artifacts and keep DVC state up to date when models or datasets change.
- Keep host/CORS scoped to localhost unless there is an explicit deployment task.

## Digit Dataset Expansion Loop (`4/5/6/9`)
1. Refresh capture planning with `python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`.
2. Add labeled captures with QA previews.
3. Confirm `data/digit_dataset/manifests/{windows.csv,canonical_windows.csv,sections.csv,section_labels.csv}` are regenerated and consistent with current splits.
4. Retrain the per-cell classifier only after class coverage improves.
5. Retrain the strip reader after canonical windows change; keep it shadow-only until UI benchmark exact-match and `MAE` beat the current primary path.
6. Promote new checkpoints only when benchmarked OCR `MAE` improves without exact-match or no-read regressions.
