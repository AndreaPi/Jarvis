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
- Health check: `GET /health` should report `ready: true`, `roi_ready: true`, `digit_ready: true`, and the expected `model_path`.
- Default ROI endpoint: `http://127.0.0.1:8001/roi/detect`
- Default digit endpoint: `http://127.0.0.1:8001/digit/predict-cells`
- Backend default ROI model is pinned to `backend/models/roi-rotaug-e30-640.pt` (override with `ROI_MODEL_PATH`).
- Digit classifier path is `backend/models/digit_classifier.pt`.

## Dataset and Training Commands
- `cd backend && source .venv/bin/activate && python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --rotation-angles 90,180,270,360 --heavy-augment`: Fine-tune the ROI detector.
- `cd backend && source .venv/bin/activate && python extract_digit_windows.py --clean`: Rebuild split-wise digit windows from ROI labels.
- `cd backend && source .venv/bin/activate && python split_digit_windows.py --clean`: Canonicalize and split digit windows into 4 equispaced sections.
- `cd backend && source .venv/bin/activate && python label_digit_sections.py --clean`: Build labeled section datasets.
- `cd backend && source .venv/bin/activate && python validate_digit_dataset.py`: Validate legacy strip/cell manifests when using `build_digit_dataset.py`.
- `cd backend && source .venv/bin/activate && python generate_synthetic_digit_dataset.py --clean --direct-per-real 6 --compose-window-count 180`: Generate synthetic train-only digit sections.
- `cd backend && source .venv/bin/activate && python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`: Refresh the targeted capture checklist.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu`: Train the real-only digit classifier.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu --synthetic-root data/digit_dataset/sections_synthetic --synthetic-target-ratio 2.0`: Train on mixed real + synthetic data while keeping val/test real-only.

## Backend Policy
- `train_roi.py` should keep heavy augmentation and rotation expansion `90,180,270,360`; weaker runs require explicit override with `--allow-no-augment-policy`.
- Treat promoted checkpoints under `backend/models/*.pt` as must-retain artifacts and keep DVC state up to date when models or datasets change.
- Keep host/CORS scoped to localhost unless there is an explicit deployment task.

## Digit Dataset Expansion Loop (`4/5/6/9`)
1. Refresh capture planning with `python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`.
2. Add labeled captures with QA previews.
3. Confirm `data/digit_dataset/manifests/{windows.csv,canonical_windows.csv,sections.csv,section_labels.csv}` are regenerated and consistent with current splits.
4. Retrain the classifier only after class coverage improves.
5. Promote new checkpoints only when benchmarked OCR `MAE` improves without exact-match or no-read regressions.
