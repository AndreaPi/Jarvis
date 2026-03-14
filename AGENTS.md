# Repository Guidelines

## Project Structure & Module Organization
- `index.html`: Single-page UI layout and content.
- `styles.css`: Global styles and visual system.
- `app.js`: Thin module entrypoint that imports `src/main.js`.
- `src/main.js`: UI orchestration and event wiring.
- `src/ocr/`: Neural-ROI-first OCR pipeline with strip-first decoding and selection safeguards.
- `src/email/`: Email draft generation and link helpers.
- `src/testset/`: Manual test-set runner logic.
- `src/debug/`: Debug overlay rendering helpers.
- `backend/`: Optional FastAPI service for neural ROI + digit classifier inference and training scripts.
- `backend/extract_digit_windows.py`: Export ROI digit windows from ROI labels into split folders.
- `backend/split_digit_windows.py`: Canonicalize window orientation and split each window into 4 equispaced sections.
- `backend/label_digit_sections.py`: Auto-assign section labels from the 4-digit reading string.
- `backend/build_digit_dataset.py`: Legacy strip/cell dataset exporter (kept for backward compatibility tooling).
- `backend/generate_synthetic_digit_dataset.py`: Build synthetic train-only digit sections (direct cell augmentation + optional composed windows re-split equispaced).
- `backend/plan_digit_expansion.py`: Generate prioritized capture plan for underrepresented digits.
- `backend/validate_digit_dataset.py`: Validate legacy strip/cell manifests and QA preview coverage.
- `backend/train_digit_classifier.py`: Train per-cell digit classifier checkpoint.
- `package.json`: Local dev scripts.
- `README.md`: Project overview and setup notes.
- `assets/`: Static assets and example uploads.

## Build, Test, and Development Commands
- `npm run serve`: Start a simple local web server on port 8000.
- `npm run dev`: Alias of `npm run serve`.
- `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`: Backend setup.
- For any Python task that depends on computer-vision packages or image tooling (for example `ultralytics`, `opencv`, or `Pillow`), use `backend/.venv` rather than the system Python.
- `cd backend && source .venv/bin/activate && python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --rotation-angles 90,180,270,360 --heavy-augment`: Fine-tune pretrained ROI detector with enforced augmentation policy.
- `cd backend && source .venv/bin/activate && python extract_digit_windows.py --clean`: Rebuild split-wise digit windows from ROI labels.
- `cd backend && source .venv/bin/activate && python split_digit_windows.py --clean`: Canonicalize + split windows into 4 equispaced sections.
- `cd backend && source .venv/bin/activate && python label_digit_sections.py --clean`: Build labeled section dataset (`sections_labeled/train|val|test/<digit>`).
- `cd backend && source .venv/bin/activate && python validate_digit_dataset.py`: Validate legacy strip/cell manifests (only needed when using `build_digit_dataset.py`).
- `cd backend && source .venv/bin/activate && python generate_synthetic_digit_dataset.py --clean --direct-per-real 6 --compose-window-count 180`: Generate synthetic train-only digit sections from real train labels.
- `cd backend && source .venv/bin/activate && python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`: Refresh targeted capture checklist.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu`: Train per-cell digit classifier model (real-only).
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu --synthetic-root data/digit_dataset/sections_synthetic --synthetic-target-ratio 2.0`: Train on mixed real + synthetic train split while keeping val/test real-only.
- `cd backend && source .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8001 --reload`: Run neural ROI API.

Open `http://localhost:8000` after running a serve command. Backend endpoints default to `http://127.0.0.1:8001/roi/detect` and `http://127.0.0.1:8001/digit/predict-cells`.

## Coding Style & Naming Conventions
- Use 2-space indentation in HTML/CSS/JS.
- Keep files ASCII-only unless there is a strong reason for Unicode.
- Use descriptive, lower-case IDs and class names (e.g., `photo-input`, `module-grid`).
- Prefer clear, small functions in `src/` modules and avoid deep nesting.

## Testing Guidelines
- Automated browser tests are configured with Playwright.
- `npm run test:e2e`: Runs `tests/e2e/neural-roi.spec.js` (neural ROI failure handling + ROI geometry + strip-only OCR behavior).
- CI: `.github/workflows/e2e.yml` runs on each pull request and on pushes to `master`.
- Frontend manual checks: upload image, run OCR, verify email draft fields, and confirm Gmail draft link.
- OCR test-set checks: run "Run test set" and inspect `MAE`, `Exact Match`, `No-read`, `Failure Reason`, and debug stages.
- Backend sanity checks: `GET /health` and confirm `ready: true`, `roi_ready: true`, and expected `model_path`.
- Prefer running the test set from UI with debug overlay enabled.
- Before committing OCR changes, run both `npm run test:e2e` and the UI "Run test set".
- ROI training policy: always use heavy augmentation and rotation expansion (`90,180,270,360`). `train_roi.py` enforces this by default and only allows weaker runs with `--allow-no-augment-policy`.

## Commit & Pull Request Guidelines
- No commit message convention is established in this repo.
- Suggested pattern: short, imperative subject (e.g., "Improve OCR preview").
- PRs should include: summary of changes, screenshots for UI changes, and any manual test notes.

## Security & Configuration Tips
- The Gmail draft flow opens a client-side draft; no credentials are stored in code.
- OCR runs in the browser; avoid adding API keys to the client without a secure proxy.
- Backend is intended for local use; keep host/CORS scoped to localhost unless explicitly deploying.

## Artifact Retention
- Treat raw `assets/` photos, `assets/meter_readings.csv`, ROI images/labels, digit manifests, and promoted `backend/models/*.pt` checkpoints as must-retain artifacts.
- Use DVC for large Tier 1 binaries:
  - per-file DVC tracking for raw meter photos in `assets/`
  - `backend/data/roi_dataset/images.dvc`
  - per-file DVC tracking for promoted `backend/models/*.pt`
- After dataset ingestion or model promotion, run the relevant `dvc add ...` commands, then `dvc push`.
- Use `scripts/package-tier1-artifacts.sh` plus the manual `Publish Artifacts` workflow for release-style snapshots after the DVC remote is up to date.

## IMPORTANT
- When using Playwright in this environment, global `playwright-cli` may be more reliable than the wrapper if npm network is flaky.

## OCR Working State

- App + backend run locally on `127.0.0.1:8000` and `127.0.0.1:8001`.
- Neural ROI is mandatory in the frontend OCR flow (heuristic ROI fallback removed).
- On neural ROI failure, the UI shows an explicit reason and asks for manual measurement input.
- Backend default ROI model is pinned to `backend/models/roi-rotaug-e30-640.pt` (override with `ROI_MODEL_PATH`).
- `train_roi.py` enforces augmentation policy by default: heavy online augmentation + rotation expansion `90,180,270,360`.
- Digit-classifier inference is mandatory in the frontend OCR flow (`OCR_CONFIG.digitClassifier.enabled` defaults to `true`).
- Backend serves ROI + digit endpoints and reports readiness via `GET /health`.
- Test-set table includes `Detected`, `Absolute Error`, `Failure Reason`, and `Result`.
- Frontend OCR branch evaluation is strip-only classifier-first candidate decoding (no Tesseract word-pass/sparse-scan stages).
- Neural-ROI OCR now prefers `90/270` edge candidates first; same-angle base candidates are fallback-only when edge candidates fail.
- Opposite-orientation retry is disabled by default (`roiDeterministic.tryOppositeOrientation=false`).
- Final edge acceptance still uses confidence thresholds, but the selector no longer requires non-edge corroboration by default.
- Edge-derived candidate generation is toggleable via `roiDeterministic.useEdgeCandidates` (default `true`) for controlled A/B experiments.
- Debug overlay semantics:
  - `6a. OCR input candidate (initial preview)` = first valid ROI candidate before classifier ranking.
  - `6. OCR input candidate` = winning decode input (exact strip variant/angle used by final selection).
- Current local benchmark set has `17` images.
- Historical checkpoint comparison (March 2, 2026, legacy fallback `OFF`, 14-image snapshot):
  - `roi-rotaug-e30-640.pt` (default pinned): exact-match `0/14`, failure mix `ocr-no-digits` (7), `mismatch` (6), `no-detection` (1).
  - `roi.pt` (challenger): exact-match `0/14`, failure mix `ocr-no-digits` (10), `mismatch` (4), `no-detection` (0).
- Recent force-mode A/B (March 4, 2026, 15-image set): forcing the initial preview candidate reduced MAE but sharply increased no-read due to `classifier-edge-gate-final-drop`; keep force mode disabled by default.
- Automated diff workflow is available via `npm run benchmark:roi-diff`.
- Diff artifacts now use `*-neural-digit` output folders because digit decoding is always neural-classifier-only.
- Promotion and rollback decisions should now use `MAE` from `roi-diff-report` as the primary signal, with exact-match and no-read as guardrails.
- ROI diff reports now include per-image selected metadata columns (`sourceLabel`, `method`, `preprocessMode`) and explicitly export the last stage `6. OCR input candidate` snapshot (winning decode strip variant).

## Next TODOs

1. Keep `roi-rotaug-e30-640.pt` as default until a challenger beats it on end-to-end OCR metrics, not only detection presence.
2. Re-run `npm run benchmark:roi-diff` after each ROI challenger to track per-image movement (`Detected`, stage `5/6` snapshots, reject reason), then summarize deltas in notes/PR.
3. Tune strip preprocessing and candidate ranking on the current highest-absolute-error rows from each new test-set run.
4. Improve classifier-first candidate ranking and acceptance thresholds to reduce `mismatch` while preserving low no-read.
5. Enforce checkpoint promotion gates from docs: no MAE regression, no exact-match regression, no no-read regression, and no regression in dominant no-read buckets (`classifier-edge-gate-final-drop` / `ocr-no-digits`).
6. Keep running both `npm run test:e2e` and UI `Run test set` before commits; include histogram deltas in commit/PR notes.
7. Medium-term: evaluate YOLO OBB ROI detection to reduce rotation/edge ambiguity; this requires OBB relabeling, retraining, and backend response/schema changes before frontend adoption.

### OBB Notes (Re-verify Before Implementation)

- OBB inference outputs rotated geometry (`xywhr`) and polygon corners.
- OBB training labels use corners format: `class x1 y1 x2 y2 x3 y3 x4 y4`.
- OBB angle handling has constraints (Ultralytics OBB uses angles in the `0-90` exclusive range).

## Dataset Expansion Loop (`4/5/6/9`)

1. Refresh capture planning:
   - `cd backend && source .venv/bin/activate && python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`
2. Add labeled captures with QA previews.
3. Validate manifests after each dataset update:
   - Confirm `data/digit_dataset/manifests/{windows.csv,canonical_windows.csv,sections.csv,section_labels.csv}` are regenerated and consistent with current splits.
   - Use `python validate_digit_dataset.py` only for legacy strip/cell exports (`build_digit_dataset.py` path).
4. Retrain classifier only after class coverage improves:
   - `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu`
5. Keep classifier training/dataset refresh loop active; promote new checkpoints only when benchmarked `MAE` improves without exact-match/no-read guardrail regressions.
