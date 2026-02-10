# Repository Guidelines

## Project Structure & Module Organization
- `index.html`: Single-page UI layout and content.
- `styles.css`: Global styles and visual system.
- `app.js`: Thin module entrypoint that imports `src/main.js`.
- `src/main.js`: UI orchestration and event wiring.
- `src/ocr/`: OCR pipeline and image processing modules.
- `src/email/`: Email draft generation and link helpers.
- `src/testset/`: Manual test-set runner logic.
- `src/debug/`: Debug overlay rendering helpers.
- `package.json`: Local dev scripts.
- `README.md`: Project overview and setup notes.
- `assets/`: Static assets and example uploads.
- `assets/meter_13012026.jpg`: Example upload asset.

## Build, Test, and Development Commands
- `npm run serve`: Start a simple local web server on port 8000.
- `npm run dev`: Alias of `npm run serve`.

Open `http://localhost:8000` after running a serve command.

## Coding Style & Naming Conventions
- Use 2-space indentation in HTML/CSS/JS.
- Keep files ASCII-only unless there is a strong reason for Unicode.
- Use descriptive, lower-case IDs and class names (e.g., `photo-input`, `module-grid`).
- Prefer clear, small functions in `src/` modules and avoid deep nesting.

## Testing Guidelines
- No automated tests are configured.
- Manual checks: upload image, run OCR, verify email draft fields, and confirm Gmail draft link.

## Commit & Pull Request Guidelines
- No commit message convention is established in this repo.
- Suggested pattern: short, imperative subject (e.g., "Improve OCR preview").
- PRs should include: summary of changes, screenshots for UI changes, and any manual test notes.

## Security & Configuration Tips
- The Gmail draft flow opens a client-side draft; no credentials are stored in code.
- OCR runs in the browser; avoid adding API keys to the client without a secure proxy.

## OCR Handoff Notes (2026-02-09)

### Current status
- App runs on `http://127.0.0.1:8000` (`npm run serve`).
- Test set currently ends at `0/8` accuracy.
- Latest observed table:
  - `meter_07012020.JPEG`: expected `1784`, detected `4000`, score `0.80`
  - `meter_11112020.JPEG`: expected `1819`, detected `3084`, score `0.97`
  - `meter_10092025.JPEG`: expected `2279`, detected `1040`, score `0.80`
  - `meter_10092025_b.JPEG`: expected `2279`, detected `0071`, score `0.80`
  - `meter_01122026.JPEG`: expected `2302`, detected `2000`, score `0.93`
  - `meter_01132026.jpg`: expected `2302`, detected `5407`, score `0.98`
  - `meter_01302026.JPEG`: expected `2307`, detected `0440`, score `0.81`
  - `meter_02022026.JPEG`: expected `2308`, detected `0130`, score `0.80`

### What worked
- Added visual debug pipeline to UI test section:
  - Stage 1: face detection overlay
  - Stage 2: aligned frame + ROI boxes
  - Stage 3: detected strip crop
  - Stage 4: OCR input candidate
- Debug capture is available directly from "Run test set" output, so each image can be inspected.
- Face/circle detection is often roughly centered on the meter body.

### What did not work
- Strip ROI detection is still wrong on most images.
- Stage 2 boxes frequently lock onto lid/background/rim instead of the numeric window.
- Stage 3 crops are often non-digit regions (lid edge, blank white region, red dials).
- Rotation choice is coupled with noisy strip scoring, so bad orientation/ROI combinations are selected.
- OCR confidence score is over-optimistic in some paths, masking ROI failures.

### Evidence artifacts
- Debug screenshots exported to: `output/playwright/debug-roi/`
- Final test snapshot with table + debug panels: `.playwright-cli/page-2026-02-09T22-52-09-298Z.yml`

### Next plan (priority order)
1. Remove score floor inflation in `readDigitsByCells` (`Math.max(score, 0.8/0.82)`), so bad ROI cannot look "good".
2. Make alignment two-pass:
   - Pass A: detect face + canonical rotation only.
   - Pass B: search strip only inside a strict meter-window search band (relative to face center/radius).
3. Add hard strip acceptance gates before OCR:
   - low red ratio (to avoid dial clusters),
   - expected aspect range,
   - expected distance from face center,
   - minimum digit-like vertical stroke periodicity.
4. If strip confidence is low, skip strip ROI and use deterministic fallback ROI tied to canonical meter pose.
5. Add two debug frames:
   - strip score heatmap / top-k strip boxes with scores,
   - binary/edge map used for strip decision.
6. Re-run full test set and verify first target:
   - detection exists for all 8 images,
   - then optimize per-image score threshold (`>= 0.8`) with truthful scoring (no artificial floor).

### Practical notes for tomorrow
- Prefer running the test set from UI with debug overlay enabled.
- When using Playwright in this environment, global `playwright-cli` may be more reliable than the wrapper if npm network is flaky.
- Maintainability: if `src/ocr/canvas-utils.js` keeps growing, split it into focused modules (for example `image-ops.js` and `region-analysis.js`) before adding new OCR features.
