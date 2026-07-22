# App Logic

This document describes the current Jarvis OCR execution path, with neural ROI gating, per-cell neural digit-classifier selection, whole-strip shadow-reader logging, and guarded `23xx` shadow-reader diagnostics.

## End-to-End OCR Flow

```mermaid
flowchart TD
  A["UI: Read meter"] --> B["runMeterOcr(file)"]
  B --> C["Load image + start debug session"]
  C --> D["detectNeuralRoi(file, config)"]
  D --> E{"ROI probe ok?"}

  E -- "No" --> F["ROI fail reason<br/>disabled | http-error | invalid-json | no-detection | invalid-bbox | invalid-confidence | low-confidence | invalid-geometry (center/area/aspect)"]
  F --> G["Show miss in debug + ask manual entry"]

  E -- "Yes" --> H["Expand + crop ROI"]
  H --> I["Build ROI candidates<br/>(rotations + optional edge crops)"]
  I --> J{"Candidates available?"}
  J -- "No" --> K["Ask manual entry"]
  J -- "Yes" --> L["Per candidate:<br/>run strip reader shadow + classifier cells"]
  L --> M["Cell classifier result remains primary<br/>(strip readers are shadow-only)"]
  M --> O["finalizeSelection()<br/>evidence ranking + edge safeguard"]

  O --> P{"Final selection exists?"}
  P -- "Yes" --> Q["Return reading + fill UI input"]
  P -- "No" --> R["Return null + ask manual entry"]

  O --> S["Push selection log<br/>window.__jarvisOcrSelectionLogs<br/>(includes selected source/method/mode)"]
  S --> T["Run test set => failure/reject histograms"]
```

![App Logic Flow Render](./app-logic-flow.png)

## Main Decision Gates

1. Neural ROI gate
   - OCR does not continue unless `detectNeuralRoi` returns `ok: true`.
   - Geometry sanity checks (`centerX`, `centerY`, `area`, `aspect`) are applied before accepting ROI.

2. Candidate availability gate
   - If ROI crop cannot produce valid OCR candidates, the app falls back to manual input.

3. OCR acceptance gate
   - Candidate strips are decoded by splitting into four cells and calling the backend digit classifier.
   - The whole-strip reader and guarded `23xx` reader also run for candidates in shadow mode and record diagnostics without affecting selection.
   - `finalizeSelection` ranks evidence across classifier passes before returning a value.
   - The selector prioritizes `90/270` edge candidates, but the primary pass also evaluates top base-strip candidates when they are available.
   - A narrow `scan-roi` / base fallback rerun is only used when base candidates were not already evaluated and the top edge evidence is still weak or edge-only.
   - Weak edge-only reads can still be rejected by configured per-cell confidence thresholds in `finalizeSelection`.
   - Digit classifier is enabled by default (`digitClassifier.enabled=true`).
   - Strip reader shadow logging is enabled by default (`digitStripReader.enabled=true`, `digitStripReader.shadowOnly=true`).
   - Guarded `23xx` shadow logging is enabled by default (`digitStripReader23xx.enabled=true`, `digitStripReader23xx.shadowOnly=true`).

## What Gets Logged

- Per-image selection logs are appended to `window.__jarvisOcrSelectionLogs`, retaining the latest 300 entries.
- `selected` metadata includes `sourceLabel`, `method`, and `preprocessMode` for each accepted reading.
- `stripReader` metadata contains the best shadow whole-strip prediction, confidence, source label, and per-position confidence summary.
- `stripReader23xx` metadata contains accepted/abstained guarded-prefix diagnostics and suffix confidences.
- The test-set runner reads those logs to build:
  - `Failure Reason` values (`mismatch`, `classifier-edge-gate-final-drop`, `ocr-no-digits`, etc.)
  - Reject histograms from OCR branch reject reasons.

## Photo Replacement Safety

- Selecting a new photo immediately clears the previous detected reading and regenerates the email draft without that stale value.
- Each photo change invalidates any in-flight OCR request. Progress, errors, and final readings from an older photo are ignored if they arrive after the replacement.
- Preview callbacks use the same photo revision guard, so a delayed file read cannot replace the current preview.

## Source Files

- OCR orchestration: `src/ocr/pipeline.js`
- Photo selection and stale-request guards: `src/main.js`
- Neural ROI probe and sanity checks: `src/ocr/neural-roi.js`
- Candidate generation: `src/ocr/alignment.js`
- OCR ranking/reading extraction: `src/ocr/recognition.js`
- Test-set analysis and histograms: `src/testset/run-test-set.js`
