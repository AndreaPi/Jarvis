# App Logic

This document describes the current Jarvis OCR execution path, with neural ROI gating and neural digit-classifier-only decoding.

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
  J -- "Yes" --> L["Classifier-first OCR per candidate<br/>(4 cells, orientation-aware)"]
  L --> O["finalizeSelection()<br/>evidence ranking + edge safeguard"]

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
   - Candidate strips are decoded directly with the backend digit classifier.
   - `finalizeSelection` ranks evidence across classifier passes before returning a value.
   - Edge-only winners are rejected unless corroborated by non-edge evidence or very strong per-cell confidence.
   - Digit classifier is enabled by default (`digitClassifier.enabled=true`).

## What Gets Logged

- Per-image selection logs are appended to `window.__jarvisOcrSelectionLogs`.
- `selected` metadata includes `sourceLabel`, `method`, and `preprocessMode` for each accepted reading.
- The test-set runner reads those logs to build:
  - `Failure Reason` values (`mismatch`, `ocr-no-digits`, etc.)
  - Reject histograms from OCR branch reject reasons.

## Source Files

- OCR orchestration: `src/ocr/pipeline.js`
- Neural ROI probe and sanity checks: `src/ocr/neural-roi.js`
- Candidate generation: `src/ocr/alignment.js`
- OCR ranking/reading extraction: `src/ocr/recognition.js`
- Test-set analysis and histograms: `src/testset/run-test-set.js`
