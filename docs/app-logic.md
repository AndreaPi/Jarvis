# App Logic

This document describes the current Jarvis OCR execution path, including neural ROI gating and the optional gated classifier fallback.

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
  H --> I["Build ROI candidates<br/>(rotations + edge crops)"]
  I --> J{"Candidates available?"}
  J -- "No" --> K["Ask manual entry"]
  J -- "Yes" --> L["Word-pass OCR per candidate<br/>(SINGLE_WORD, digits only)"]

  L --> M{"Best 4-digit reading found?"}
  M -- "No" --> N["Sparse scan OCR on ROI crop<br/>(SPARSE_TEXT, soft)"]
  M -- "Yes" --> O["finalizeSelection()<br/>evidence ranking + word-pass support guardrail"]
  N --> N2{"Still no accepted reading?"}
  N2 -- "No" --> O
  N2 -- "Yes" --> N3{"Classifier fallback enabled<br/>and no-digits rejects seen?"}
  N3 -- "No" --> O
  N3 -- "Yes" --> N4["Digit-classifier fallback<br/>(4 cells from ROI candidate)"]
  N4 --> O

  O --> P{"Final selection exists?"}
  P -- "Yes" --> Q["Return reading + fill UI input"]
  P -- "No" --> R["Return null + ask manual entry"]

  O --> S["Push selection log<br/>window.__jarvisOcrSelectionLogs"]
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
   - Word-pass result is preferred.
   - Sparse scan is attempted if no word-pass result is available.
   - Optional classifier fallback runs only when enabled and the branch has `ocr-no-digits` rejects.
   - `finalizeSelection` ranks evidence across OCR passes and applies the active word-pass support guardrail (`hits` / `topHits` vs `minWordPassHits`) before returning a value.
   - Default config keeps classifier fallback disabled (`digitClassifier.enabled=false`) because current benchmark shows no MAE gain without exact-match/no-read guardrail safety.

## What Gets Logged

- Per-image selection logs are appended to `window.__jarvisOcrSelectionLogs`.
- The test-set runner reads those logs to build:
  - `Failure Reason` values (`mismatch`, `ocr-no-digits`, etc.)
  - Reject histograms from OCR branch reject reasons.

## Source Files

- OCR orchestration: `src/ocr/pipeline.js`
- Neural ROI probe and sanity checks: `src/ocr/neural-roi.js`
- Candidate generation: `src/ocr/alignment.js`
- OCR ranking/reading extraction: `src/ocr/recognition.js`
- Test-set analysis and histograms: `src/testset/run-test-set.js`
