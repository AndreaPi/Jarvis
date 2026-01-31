# Jarvis

Jarvis is a lightweight personal assistant web app. The first module helps you read a water meter photo, review the detected value, and draft an email in Gmail.

## Features
- Upload a meter photo and preview it.
- OCR the reading (manual override supported).
- Auto-fill an email draft with the current date in Italian format.
- Open a Gmail draft or use a mailto fallback.

## Local Development
1. Install dependencies (none required beyond Python).
2. Run the dev server:

```bash
npm run serve
```

Then open `http://localhost:8000`.

## File Overview
- `index.html`: UI layout.
- `styles.css`: Styling.
- `app.js`: OCR + email draft logic.
- `AGENTS.md`: Contributor guide.
- `assets/`: Static assets and example uploads.

## Notes
- OCR runs fully in the browser using Tesseract.js.
- The Gmail flow opens a draft; you always review and send manually.
