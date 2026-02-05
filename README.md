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
- The upload date defaults to the photo EXIF capture date (DateTimeOriginal) when available; otherwise it uses today.

## Backend (Monotonic Readings)
To keep readings consistent across devices, Jarvis uses a small Supabase-backed API (`api/reading.js`) that stores the latest reading and enforces monotonic increases.

### Supabase Setup
1. Create a Supabase project (EU region recommended).
2. Create the table below:

```sql
create table if not exists public.meter_readings (
  meter_code text primary key,
  reading integer not null,
  reading_date date not null,
  updated_at timestamptz not null default now()
);
```

### Environment Variables (Vercel)
Set these in your Vercel project:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

The service role key is only used server-side in the API function. It should never be exposed to the browser.

### Local Development
`npm run serve` serves static files only; it does not run serverless functions. To test the API locally, use `vercel dev` or deploy to Vercel.

## Asset Naming (Meter Images)
- Use the EXIF `DateTimeOriginal` value as the source of truth for the acquisition date.
- Rename files to `meter_mmddyyyy` (zero-padded) and keep the original extension.
- If multiple images share the same date, keep one as-is and add suffixes to the rest (e.g., `_b`, `_c`).
- If EXIF is missing, prefer a known date from the filename or capture notes and document it.
