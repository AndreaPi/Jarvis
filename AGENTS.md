# Repository Guidelines

## Project Structure & Module Organization
- `index.html`: Single-page UI layout and content.
- `styles.css`: Global styles and visual system.
- `app.js`: Client-side logic (OCR flow, email draft generation).
- `package.json`: Local dev scripts.
- Assets are currently referenced directly from uploads; no dedicated assets folder yet.

## Build, Test, and Development Commands
- `npm run serve`: Start a simple local web server on port 8000.
- `npm run dev`: Alias of `npm run serve`.

Open `http://localhost:8000` after running a serve command.

## Coding Style & Naming Conventions
- Use 2-space indentation in HTML/CSS/JS.
- Keep files ASCII-only unless there is a strong reason for Unicode.
- Use descriptive, lower-case IDs and class names (e.g., `photo-input`, `module-grid`).
- Prefer clear, small functions in `app.js` and avoid deep nesting.

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
