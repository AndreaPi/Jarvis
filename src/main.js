import { runMeterOcr, setOcrDebugHooks } from './ocr/pipeline.js';
import { createDebugOverlayManager } from './debug/overlay.js';
import { createTestSetRunner } from './testset/run-test-set.js';
import {
  DEFAULT_SUBJECT,
  DEFAULT_CUSTOMER_DETAILS,
  createEmailDraftController
} from './email/draft.js';

const photoInput = document.getElementById('photo-input');
const photoPreview = document.getElementById('photo-preview');
const readBtn = document.getElementById('read-btn');
const ocrStatus = document.getElementById('ocr-status');
const readingInput = document.getElementById('reading-input');
const dateInput = document.getElementById('date-input');
const fromInput = document.getElementById('from-input');
const toInput = document.getElementById('to-input');
const subjectInput = document.getElementById('subject-input');
const bodyInput = document.getElementById('body-input');
const regenBtn = document.getElementById('regen-btn');
const sendBtn = document.getElementById('send-btn');
const mailtoLink = document.getElementById('mailto-link');
const runTestBtn = document.getElementById('run-test-btn');
const testStatus = document.getElementById('test-status');
const testResults = document.getElementById('test-results');
const debugOverlayToggle = document.getElementById('debug-overlay-toggle');
const clearDebugBtn = document.getElementById('clear-debug-btn');
const debugResults = document.getElementById('debug-results');

const DEBUG_CONFIG = {
  maxSessions: 14,
  previewWidth: 260
};

let currentPhotoFile = null;

const debugOverlayManager = createDebugOverlayManager({
  toggleEl: debugOverlayToggle,
  resultsEl: debugResults,
  maxSessions: DEBUG_CONFIG.maxSessions,
  previewWidth: DEBUG_CONFIG.previewWidth
});

setOcrDebugHooks(debugOverlayManager.ocrHooks);

const emailDraftController = createEmailDraftController({
  subjectInput,
  bodyInput,
  toInput,
  mailtoLink,
  sendBtn,
  readingInput,
  dateInput,
  defaultSubject: DEFAULT_SUBJECT,
  customerDetails: DEFAULT_CUSTOMER_DETAILS
});

const testSetRunner = createTestSetRunner({
  runButton: runTestBtn,
  statusEl: testStatus,
  resultsEl: testResults,
  runMeterOcr
});

const setStatus = (message) => {
  ocrStatus.textContent = message;
};

photoInput.addEventListener('click', () => {
  photoInput.value = '';
});

photoInput.addEventListener('change', () => {
  const file = photoInput.files && photoInput.files[0];
  if (!file) {
    currentPhotoFile = null;
    photoPreview.innerHTML = '<p class="muted">No photo loaded yet.</p>';
    setStatus('Waiting for a photo.');
    return;
  }

  currentPhotoFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    photoPreview.innerHTML = '';
    const img = document.createElement('img');
    img.src = reader.result;
    img.alt = 'Water meter preview';
    photoPreview.appendChild(img);
  };
  reader.onerror = () => {
    photoPreview.innerHTML = '<p class="muted">Preview unavailable.</p>';
  };
  reader.readAsDataURL(file);
  setStatus('Photo ready. Click "Read meter".');
});

readBtn.addEventListener('click', async () => {
  if (!currentPhotoFile) {
    setStatus('Upload a photo first.');
    return;
  }

  if (!window.Tesseract) {
    setStatus('OCR library not available. Check your connection and try again.');
    return;
  }

  readBtn.disabled = true;
  setStatus('Reading image...');

  try {
    const result = await runMeterOcr(currentPhotoFile, (message) => setStatus(message));
    if (result && result.value) {
      readingInput.value = result.value;
      setStatus(`Reading detected: ${result.value}. Review if needed.`);
      emailDraftController.updateBody();
    } else {
      setStatus('No clear reading detected. Enter it manually.');
    }
  } catch (error) {
    console.error(error);
    setStatus('OCR failed. Enter the reading manually.');
  } finally {
    readBtn.disabled = false;
  }
});

readingInput.addEventListener('input', () => {
  const sanitized = readingInput.value.replace(/\D/g, '');
  if (sanitized !== readingInput.value) {
    readingInput.value = sanitized;
  }
  emailDraftController.updateBody();
});

readingInput.addEventListener('blur', () => {
  emailDraftController.updateBody();
});

dateInput.addEventListener('change', () => {
  emailDraftController.updateBody();
});

toInput.addEventListener('input', () => {
  emailDraftController.updateMailLinks();
});

subjectInput.addEventListener('input', () => {
  emailDraftController.markSubjectTouched();
  emailDraftController.updateMailLinks();
});

bodyInput.addEventListener('input', () => {
  emailDraftController.markBodyTouched();
  emailDraftController.updateMailLinks();
});

if (runTestBtn) {
  runTestBtn.addEventListener('click', () => {
    testSetRunner.runTestSet();
  });
}

if (clearDebugBtn) {
  clearDebugBtn.addEventListener('click', () => {
    debugOverlayManager.clear();
  });
}

if (debugOverlayToggle) {
  debugOverlayToggle.addEventListener('change', () => {
    if (!debugOverlayToggle.checked) {
      debugOverlayManager.clear();
      return;
    }
    debugOverlayManager.render();
  });
}

regenBtn.addEventListener('click', () => {
  emailDraftController.resetTouched();
  emailDraftController.updateSubject({ force: true });
  emailDraftController.updateBody({ force: true });
});

sendBtn.addEventListener('click', () => {
  if (!sendBtn.dataset.gmailUrl) {
    emailDraftController.updateMailLinks();
  }
  window.open(sendBtn.dataset.gmailUrl, '_blank', 'noopener');
});

const init = () => {
  const now = new Date();
  const isoDate = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0')
  ].join('-');

  dateInput.value = isoDate;
  fromInput.value = fromInput.value || 'andrea.panizza75@gmail.com';

  emailDraftController.updateSubject({ force: true });
  emailDraftController.updateBody({ force: true });
  emailDraftController.updateMailLinks();
  debugOverlayManager.render();
};

init();
