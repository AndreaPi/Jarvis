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

const DEFAULT_SUBJECT = 'Lettura acqua da F9C397';
const CUSTOMER_DETAILS = {
  name: 'Andrea Panizza',
  address: 'Via delle cinque giornate 15 piano 1 e 1/2',
  city: 'Firenze FI',
  code: 'F9C397'
};

let currentPhotoFile = null;
let bodyTouched = false;
let subjectTouched = false;

const formatItalianDate = (dateValue) => {
  if (!dateValue) {
    return '';
  }
  const [year, month, day] = dateValue.split('-').map(Number);
  if (!year || !month || !day) {
    return '';
  }
  const localDate = new Date(year, month - 1, day);
  return new Intl.DateTimeFormat('it-IT').format(localDate);
};

const buildBodyTemplate = (dateDisplay, readingValue) => {
  const safeReading = readingValue || '____';
  return [
    `Intestatario: ${CUSTOMER_DETAILS.name}`,
    CUSTOMER_DETAILS.address,
    CUSTOMER_DETAILS.city,
    `Codice Utente: ${CUSTOMER_DETAILS.code}`,
    `Data: ${dateDisplay}`,
    `Lettura: ${safeReading}`
  ].join('\n');
};

const replaceLine = (body, label, value) => {
  if (!body.trim()) {
    return '';
  }
  const lines = body.split('\n');
  const index = lines.findIndex((line) => line.trim().startsWith(label));
  const nextLine = `${label} ${value}`;
  if (index >= 0) {
    lines[index] = nextLine;
  } else {
    lines.push(nextLine);
  }
  return lines.join('\n');
};

const extractReading = (text) => {
  const matches = text.match(/\d{4,7}/g);
  if (!matches || matches.length === 0) {
    return '';
  }
  const longestLength = Math.max(...matches.map((value) => value.length));
  const candidates = matches.filter((value) => value.length === longestLength);
  return candidates[candidates.length - 1];
};

const setStatus = (message) => {
  ocrStatus.textContent = message;
};

const buildEmailDraft = () => {
  return {
    to: toInput.value.trim(),
    subject: subjectInput.value.trim() || DEFAULT_SUBJECT,
    body: bodyInput.value.trim()
  };
};

const updateMailLinks = () => {
  const { to, subject, body } = buildEmailDraft();
  const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=${encodeURIComponent(to)}&su=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  sendBtn.dataset.gmailUrl = gmailUrl;
  mailtoLink.href = `mailto:${encodeURIComponent(to)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
};

const updateSubject = ({ force = false } = {}) => {
  if (force || !subjectTouched) {
    subjectInput.value = DEFAULT_SUBJECT;
    subjectTouched = false;
  }
  updateMailLinks();
};

const updateBody = ({ force = false } = {}) => {
  const dateDisplay = formatItalianDate(dateInput.value);
  const readingValue = readingInput.value.trim();

  if (force || !bodyTouched) {
    bodyInput.value = buildBodyTemplate(dateDisplay, readingValue);
    bodyTouched = false;
  } else {
    let updated = bodyInput.value;
    updated = replaceLine(updated, 'Data:', dateDisplay);
    updated = replaceLine(updated, 'Lettura:', readingValue || '____');
    bodyInput.value = updated || buildBodyTemplate(dateDisplay, readingValue);
  }

  updateMailLinks();
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
    const { data } = await Tesseract.recognize(currentPhotoFile, 'eng', {
      tessedit_char_whitelist: '0123456789',
      logger: (message) => {
        if (message.status === 'recognizing text') {
          const progress = Math.round(message.progress * 100);
          setStatus(`Reading image ${progress}%`);
        } else if (message.status) {
          setStatus(message.status);
        }
      }
    });

    const detected = extractReading(data.text || '');
    if (detected) {
      readingInput.value = detected;
      setStatus(`Reading detected: ${detected}. Review if needed.`);
      updateBody();
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
  updateBody();
});

readingInput.addEventListener('blur', () => {
  updateBody();
});

dateInput.addEventListener('change', () => {
  updateBody();
});

toInput.addEventListener('input', () => {
  updateMailLinks();
});

subjectInput.addEventListener('input', () => {
  subjectTouched = true;
  updateMailLinks();
});

bodyInput.addEventListener('input', () => {
  bodyTouched = true;
  updateMailLinks();
});

regenBtn.addEventListener('click', () => {
  bodyTouched = false;
  subjectTouched = false;
  updateSubject({ force: true });
  updateBody({ force: true });
});

sendBtn.addEventListener('click', () => {
  if (!sendBtn.dataset.gmailUrl) {
    updateMailLinks();
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

  updateSubject({ force: true });
  updateBody({ force: true });
  updateMailLinks();
};

init();
