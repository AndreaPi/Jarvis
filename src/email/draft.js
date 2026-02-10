const DEFAULT_SUBJECT = 'Lettura acqua da F9C397';
const DEFAULT_CUSTOMER_DETAILS = {
  name: 'Andrea Panizza',
  address: 'Via delle cinque giornate 15 piano 1 e 1/2',
  city: 'Firenze FI',
  code: 'F9C397'
};

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

const buildBodyTemplate = (customerDetails, dateDisplay, readingValue) => {
  const safeReading = readingValue || '____';
  return [
    `Intestatario: ${customerDetails.name}`,
    customerDetails.address,
    customerDetails.city,
    `Codice Utente: ${customerDetails.code}`,
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

const createEmailDraftController = ({
  subjectInput,
  bodyInput,
  toInput,
  mailtoLink,
  sendBtn,
  readingInput,
  dateInput,
  defaultSubject = DEFAULT_SUBJECT,
  customerDetails = DEFAULT_CUSTOMER_DETAILS
}) => {
  let bodyTouched = false;
  let subjectTouched = false;

  const buildEmailDraft = () => {
    return {
      to: toInput.value.trim(),
      subject: subjectInput.value.trim() || defaultSubject,
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
      subjectInput.value = defaultSubject;
      subjectTouched = false;
    }
    updateMailLinks();
  };

  const updateBody = ({ force = false } = {}) => {
    const dateDisplay = formatItalianDate(dateInput.value);
    const readingValue = readingInput.value.trim();

    if (force || !bodyTouched) {
      bodyInput.value = buildBodyTemplate(customerDetails, dateDisplay, readingValue);
      bodyTouched = false;
    } else {
      let updated = bodyInput.value;
      updated = replaceLine(updated, 'Data:', dateDisplay);
      updated = replaceLine(updated, 'Lettura:', readingValue || '____');
      bodyInput.value = updated || buildBodyTemplate(customerDetails, dateDisplay, readingValue);
    }

    updateMailLinks();
  };

  const markSubjectTouched = () => {
    subjectTouched = true;
  };

  const markBodyTouched = () => {
    bodyTouched = true;
  };

  const resetTouched = () => {
    subjectTouched = false;
    bodyTouched = false;
  };

  return {
    updateMailLinks,
    updateSubject,
    updateBody,
    markSubjectTouched,
    markBodyTouched,
    resetTouched
  };
};

export {
  DEFAULT_SUBJECT,
  DEFAULT_CUSTOMER_DETAILS,
  createEmailDraftController
};
