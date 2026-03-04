const scaleCanvasForPreview = (source, targetWidth) => {
  if (source.width <= targetWidth) {
    return source;
  }
  const scale = targetWidth / source.width;
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(source.width * scale);
  canvas.height = Math.round(source.height * scale);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
};

const toDebugDataUrl = (source, previewWidth) => {
  const preview = scaleCanvasForPreview(source, previewWidth);
  return preview.toDataURL('image/jpeg', 0.9);
};

const renderDebugSessions = (resultsEl, sessions) => {
  if (!resultsEl) {
    return;
  }

  resultsEl.innerHTML = '';
  if (!sessions.length) {
    const empty = document.createElement('p');
    empty.className = 'muted';
    empty.textContent = 'No debug frames yet.';
    resultsEl.appendChild(empty);
    return;
  }

  sessions.forEach((session) => {
    const wrapper = document.createElement('article');
    wrapper.className = 'debug-session';

    const title = document.createElement('p');
    title.className = 'debug-session-title';
    title.textContent = session.label;
    wrapper.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'debug-stage-grid';
    session.stages.forEach((stage) => {
      const card = document.createElement('article');
      card.className = 'debug-stage';

      const img = document.createElement('img');
      img.src = stage.dataUrl;
      img.alt = `${session.label} ${stage.name}`;
      card.appendChild(img);

      const caption = document.createElement('p');
      caption.className = 'debug-stage-name';
      caption.textContent = stage.name;
      card.appendChild(caption);

      grid.appendChild(card);
    });

    wrapper.appendChild(grid);
    resultsEl.appendChild(wrapper);
  });
};

const createDebugOverlayManager = ({
  toggleEl,
  resultsEl,
  maxSessions = 14,
  previewWidth = 260
}) => {
  const sessions = [];

  const render = () => {
    renderDebugSessions(resultsEl, sessions);
  };

  const clear = () => {
    sessions.length = 0;
    render();
  };

  const startSession = (label) => {
    if (!toggleEl || !toggleEl.checked) {
      return null;
    }
    return { label, stages: [] };
  };

  const addStage = (session, name, canvas) => {
    if (!session || !canvas) {
      return;
    }
    session.stages.push({
      name,
      dataUrl: toDebugDataUrl(canvas, previewWidth)
    });
  };

  const commitSession = (session) => {
    if (!session || !session.stages.length) {
      return;
    }
    sessions.unshift(session);
    if (sessions.length > maxSessions) {
      sessions.length = maxSessions;
    }
    render();
  };

  return {
    ocrHooks: {
      startSession,
      addStage,
      commitSession
    },
    clear,
    render
  };
};

export { createDebugOverlayManager };
