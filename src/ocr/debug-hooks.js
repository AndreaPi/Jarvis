let debugHooks = null;

const setOcrDebugHooks = (hooks) => {
  debugHooks = hooks || null;
};

const startDebugSession = (label) => {
  if (!debugHooks || typeof debugHooks.startSession !== 'function') {
    return null;
  }
  return debugHooks.startSession(label);
};

const addDebugStage = (session, name, canvas) => {
  if (!debugHooks || typeof debugHooks.addStage !== 'function') {
    return;
  }
  debugHooks.addStage(session, name, canvas);
};

const commitDebugSession = (session) => {
  if (!debugHooks || typeof debugHooks.commitSession !== 'function') {
    return;
  }
  debugHooks.commitSession(session);
};

export { setOcrDebugHooks, startDebugSession, addDebugStage, commitDebugSession };
