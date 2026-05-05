const OCR_CONFIG = {
  maxDimension: 4096,
  meterCropScale: 0.95,
  digitCrops: [
    { name: 'top-wide', x: 0.04, y: 0.02, width: 0.92, height: 0.32 },
    { name: 'top-left', x: 0.0, y: 0.05, width: 0.7, height: 0.4 },
    { name: 'left-tall', x: 0.0, y: 0.12, width: 0.55, height: 0.58 },
    { name: 'upper-band', x: 0.1, y: 0.08, width: 0.8, height: 0.28 }
  ],
  preferredDigits: 4,
  digitCellCount: 4,
  digitCellOverlap: 0.08,
  minDigitWidth: 96,
  minDigits: 3,
  earlyStopScore: 0.84,
  fallbackScoreThreshold: 0.72,
  fallbackCandidates: 6,
  minScaleWidth: 480,
  geometry: {
    minCandidateWidth: 120,
    minCandidateHeight: 28,
    minCandidateAspect: 0.12,
    maxCandidateAspect: 18,
    minStripAspect: 1.15,
    maxStripAspect: 12,
    minCellWidth: 20,
    minCellHeight: 24
  },
  roiDeterministic: {
    tightenInk: 0.08,
    cellOverlap: 0.03,
    requireAllCells: true,
    tryOppositeOrientation: false,
    primaryAngles: [90, 270],
    wordPassModes: ['raw'],
    minWordPassHits: 1,
    minStripAspect: 1.45,
    maxStripAspect: 8.2,
    normalizeWidth: 520,
    deskewMaxAngle: 8,
    deskewStep: 2,
    useEdgeCandidates: true
  },
  digitClassifier: {
    enabled: true,
    endpoint: 'http://127.0.0.1:8001/digit/predict-cells',
    timeoutMs: 1800,
    minCellConfidence: 0.18,
    forceInitialPreviewCandidate: false,
    fallbackPreferNonEdge: false,
    fallbackTargetAspect: 2.6,
    maxPrimaryCandidates: 4,
    fallbackEdgeMinAverageConfidence: 25,
    fallbackEdgeMinCellConfidence: 18,
    fallbackEdgeRequireNonEdgeSupport: false,
    disableAfterFailures: 2,
    cooldownMs: 8000
  },
  digitStripReader: {
    enabled: true,
    endpoint: 'http://127.0.0.1:8001/digit/predict-strip',
    timeoutMs: 1800,
    minConfidence: 0,
    shadowOnly: true,
    disableAfterFailures: 2,
    cooldownMs: 8000
  },
  digitStripReader23xx: {
    enabled: true,
    endpoint: 'http://127.0.0.1:8001/digit/predict-strip-23xx',
    timeoutMs: 1800,
    guardThreshold: 0.98,
    shadowOnly: true,
    disableAfterFailures: 2,
    cooldownMs: 8000
  },
  neuralRoi: {
    enabled: true,
    endpoint: 'http://127.0.0.1:8001/roi/detect',
    timeoutMs: 8000,
    minConfidence: 0.01,
    expandX: 0.26,
    expandY: 0.16,
    sanity: {
      minCenterX: 0.28,
      maxCenterX: 0.62,
      minCenterY: 0.28,
      maxCenterY: 0.68,
      minArea: 0.003,
      maxArea: 0.03,
      minAspect: 0.35,
      maxAspect: 3.2
    }
  }
};

const applyRuntimeOverrides = () => {
  if (typeof window === 'undefined') {
    return;
  }
  const overrides = window.__JARVIS_OCR_CONFIG_OVERRIDE__;
  if (!overrides || typeof overrides !== 'object') {
    return;
  }
  if (overrides.digitClassifier && typeof overrides.digitClassifier === 'object') {
    OCR_CONFIG.digitClassifier = {
      ...OCR_CONFIG.digitClassifier,
      ...overrides.digitClassifier
    };
  }
  if (overrides.digitStripReader && typeof overrides.digitStripReader === 'object') {
    OCR_CONFIG.digitStripReader = {
      ...OCR_CONFIG.digitStripReader,
      ...overrides.digitStripReader
    };
  }
  if (overrides.digitStripReader23xx && typeof overrides.digitStripReader23xx === 'object') {
    OCR_CONFIG.digitStripReader23xx = {
      ...OCR_CONFIG.digitStripReader23xx,
      ...overrides.digitStripReader23xx
    };
  }
};

applyRuntimeOverrides();

const ALIGNMENT_CONFIG = {
  centerOffsetLimit: 0.14,
  centerOffsetStep: 0.05,
  radiusMinRatio: 0.25,
  radiusMaxRatio: 0.42,
  radiusStepRatio: 0.02,
  facePadding: 1.04,
  stripSearchBand: {
    offsetXFromCenterRadius: -0.36,
    offsetYFromCenterRadius: -0.44,
    widthFromRadius: 1.26,
    heightFromRadius: 0.82
  },
  stripGates: {
    maxRedRatio: 0.12,
    minAspect: 1.35,
    maxAspect: 4.4,
    minDistanceFromCenterRadius: 0.32,
    maxDistanceFromCenterRadius: 0.92,
    expectedDistanceFromCenterRadius: 0.6,
    distanceTolerance: 0.4,
    minVerticalPeriodicity: 0.2,
    minAcceptedScore: 2.45
  },
  stripDebugTopK: 6,
  stripWindows: [
    { name: 'strip-main', scaleX: 1.0, scaleY: 1.0, shiftX: 0, shiftY: 0 },
    { name: 'strip-tight', scaleX: 0.86, scaleY: 0.84, shiftX: 0.02, shiftY: 0.02 }
  ],
  fallbackWindows: [
    { name: 'fallback-main', x: 0.08, y: 0.14, width: 0.48, height: 0.24 },
    { name: 'fallback-tight', x: 0.1, y: 0.17, width: 0.44, height: 0.2 }
  ]
};

const DEBUG_CONFIG = {
  maxSessions: 14,
  previewWidth: 260,
  colors: ['#ef4444', '#2563eb', '#16a34a', '#f59e0b', '#ec4899']
};

export { OCR_CONFIG, ALIGNMENT_CONFIG, DEBUG_CONFIG };
