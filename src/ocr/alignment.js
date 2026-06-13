import { OCR_CONFIG } from './config.js';
import {
  rotateCanvas,
  scaleCanvas,
  cropCanvas,
  findDigitWindowByEdges,
  preprocessCanvas
} from './canvas-utils.js';

const buildNeuralRoiCandidates = (source, debugSession, addDebugStageFn = () => {}) => {
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const normalizeWidth = Number.isFinite(roiDeterministic.normalizeWidth)
    ? roiDeterministic.normalizeWidth
    : OCR_CONFIG.minScaleWidth;
  const useEdgeCandidates = roiDeterministic.useEdgeCandidates !== false;
  const debugWordMode = Array.isArray(roiDeterministic.wordPassModes) && roiDeterministic.wordPassModes.length
    ? roiDeterministic.wordPassModes.find((mode) => mode === 'soft' || mode === 'binary' || mode === 'raw') || 'raw'
    : 'raw';

  const configuredAngles = Array.isArray(roiDeterministic.primaryAngles) && roiDeterministic.primaryAngles.length
    ? roiDeterministic.primaryAngles
    : [90, 270];
  const angles = configuredAngles
    .map((angle) => Number.parseInt(angle, 10))
    .filter((angle, index, values) => Number.isFinite(angle) && values.indexOf(angle) === index);
  const candidates = [];
  let debugStripSource = null;
  const baseCandidates = [];

  const roundNumber = (value, digits = 3) => (
    Number.isFinite(value) ? Number(value.toFixed(digits)) : null
  );

  const serializeRect = (rect) => {
    if (!rect) {
      return null;
    }
    return {
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    };
  };

  const buildGeometryMetadata = ({ angle, family, rotated, cropRect, edgeRect = null, extra = {} }) => {
    const safeCropRect = serializeRect(cropRect);
    const rotatedWidth = rotated && Number.isFinite(rotated.width) ? rotated.width : null;
    const rotatedHeight = rotated && Number.isFinite(rotated.height) ? rotated.height : null;
    const cropAspect = safeCropRect
      ? roundNumber(safeCropRect.width / Math.max(1, safeCropRect.height))
      : null;
    const cropAreaRatio = safeCropRect && rotatedWidth && rotatedHeight
      ? roundNumber((safeCropRect.width * safeCropRect.height) / Math.max(1, rotatedWidth * rotatedHeight))
      : null;
    return {
      angle,
      family,
      rotatedSize: {
        width: rotatedWidth,
        height: rotatedHeight
      },
      cropRect: safeCropRect,
      edgeRect: serializeRect(edgeRect),
      cropAspect,
      cropAreaRatio,
      cropFrame: safeCropRect && rotatedWidth && rotatedHeight ? {
        left: roundNumber(safeCropRect.x / Math.max(1, rotatedWidth)),
        right: roundNumber((safeCropRect.x + safeCropRect.width) / Math.max(1, rotatedWidth)),
        top: roundNumber(safeCropRect.y / Math.max(1, rotatedHeight)),
        bottom: roundNumber((safeCropRect.y + safeCropRect.height) / Math.max(1, rotatedHeight))
      } : null,
      ...extra
    };
  };

  const edgeContextPaddingX = Number.isFinite(roiDeterministic.edgeContextPaddingX)
    ? Math.max(0, Math.min(0.5, roiDeterministic.edgeContextPaddingX))
    : 0.18;
  const edgeContextPaddingY = Number.isFinite(roiDeterministic.edgeContextPaddingY)
    ? Math.max(0, Math.min(0.4, roiDeterministic.edgeContextPaddingY))
    : 0.08;
  const edgeContextShiftRatios = Array.isArray(roiDeterministic.edgeContextShiftRatios)
    ? roiDeterministic.edgeContextShiftRatios
      .filter((ratio) => Number.isFinite(ratio))
      .map((ratio) => Math.max(-0.35, Math.min(0.35, ratio)))
    : [-0.08, 0, 0.08];
  const edgeContextMaxVariantsPerAngle = Number.isFinite(roiDeterministic.edgeContextMaxVariantsPerAngle)
    ? Math.max(0, Math.min(12, Math.round(roiDeterministic.edgeContextMaxVariantsPerAngle)))
    : 3;
  const normalizationProbe = roiDeterministic.normalizationProbe || {};
  const normalizationProbeEnabled = normalizationProbe.enabled !== false;
  const normalizationProbeShadowOnly = normalizationProbe.shadowOnly !== false;
  const normalizationProbeTargetAspects = Array.isArray(normalizationProbe.targetAspects)
    ? normalizationProbe.targetAspects
      .filter((ratio) => Number.isFinite(ratio))
      .map((ratio) => Math.max(1.4, Math.min(6, ratio)))
    : [2.4, 2.8, 3.2];
  const normalizationProbeHeightRatios = Array.isArray(normalizationProbe.heightRatios)
    ? normalizationProbe.heightRatios
      .filter((ratio) => Number.isFinite(ratio))
      .map((ratio) => Math.max(0.8, Math.min(2.2, ratio)))
    : [1, 1.16];
  const normalizationProbeShiftRatios = Array.isArray(normalizationProbe.shiftRatios)
    ? normalizationProbe.shiftRatios
      .filter((ratio) => Number.isFinite(ratio))
      .map((ratio) => Math.max(-0.4, Math.min(0.4, ratio)))
    : [-0.12, 0, 0.12];
  const normalizationProbeMaxVariantsPerAngle = Number.isFinite(normalizationProbe.maxVariantsPerAngle)
    ? Math.max(0, Math.min(48, Math.round(normalizationProbe.maxVariantsPerAngle)))
    : 12;

  const pushCandidate = (canvas, label, metadata = {}) => {
    if (!canvas) {
      return;
    }
    const normalized = scaleCanvas(canvas, normalizeWidth);
    if (!normalized || normalized.width < 24 || normalized.height < 16) {
      return;
    }
    candidates.push({ canvas: normalized, label, ...metadata });
    if (!debugStripSource) {
      debugStripSource = canvas;
    }
  };

  const normalizeCropRect = (canvas, rect) => {
    if (!canvas || !rect) {
      return null;
    }
    const x = Math.max(0, Math.min(canvas.width - 1, Math.round(rect.x)));
    const y = Math.max(0, Math.min(canvas.height - 1, Math.round(rect.y)));
    const width = Math.max(1, Math.min(canvas.width - x, Math.round(rect.width)));
    const height = Math.max(1, Math.min(canvas.height - y, Math.round(rect.height)));
    return { x, y, width, height };
  };

  const pushEdgeContextCandidates = (rotated, edgeRect, angle) => {
    if (!rotated || !edgeRect || edgeContextMaxVariantsPerAngle <= 0) {
      return;
    }
    const paddedWidth = Math.min(
      rotated.width,
      edgeRect.width * (1 + edgeContextPaddingX * 2)
    );
    const paddedHeight = Math.min(
      rotated.height,
      edgeRect.height * (1 + edgeContextPaddingY * 2)
    );
    if (paddedWidth <= edgeRect.width + 1 && paddedHeight <= edgeRect.height + 1) {
      return;
    }

    const centerX = edgeRect.x + edgeRect.width * 0.5;
    const centerY = edgeRect.y + edgeRect.height * 0.5;
    const seen = new Set();
    let emitted = 0;
    edgeContextShiftRatios.forEach((shiftRatio) => {
      if (emitted >= edgeContextMaxVariantsPerAngle) {
        return;
      }
      const rect = normalizeCropRect(rotated, {
        x: centerX - paddedWidth * 0.5 + edgeRect.width * shiftRatio,
        y: centerY - paddedHeight * 0.5,
        width: paddedWidth,
        height: paddedHeight
      });
      if (!rect) {
        return;
      }
      const key = `${Math.round(rect.x / 3)}:${Math.round(rect.y / 3)}:${Math.round(rect.width / 3)}:${Math.round(rect.height / 3)}`;
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      emitted += 1;
      const shiftPercent = Math.round(Math.abs(shiftRatio) * 100);
      const suffix = shiftRatio === 0
        ? 'center'
        : (shiftRatio > 0 ? `right${shiftPercent}` : `left${shiftPercent}`);
      pushCandidate(
        cropCanvas(rotated, rect),
        `roi-${angle}-edge-context-${suffix}`,
        {
          geometry: buildGeometryMetadata({
            angle,
            family: 'edge-context',
            rotated,
            cropRect: rect,
            edgeRect,
            extra: { shiftRatio: Number(shiftRatio.toFixed(3)) }
          })
        }
      );
    });
  };

  const pushNormalizationProbeCandidates = (rotated, edgeRect, angle) => {
    if (
      !normalizationProbeEnabled
      || !rotated
      || !edgeRect
      || normalizationProbeMaxVariantsPerAngle <= 0
      || !normalizationProbeTargetAspects.length
      || !normalizationProbeHeightRatios.length
      || !normalizationProbeShiftRatios.length
    ) {
      return;
    }

    const centerX = edgeRect.x + edgeRect.width * 0.5;
    const centerY = edgeRect.y + edgeRect.height * 0.5;
    const seen = new Set();
    let emitted = 0;

    normalizationProbeTargetAspects.forEach((targetAspect) => {
      normalizationProbeHeightRatios.forEach((heightRatio) => {
        normalizationProbeShiftRatios.forEach((shiftRatio) => {
          if (emitted >= normalizationProbeMaxVariantsPerAngle) {
            return;
          }
          const targetHeight = Math.min(rotated.height, Math.max(edgeRect.height, edgeRect.height * heightRatio));
          const targetWidth = Math.min(
            rotated.width,
            Math.max(edgeRect.width * 1.04, targetHeight * targetAspect)
          );
          const rect = normalizeCropRect(rotated, {
            x: centerX - targetWidth * 0.5 + edgeRect.width * shiftRatio,
            y: centerY - targetHeight * 0.5,
            width: targetWidth,
            height: targetHeight
          });
          if (!rect || rect.width < 8 || rect.height < 8) {
            return;
          }
          const key = `${Math.round(rect.x / 3)}:${Math.round(rect.y / 3)}:${Math.round(rect.width / 3)}:${Math.round(rect.height / 3)}`;
          if (seen.has(key)) {
            return;
          }
          seen.add(key);
          emitted += 1;
          const aspectToken = String(Math.round(targetAspect * 10)).padStart(2, '0');
          const heightToken = String(Math.round(heightRatio * 100)).padStart(3, '0');
          const shiftPercent = Math.round(Math.abs(shiftRatio) * 100);
          const suffix = shiftRatio === 0
            ? 'center'
            : (shiftRatio > 0 ? `right${shiftPercent}` : `left${shiftPercent}`);
          pushCandidate(
            cropCanvas(rotated, rect),
            `roi-${angle}-normprobe-a${aspectToken}-h${heightToken}-${suffix}`,
            {
              diagnosticOnly: normalizationProbeShadowOnly,
              probeKind: 'normalization',
              geometry: buildGeometryMetadata({
                angle,
                family: 'normprobe',
                rotated,
                cropRect: rect,
                edgeRect,
                extra: {
                  targetAspect: roundNumber(targetAspect),
                  heightRatio: roundNumber(heightRatio),
                  shiftRatio: roundNumber(shiftRatio)
                }
              })
            }
          );
        });
      });
    });
  };

  angles.forEach((angle) => {
    const rotated = angle === 0 ? source : rotateCanvas(source, angle);

    if (useEdgeCandidates) {
      const edgeRect = findDigitWindowByEdges(rotated);
      if (edgeRect) {
        const edgeCrop = cropCanvas(rotated, edgeRect);
        pushCandidate(edgeCrop, `roi-${angle}-edge`, {
          geometry: buildGeometryMetadata({
            angle,
            family: 'edge',
            rotated,
            cropRect: edgeRect,
            edgeRect
          })
        });
        pushEdgeContextCandidates(rotated, edgeRect, angle);
        pushNormalizationProbeCandidates(rotated, edgeRect, angle);
      }
    }

    const normalized = scaleCanvas(rotated, normalizeWidth);
    if (normalized && normalized.width >= 24 && normalized.height >= 16) {
      baseCandidates.push({
        canvas: normalized,
        label: `roi-${angle}-base`,
        geometry: buildGeometryMetadata({
          angle,
          family: 'base',
          rotated,
          cropRect: {
            x: 0,
            y: 0,
            width: rotated.width,
            height: rotated.height
          }
        })
      });
      if (!debugStripSource) {
        debugStripSource = rotated;
      }
    }
  });

  if (baseCandidates.length) {
    candidates.push(...baseCandidates);
  }

  if (!candidates.length) {
    const fallback = scaleCanvas(source, normalizeWidth);
    candidates.push({
      canvas: fallback,
      label: 'roi-base-fallback',
      geometry: buildGeometryMetadata({
        angle: 0,
        family: 'base',
        rotated: source,
        cropRect: {
          x: 0,
          y: 0,
          width: source.width,
          height: source.height
        }
      })
    });
    debugStripSource = source;
  }

  if (debugSession) {
    const stripPreview = preprocessCanvas(debugStripSource || source, 'soft');
    addDebugStageFn(debugSession, '5. detected strip crop', stripPreview);
    const ocrPreview = debugWordMode === 'raw'
      ? candidates[0].canvas
      : preprocessCanvas(candidates[0].canvas, debugWordMode);
    addDebugStageFn(debugSession, '6a. OCR input candidate (initial preview)', ocrPreview);
  }

  return candidates;
};

const buildDigitCandidates = (source, debugSession = null, addDebugStageFn = () => {}) => {
  return buildNeuralRoiCandidates(source, debugSession, addDebugStageFn);
};

export { buildDigitCandidates };
