// Excalidraw scene builder + roughjs SVG renderer for the NNFS series.
//
// Two outputs from one scene description:
//   1. a .excalidraw file that opens and edits at excalidraw.com
//   2. a hand-drawn .svg rendered with roughjs, themed with the repo's
//      --ce-* design tokens so it works in light and dark mode
//
// Determinism matters: every element carries an explicit seed derived from its
// index, so regenerating a scene produces a byte-identical file. A random seed
// would make every rebuild a diff.

const rough = require('roughjs');
const GEN = (rough.default ? rough.default : rough).generator();

// --- palette (light-mode token values; the SVG swaps them via CSS) ----------
// These are the same --ce-* values the clean vector figures in posts/*/diagrams
// already inline, so a hand-drawn alternate and the figure it accompanies
// resolve to identical colours in both themes.
const T = {
  bg: '#FAFAF7', surface: '#FFFFFF', ink: '#1A1A1A', inkMuted: '#5C5C5C',
  inkSubtle: '#9A9A9A', border: '#D9D9D4', primary: '#5B7FBF', accent: '#D98E5F',
  success: '#5C9E78', warn: '#B8895A', alert: '#C66B5E',
  neutral1: '#EAEAE4', neutral2: '#CFCFC8', neutral3: '#8E8E88', onFill: '#FFFDF9',
  // Text that sits on a saturated fill. --ce-on-accent is deliberately dark ink
  // in BOTH themes, unlike --ce-on-fill which flips: light text on the success
  // and alert fills measures 3.12:1 and 3.65:1 in light mode, and dark ink
  // measures 5.48 and 4.70. The literal is one bit off T.ink on purpose,
  // because VAR below is keyed by hex and a duplicate key would silently
  // rewrite every T.ink use into var(--ce-on-accent).
  onAccent: '#1A1A1B',
};

// Map a literal hex back to a CSS variable so the rendered SVG is theme-aware.
const VAR = Object.fromEntries(Object.entries({
  bg: 'bg', surface: 'surface', ink: 'ink', inkMuted: 'ink-muted', inkSubtle: 'ink-subtle',
  border: 'border', primary: 'primary', accent: 'accent', success: 'success',
  warn: 'warn', alert: 'alert', neutral1: 'neutral-1', neutral2: 'neutral-2',
  neutral3: 'neutral-3', onFill: 'on-fill', onAccent: 'on-accent',
}).map(([k, v]) => [T[k], `var(--ce-${v})`]));

const themed = (hex) => (hex && VAR[hex]) ? VAR[hex] : (hex || 'none');

let SEQ = 0;
const resetSeq = () => { SEQ = 0; };

const base = (type, x, y, w, h, o = {}) => ({
  id: `el-${++SEQ}`, type, x, y, width: w, height: h, angle: 0,
  strokeColor: o.stroke ?? T.ink,
  backgroundColor: o.fill ?? 'transparent',
  fillStyle: o.fillStyle ?? 'solid',
  strokeWidth: o.strokeWidth ?? 1,
  strokeStyle: o.strokeStyle ?? 'solid',
  roughness: o.roughness ?? 1,
  opacity: o.opacity ?? 100,
  groupIds: o.groupIds ?? [], frameId: null,
  roundness: o.roundness === null ? null : (o.roundness ?? { type: 3 }),
  seed: 1000 + SEQ * 7919, version: 1, versionNonce: 1000 + SEQ * 104729,
  isDeleted: false, boundElements: o.boundElements ?? null, updated: 1,
  link: null, locked: false,
});

// --- element factories ------------------------------------------------------
const rect = (x, y, w, h, o = {}) => base('rectangle', x, y, w, h, o);
const ellipse = (x, y, w, h, o = {}) => base('ellipse', x, y, w, h, { ...o, roundness: null });
const diamond = (x, y, w, h, o = {}) => base('diamond', x, y, w, h, { ...o, roundness: null });

// A circle by centre and radius, which is how every numbered badge is placed.
const circle = (cx, cy, r, o = {}) => ellipse(cx - r, cy - r, r * 2, r * 2, o);

const text = (x, y, str, o = {}) => {
  const size = o.size ?? 16;
  const family = o.family ?? 5;               // 5 = Excalifont (hand-drawn)
  const lines = String(str).split('\n');
  const w = o.width ?? Math.max(...lines.map((l) => l.length)) * size * 0.55;
  const h = lines.length * size * 1.25;
  return {
    ...base('text', x, y, w, h, { ...o, roundness: null, strokeWidth: 1 }),
    text: String(str), originalText: String(str),
    fontSize: size, fontFamily: family,
    textAlign: o.align ?? 'left', verticalAlign: o.valign ?? 'top',
    containerId: o.containerId ?? null, lineHeight: 1.25, autoResize: true,
  };
};

const line = (pts, o = {}) => {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  return {
    ...base(o.arrow ? 'arrow' : 'line', x, y, Math.max(...xs) - x, Math.max(...ys) - y, o),
    points: pts.map((p) => [p[0] - x, p[1] - y]),
    lastCommittedPoint: null, startBinding: null, endBinding: null,
    startArrowhead: o.startArrowhead ?? null,
    endArrowhead: o.arrow ? (o.endArrowhead ?? 'arrow') : null,
    elbowed: false,
  };
};
const arrow = (pts, o = {}) => line(pts, { ...o, arrow: true });

// --- roughjs -> SVG path ----------------------------------------------------
const opsToPath = (ops) => {
  let d = '';
  for (const op of ops) {
    const p = op.data.map((n) => Number(n.toFixed(2)));
    if (op.op === 'move') d += `M${p[0]} ${p[1]}`;
    else if (op.op === 'lineTo') d += `L${p[0]} ${p[1]}`;
    else if (op.op === 'bcurveTo') d += `C${p[0]} ${p[1]} ${p[2]} ${p[3]} ${p[4]} ${p[5]}`;
  }
  return d;
};

const drawableToSvg = (drawable, { stroke, fill, strokeWidth, dash }) => {
  let out = '';
  for (const set of drawable.sets) {
    const d = opsToPath(set.ops);
    if (!d) continue;
    if (set.type === 'path') {
      const da = dash ? ` stroke-dasharray="${dash}"` : '';
      out += `<path d="${d}" fill="none" stroke="${themed(stroke)}" stroke-width="${strokeWidth}" stroke-linecap="round" stroke-linejoin="round"${da}/>`;
    } else if (set.type === 'fillPath') {
      out += `<path d="${d}" fill="${themed(fill)}" stroke="none"/>`;
    } else if (set.type === 'fillSketch') {
      out += `<path d="${d}" fill="none" stroke="${themed(fill)}" stroke-width="${Math.max(1, strokeWidth * 0.8)}"/>`;
    }
  }
  return out;
};

const ESC = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const FONT = {
  5: `"Excalifont","Virgil","Segoe Print","Comic Sans MS",ui-rounded,cursive`,
  1: `"Excalifont","Virgil","Segoe Print","Comic Sans MS",ui-rounded,cursive`,
  2: `Inter,ui-sans-serif,system-ui,sans-serif`,
  3: `"JetBrains Mono",ui-monospace,Menlo,monospace`,
};

// The token block every rendered SVG inlines. It matches the block the clean
// vector figures carry, so the two forms stay in step when a token moves.
const TOKENS_LIGHT = '--ce-bg:#FAFAF7;--ce-surface:#FFFFFF;--ce-ink:#1A1A1A;--ce-ink-muted:#5C5C5C;'
  + '--ce-ink-subtle:#9A9A9A;--ce-border:#D9D9D4;--ce-primary:#5B7FBF;--ce-accent:#D98E5F;'
  + '--ce-success:#5C9E78;--ce-warn:#B8895A;--ce-alert:#C66B5E;--ce-neutral-1:#EAEAE4;'
  + '--ce-neutral-2:#CFCFC8;--ce-neutral-3:#8E8E88;--ce-on-fill:#FFFDF9;--ce-on-accent:#1A1A1A;';

const TOKENS_DARK = '--ce-bg:#0E0F12;--ce-surface:#16181C;--ce-ink:#F2F2EE;--ce-ink-muted:#B4B4AE;'
  + '--ce-ink-subtle:#6E6E68;--ce-border:#2A2D33;--ce-primary:#8BA8E0;--ce-accent:#E8B088;'
  + '--ce-success:#7FBF9B;--ce-warn:#D4B58A;--ce-alert:#D88880;--ce-neutral-1:#1F2229;'
  + '--ce-neutral-2:#2C3038;--ce-neutral-3:#6E6E68;--ce-on-fill:#14161A;--ce-on-accent:#1A1A1A;';

function renderSvg(elements, { width, height, title, desc }) {
  // Fill and stroke are generated as two passes with different roughness. A
  // filled roughjs shape traces its fill boundary independently of its outline,
  // so at equal roughness the colour visibly overshoots the ink and the result
  // reads as sloppy rather than hand-drawn. Softening only the fill keeps the
  // wobble in the line, where it belongs, and holds the colour inside it.
  const roughOpts = (el, pass) => ({
    roughness: pass === 'fill' ? Math.min(el.roughness, 0.5) : el.roughness,
    seed: el.seed, bowing: pass === 'fill' ? 0.4 : 1.1,
    stroke: el.strokeColor,
    fill: el.backgroundColor === 'transparent' ? undefined : el.backgroundColor,
    fillStyle: el.fillStyle === 'solid' ? 'solid' : el.fillStyle,
    strokeWidth: el.strokeWidth,
    hachureGap: 6, fillWeight: 1.4,
    strokeLineDash: el.strokeStyle === 'dashed' ? [9, 7] : (el.strokeStyle === 'dotted' ? [2, 5] : undefined),
  });

  const shape = (el, o) => {
    if (el.type === 'rectangle') {
      return el.roundness
        ? GEN.path(roundedRectPath(el.x, el.y, el.width, el.height, Math.min(24, el.width / 6, el.height / 6)), o)
        : GEN.rectangle(el.x, el.y, el.width, el.height, o);
    }
    if (el.type === 'ellipse') return GEN.ellipse(el.x + el.width / 2, el.y + el.height / 2, el.width, el.height, o);
    const cx = el.x + el.width / 2, cy = el.y + el.height / 2;
    return GEN.polygon([[cx, el.y], [el.x + el.width, cy], [cx, el.y + el.height], [el.x, cy]], o);
  };

  const onlyFill = (svg) => svg.replace(/<path d="[^"]*" fill="none"[^>]*\/>/g, '');
  const onlyStroke = (svg) => svg.replace(/<path d="[^"]*" fill="(?!none)[^"]*"[^>]*\/>/g, '');

  let body = '';
  for (const el of elements) {
    if (el.isDeleted) continue;
    const sw = el.strokeWidth;
    // Excalidraw carries opacity per element, 0 to 100. Emit it as a wrapping
    // group so a tint band can sit under text without swallowing it.
    const before = body.length;
    if (el.type === 'rectangle' || el.type === 'ellipse' || el.type === 'diamond') {
      const args = { stroke: el.strokeColor, fill: el.backgroundColor, strokeWidth: sw, dash: dashFor(el) };
      if (el.backgroundColor && el.backgroundColor !== 'transparent') {
        body += onlyFill(drawableToSvg(shape(el, roughOpts(el, 'fill')), args));
      }
      body += onlyStroke(drawableToSvg(shape(el, roughOpts(el, 'stroke')), args));
    } else if (el.type === 'line' || el.type === 'arrow') {
      const o = roughOpts(el, 'stroke');
      const pts = el.points.map((p) => [el.x + p[0], el.y + p[1]]);
      body += drawableToSvg(GEN.linearPath(pts, o), { stroke: el.strokeColor, fill: 'none', strokeWidth: sw, dash: dashFor(el) });
      if (el.type === 'arrow' && el.endArrowhead) body += head(pts.at(-2), pts.at(-1), el.strokeColor, sw);
      if (el.type === 'arrow' && el.startArrowhead) body += head(pts[1], pts[0], el.strokeColor, sw);
    } else if (el.type === 'text') {
      const size = el.fontSize;
      const anchor = el.textAlign === 'center' ? 'middle' : el.textAlign === 'right' ? 'end' : 'start';
      const ax = el.textAlign === 'center' ? el.x + el.width / 2 : el.textAlign === 'right' ? el.x + el.width : el.x;
      el.text.split('\n').forEach((ln, i) => {
        body += `<text xml:space="preserve" x="${r2(ax)}" y="${r2(el.y + size * (0.92 + i * 1.25))}" text-anchor="${anchor}" `
          + `font-family='${FONT[el.fontFamily] || FONT[5]}' font-size="${size}" fill="${themed(el.strokeColor)}">${ESC(ln)}</text>`;
      });
    }
    if (el.opacity !== 100 && body.length > before) {
      body = body.slice(0, before)
        + `<g opacity="${r2(el.opacity / 100)}">${body.slice(before)}</g>`;
    }
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="t d">
  <title id="t">${ESC(title)}</title>
  <desc id="d">${ESC(desc)}</desc>

  <style><![CDATA[
    :root{${TOKENS_LIGHT}}
    @media (prefers-color-scheme:dark){:root{${TOKENS_DARK}}}
    .bg{fill:var(--ce-bg);}
    text{dominant-baseline:auto;}
  ]]></style>

  <rect class="bg" width="${width}" height="${height}"/>
${body}
</svg>
`;
}

const r2 = (n) => Number(n.toFixed(2));

const dashFor = (el) => el.strokeStyle === 'dashed' ? '9 7'
  : el.strokeStyle === 'dotted' ? '1.5 5' : null;

function roundedRectPath(x, y, w, h, r) {
  return `M${x + r} ${y} L${x + w - r} ${y} Q${x + w} ${y} ${x + w} ${y + r}`
    + ` L${x + w} ${y + h - r} Q${x + w} ${y + h} ${x + w - r} ${y + h}`
    + ` L${x + r} ${y + h} Q${x} ${y + h} ${x} ${y + h - r}`
    + ` L${x} ${y + r} Q${x} ${y} ${x + r} ${y} Z`;
}

function head(from, to, stroke, sw) {
  const a = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const L = 11, S = 0.42;
  const p = (sign) => [r2(to[0] - L * Math.cos(a + sign * S)), r2(to[1] - L * Math.sin(a + sign * S))];
  const [x1, y1] = p(1), [x2, y2] = p(-1);
  return `<path d="M${x1} ${y1} L${r2(to[0])} ${r2(to[1])} L${x2} ${y2}" fill="none" stroke="${themed(stroke)}" `
    + `stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round"/>`;
}

const scene = (elements) => ({
  type: 'excalidraw', version: 2, source: 'https://excalidraw.com',
  elements, appState: { gridSize: null, viewBackgroundColor: T.bg }, files: {},
});

module.exports = { T, rect, ellipse, circle, diamond, text, line, arrow, renderSvg, scene, resetSeq };
