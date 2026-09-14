// Render a hand-edited .excalidraw scene back to a themed SVG.
//
// The generators in this directory go one way: JavaScript to scene to SVG.
// This goes the other way, for the case where you opened a scene in Excalidraw,
// moved something, and want the SVG to match.
//
// Use Excalidraw's own "Export to SVG" instead and you lose the thing that
// makes these files work in dark mode: it bakes literal hex into every shape,
// where this renderer emits var(--ce-ink) and friends.
//
//   node render-scene.js ../../../posts/01-why-context-engineering/diagrams/excalidraw/01-timeline.excalidraw
//   node render-scene.js scene.excalidraw out.svg --width 960 --height 360
//
// With no output path it writes alongside the input, replacing the extension.
// Canvas size, title and desc are taken from the SVG already sitting there, so
// a re-render after an edit keeps the accessible description you wrote. If
// there is no such file, the canvas is measured from the drawing and the two
// texts must be supplied with --title and --desc.

const fs = require('fs');
const path = require('path');
const { renderSvg } = require('./lib');

const argv = process.argv.slice(2);
const flag = (name) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? null : argv[i + 1];
};
const positional = argv.filter((a, i) => !a.startsWith('--') && !(i && argv[i - 1].startsWith('--')));

const src = positional[0];
if (!src) {
  console.error('usage: node render-scene.js <scene.excalidraw> [out.svg] [--width N] [--height N] [--title T] [--desc D]');
  process.exit(2);
}
const out = positional[1] || src.replace(/\.excalidraw$/, '') + '.svg';

const scene = JSON.parse(fs.readFileSync(src, 'utf8'));
const elements = (scene.elements || []).filter((el) => !el.isDeleted);
if (!elements.length) {
  console.error(`${src}: no elements`);
  process.exit(1);
}

// Whatever is already at the output path is the best source for the three
// things a scene file does not carry: canvas size, title, and description.
const prior = fs.existsSync(out) ? fs.readFileSync(out, 'utf8') : '';
const grab = (re) => { const m = prior.match(re); return m ? m[1] : null; };
const box = grab(/viewBox="0 0 ([\d.]+ [\d.]+)"/);

// Excalidraw text elements report a width the editor computed with real font
// metrics, so measuring the drawing is honest even for the text.
const measured = () => {
  const pad = 24;
  const x2 = Math.max(...elements.map((e) => e.x + (e.width || 0)));
  const y2 = Math.max(...elements.map((e) => e.y + (e.height || 0)));
  return [Math.ceil(x2 + pad), Math.ceil(y2 + pad)];
};

const [mw, mh] = measured();
const width = Number(flag('width') ?? (box ? box.split(' ')[0] : mw));
const height = Number(flag('height') ?? (box ? box.split(' ')[1] : mh));

const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>')
  .replace(/&quot;/g, '"').replace(/&amp;/g, '&');

const title = flag('title') ?? (grab(/<title id="t">([\s\S]*?)<\/title>/) || '');
const desc = flag('desc') ?? (grab(/<desc id="d">([\s\S]*?)<\/desc>/) || '');

if (!title || !desc) {
  console.error(`${out}: no <title> or <desc> to inherit. Every diagram in this `
    + `repository needs both; pass --title and --desc.`);
  process.exit(1);
}

const svg = renderSvg(elements, {
  width, height, title: unescape(title), desc: unescape(desc),
});
fs.writeFileSync(out, svg);

const same = prior && prior === svg;
console.log(`${path.basename(out)}: ${elements.length} elements, ${width}x${height}`
  + (same ? ' (unchanged)' : prior ? ' (updated)' : ''));

// A scene edited in Excalidraw keeps whatever colours the editor had. If a new
// shape was drawn with a colour off Excalidraw's own palette it will not map to
// a --ce-* token, and it will not follow the theme. Say so rather than shipping
// a diagram that goes invisible in dark mode.
const KNOWN = new Set(Object.values(require('./lib').T));
const strays = new Set();
for (const el of elements) {
  for (const c of [el.strokeColor, el.backgroundColor]) {
    if (c && c !== 'transparent' && !KNOWN.has(c)) strays.add(c);
  }
}
if (strays.size) {
  console.warn(`  warning: ${strays.size} colour(s) outside the --ce-* palette, `
    + `baked in as literals: ${[...strays].join(', ')}`);
}
