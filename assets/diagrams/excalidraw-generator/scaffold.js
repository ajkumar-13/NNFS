// Shared scaffolding for every postNN.js generator.
//
// Four things every scene in the series agrees on: where the left margin sits,
// how a title and its one-line subtitle are placed, how a card with a coloured
// spine is drawn, and how a built scene turns into the two files on disk.
// Keeping them here is what stops thirty-five generators from drifting apart
// one heading offset at a time.
//
// It also carries the fit checker. Text overrunning the box it was meant to sit
// in is by far the most common fault in these figures, and it is invisible in
// the SVG source — it only shows up when the thing is rendered. `fit()` records
// an estimate at build time and `emit()` reports it, so a build says "this line
// needs 264px and has 240" instead of a reviewer noticing in a PNG.

const fs = require('fs');
const path = require('path');
const { T, rect, text, line, renderSvg, scene } = require('./lib');

// Left margin shared with the clean vector figures, so a hand-drawn alternate
// lines up with the one it accompanies when a reader flips between them.
const M = 40;

// Title at 22, subtitle at 13, baselines at y=38 and y=62. Pass x when a figure
// uses a different measure.
const heading = (title, sub, x = M) => [
  text(x, 18, title, { size: 22, stroke: T.ink }),
  text(x, 50, sub, { size: 13, stroke: T.inkMuted }),
];

// --- fit checking -----------------------------------------------------------
// Advance width per character, calibrated against rendered output rather than
// font metrics: Excalifont and its fallbacks sit near 0.58em for mixed-case
// text at the sizes this series uses. Deliberately a little pessimistic, so a
// warning means "check this" rather than "this is definitely broken".
const ADVANCE = 0.58;

const estWidth = (s, size) =>
  Math.max(...String(s).split('\n').map((l) => l.length)) * size * ADVANCE;

let PENDING = [];

// Record `s` as needing to fit in maxWidth at this size. Returns s unchanged so
// it can wrap a string in place: text(x, y, fit(label, 11, 240, 'row label'), …)
function fit(s, size, maxWidth, where) {
  const need = estWidth(s, size);
  if (need > maxWidth) {
    PENDING.push({ where, s: String(s).split('\n')[0], need: Math.ceil(need), have: Math.round(maxWidth) });
  }
  return s;
}

function flushFit(name) {
  if (!PENDING.length) return 0;
  for (const o of PENDING) {
    console.warn(`  ! ${name}: ${o.where} needs ~${o.need}px, has ${o.have}px  — "${o.s}"`);
  }
  const n = PENDING.length;
  PENDING = [];
  return n;
}

// Greedy word wrap to a pixel width, returning a '\n'-joined string ready for
// text(). Splitting a sentence at a fixed word count instead is what produced
// most of the overruns this checker caught: word lengths vary and the count does
// not know that. `max` caps the line count so a wrap cannot silently grow a card.
function wrap(s, size, maxWidth, max = 99) {
  const out = [];
  let line = '';
  for (const word of String(s).split(/\s+/)) {
    const next = line ? line + ' ' + word : word;
    if (line && estWidth(next, size) > maxWidth) { out.push(line); line = word; }
    else line = next;
  }
  if (line) out.push(line);
  // A silent truncation is worse than an overflow: an overflow is visible in the
  // render, a dropped tail reads as a finished sentence that simply stops.
  if (out.length > max) {
    PENDING.push({
      where: `wrap truncated to ${max} of ${out.length} lines`,
      s: out.slice(max).join(' ').slice(0, 60),
      need: out.length, have: max,
    });
  }
  return out.slice(0, max).join('\n');
}

// --- common composites ------------------------------------------------------
// A surface card with a coloured spine down its left edge. The spine is a
// stroke rather than a thin filled rect because roughjs fills a sliver that
// narrow as scratches rather than as a bar.
function card(x, y, w, h, o = {}) {
  const els = [rect(x, y, w, h, {
    stroke: o.stroke ?? T.ink,
    fill: o.fill ?? T.surface,
    strokeWidth: o.strokeWidth ?? 1.2,
    strokeStyle: o.strokeStyle,
  })];
  if (o.spine) {
    els.push(line([[x + 5, y + 9], [x + 5, y + h - 9]],
      { stroke: o.spine, strokeWidth: o.spineWidth ?? 4, roughness: 0.6 }));
  }
  return els;
}

// A hairline rule between table rows, drawn for every row but the first.
const rule = (x1, x2, y) => line([[x1, y], [x2, y]],
  { stroke: T.border, strokeWidth: 1, roughness: 0.3 });

// --- output -----------------------------------------------------------------
// A built scene is { W, H, els, title, desc }. `title` and `desc` are not
// optional: every diagram in this repository carries a real description that
// lets someone who cannot see it follow the argument it makes.
function emit(name, built) {
  const dir = process.argv[2];
  if (!dir) {
    console.error('usage: node postNN.js <output-directory>');
    process.exit(2);
  }
  if (!built.title || !built.desc) {
    console.error(`${name}: missing title or desc`);
    process.exit(1);
  }
  const svg = renderSvg(built.els, {
    width: built.W, height: built.H, title: built.title, desc: built.desc,
  });
  fs.writeFileSync(path.join(dir, `${name}.svg`), svg);
  fs.writeFileSync(path.join(dir, `${name}.excalidraw`), JSON.stringify(scene(built.els), null, 2));
  const warned = flushFit(name);
  console.log(`${name}: ${built.els.length} elements, ${built.W}x${built.H}`
    + (warned ? `  (${warned} fit warning${warned > 1 ? 's' : ''})` : ''));
}

module.exports = { M, heading, emit, fit, wrap, estWidth, card, rule };
