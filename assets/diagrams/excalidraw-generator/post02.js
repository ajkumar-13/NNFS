// The post 02 diagrams, hand-drawn.
//
// Scenes 01 to 04 mirror the clean vector figures one directory up, at the same
// canvas sizes, so the two are interchangeable. Scene 05 is new: section 3.1
// draws a boundary around np.dot that the post has no figure for, and the
// pitfalls list names "using * instead of np.dot" as a recurring error.
//
//   01-three-forms        960 x 540
//   02-order-matters      960 x 500
//   03-shape-rule         960 x 420
//   04-batch-transpose    960 x 500
//   05-not-the-same-call  960 x 460   (new)

const { T, rect, ellipse, text, line, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A grid of empty cells: one outer rectangle plus interior rules. Separate
// rectangles per cell read as noise at these sizes.
function gridInto(els, x, y, cols, rows, cw, ch, o = {}) {
  els.push(rect(x, y, cols * cw, rows * ch, {
    stroke: o.c ?? T.ink, fill: o.fill ?? T.surface,
    strokeWidth: o.strokeWidth ?? 1.4, roundness: null,
  }));
  for (let i = 1; i < cols; i++) {
    els.push(line([[x + i * cw, y], [x + i * cw, y + rows * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
  for (let j = 1; j < rows; j++) {
    els.push(line([[x, y + j * ch], [x + cols * cw, y + j * ch]],
      { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  }
}

// Numbers laid into a grid, one text element per cell.
function numsInto(els, x, y, rowsOfVals, cw, ch, o = {}) {
  rowsOfVals.forEach((row, j) => row.forEach((v, i) => {
    els.push(text(x + i * cw, y + j * ch + (ch - (o.size ?? 12)) / 2 - 1, String(v), {
      size: o.size ?? 12, family: 3, stroke: o.stroke ?? T.ink, align: 'center', width: cw,
    }));
  }));
}

// A shape drawn as its axes, one box each, rather than typeset as "(m, n)".
//
// The alternative is to tint one character inside a string, which means
// stepping x by a guessed font advance. The guess is wrong the moment the
// reader's monospace font is not the one it was measured against — Consolas
// advances at 0.55em where JetBrains Mono is 0.60, and over a short expression
// that drift is enough to pull the pieces visibly apart. Boxes are geometry the
// renderer controls, so the marks land where they are put in every font.
function shapeBoxes(els, x, y, vals, colours, o = {}) {
  const bw = o.bw ?? 42, bh = o.bh ?? 38, size = o.size ?? 17;
  const centres = [];
  vals.forEach((v, i) => {
    els.push(rect(x + i * bw, y, bw, bh, {
      stroke: colours[i], fill: T.surface, strokeWidth: 1.5, roundness: null,
    }));
    els.push(text(x + i * bw, y + (bh - size) / 2 - 1, String(v), {
      size, family: 3, stroke: T.ink, align: 'center', width: bw,
    }));
    centres.push(x + i * bw + bw / 2);
  });
  return { centres, end: x + vals.length * bw, bottom: y + bh };
}

// The dashed tie between the two axes that have to agree, with its verdict
// centred underneath. Drawn below the boxes so it never crosses a value.
function tie(els, els2, aX, bX, y, colour, label) {
  els.push(line([[aX, y], [(aX + bX) / 2, y + 30], [bX, y]],
    { stroke: colour, strokeWidth: 1.4, strokeStyle: 'dashed', roughness: 0.5 }));
  els.push(text((aX + bX) / 2 - 150, y + 36, label,
    { size: 11, stroke: colour, align: 'center', width: 300 }));
}

// ---------------------------------------------------------------- diagram 1
// Three panels, one per form. The grids are drawn empty: this figure is about
// which shapes are legal and what comes out, and numbers in forty-odd cells
// would argue for a different reading of the same picture.
function threeForms() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The three forms of np.dot',
    'One function name, three behaviours. The shapes of the arguments decide silently which one runs.')];

  const PW = 282, PX = [M, 339, 638], PY = 86, PH = 340;
  // The band between the two rules runs 152..316, so its centre is 234, not 200.
  const MIDY = 234;

  const FORMS = [
    {
      c: T.primary, title: 'Vector · Vector', sig: '(n,) · (n,) → scalar',
      mean: 'one neuron,\none sample',
      note: 'Order does not matter here.\nThe sum is commutative.',
    },
    {
      c: T.accent, title: 'Matrix · Vector', sig: '(m, n) · (n,) → (m,)',
      mean: 'a layer of m neurons,\none sample',
      note: 'One dot product per row\nof the matrix.',
    },
    {
      c: T.success, title: 'Matrix · Matrix', sig: '(m, n) · (n, p) → (m, p)',
      mean: 'a layer of m neurons,\np samples',
      note: 'One per (row of first,\ncolumn of second) pair.',
    },
  ];

  FORMS.forEach((f, i) => {
    const x = PX[i], inner = PW - 32;
    els.push(...card(x, PY, PW, PH, { spine: f.c }));
    els.push(text(x + 20, PY + 14, fit(f.title, 15, inner, 'form title'), { size: 15, stroke: f.c }));
    els.push(text(x + 20, PY + 40, fit(f.sig, 11, inner * 0.94, 'signature'),
      { size: 11, family: 3, stroke: T.inkMuted }));
    els.push(rule(x + 16, x + PW - 16, PY + 66));
    els.push(rule(x + 16, x + PW - 16, PY + 230));
    els.push(text(x + 20, PY + 242, 'in a network', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 20, PY + 260, fit(f.mean, 12, inner, 'meaning'), { size: 12, stroke: T.ink }));
    els.push(text(x + 20, PY + 300, fit(f.note, 10, inner, 'note'), { size: 10, stroke: T.inkSubtle }));
  });

  const op = (cx, s) => els.push(text(cx - 12, 220, s,
    { size: 16, stroke: T.inkMuted, align: 'center', width: 24 }));

  // Panel 1 — two length-4 strips into one cell.
  gridInto(els, 75, MIDY - 10, 4, 1, 20, 20, { c: T.primary });
  op(163, '·');
  gridInto(els, 171, MIDY - 10, 4, 1, 20, 20, { c: T.primary });
  op(259, '=');
  gridInto(els, 267, MIDY - 10, 1, 1, 20, 20, { c: T.ink });

  // Panel 2 — (m, n) against a column.
  gridInto(els, 404, MIDY - 27, 4, 3, 20, 18, { c: T.accent });
  op(492, '·');
  gridInto(els, 500, MIDY - 36, 1, 4, 20, 18, { c: T.accent });
  op(528, '=');
  gridInto(els, 536, MIDY - 27, 1, 3, 20, 18, { c: T.ink });

  // Panel 3 — (m, n) against (n, p).
  gridInto(els, 663, MIDY - 27, 4, 3, 20, 18, { c: T.success });
  op(751, '·');
  gridInto(els, 759, MIDY - 36, 3, 4, 20, 18, { c: T.success });
  op(827, '=');
  gridInto(els, 835, MIDY - 27, 3, 3, 20, 18, { c: T.ink });

  els.push(rect(M, 452, 880, 68, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 466, 'Inner dimensions must always match.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 490, 'The last axis of the first argument is contracted with the first axis of the second.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The three forms of np.dot',
    desc: 'Three hand-drawn panels, one per form of np.dot. The first, vector by vector, takes two '
      + 'one-dimensional arrays of length n and returns a single scalar; in a network that is one '
      + 'neuron on one sample, and the order of the arguments does not matter. The second, matrix by '
      + 'vector, takes an m by n matrix and a length-n vector and returns a length-m vector, one dot '
      + 'product per row; that is a layer of m neurons on one sample. The third, matrix by matrix, '
      + 'takes an m by n matrix and an n by p matrix and returns m by p, one dot product per pair of '
      + 'a row from the first and a column from the second; that is a layer of m neurons on p '
      + 'samples. Each panel draws the operand shapes as empty cell grids. A band across the bottom '
      + 'states that inner dimensions must always match, and that the last axis of the first '
      + 'argument is contracted with the first axis of the second.',
  };
}

// ---------------------------------------------------------------- diagram 2
// The same B and the same a in both panels, with only the argument order
// changed. Both results are the post's own printed output, so the figure can be
// checked against the code block rather than taken on trust.
function orderMatters() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('Order of arguments matters once a matrix is involved',
    'Same matrix, same vector, two orderings, two different answers. np.dot is not commutative here.')];

  const B = [[4, 5, 6], [7, 8, 9], [10, 11, 12]];
  const CWD = 34, CHT = 28;

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 292, { spine: o.c }));
    els.push(text(x + 20, 100, o.call, { size: 16, family: 3, stroke: o.c }));
    els.push(text(x + 20, 126, fit(o.sub, 11, 380, 'panel sub'), { size: 11, stroke: T.inkMuted }));
    els.push(rule(x + 16, x + 404, 150));
  };

  // --- left: np.dot(a, B) — a is a row, walking across columns
  panel(M, {
    c: T.primary, call: 'np.dot(a, B)',
    sub: 'a is a (1, 3) row · one dot product per column of B',
  });
  // a row, then B, then the result row.
  gridInto(els, 62, 222, 3, 1, CWD, CHT, { c: T.primary });
  numsInto(els, 62, 222, [[1, 2, 3]], CWD, CHT);
  els.push(text(62, 256, 'a', { size: 11, stroke: T.inkSubtle, align: 'center', width: 102 }));
  els.push(text(176, 224, '·', { size: 16, stroke: T.inkMuted, align: 'center', width: 24 }));
  gridInto(els, 212, 194, 3, 3, CWD, CHT, { c: T.primary });
  numsInto(els, 212, 194, B, CWD, CHT);
  els.push(text(212, 284, 'B', { size: 11, stroke: T.inkSubtle, align: 'center', width: 102 }));
  els.push(text(326, 224, '=', { size: 16, stroke: T.inkMuted, align: 'center', width: 24 }));
  gridInto(els, 356, 222, 3, 1, CWD, CHT, { c: T.ink });
  numsInto(els, 356, 222, [[48, 54, 60]], CWD, CHT, { size: 11 });
  // The walked axis, marked on the operand it is walked over.
  els.push(rect(212, 194, CWD, 3 * CHT, { stroke: T.primary, strokeWidth: 2, strokeStyle: 'dashed', roundness: null }));
  els.push(text(M, 320, '1·4 + 2·7 + 3·10 = 48   (column 1)',
    { size: 11, family: 3, stroke: T.primary, align: 'center', width: 420 }));
  els.push(text(M, 344, 'the vector lies flat and walks left to right',
    { size: 11, stroke: T.inkSubtle, align: 'center', width: 420 }));

  // --- right: np.dot(B, a) — a is a column, walking down rows
  panel(500, {
    c: T.accent, call: 'np.dot(B, a)',
    sub: 'a is a (3, 1) column · one dot product per row of B',
  });
  gridInto(els, 560, 194, 3, 3, CWD, CHT, { c: T.accent });
  numsInto(els, 560, 194, B, CWD, CHT);
  els.push(text(560, 284, 'B', { size: 11, stroke: T.inkSubtle, align: 'center', width: 102 }));
  els.push(text(674, 224, '·', { size: 16, stroke: T.inkMuted, align: 'center', width: 24 }));
  gridInto(els, 704, 194, 1, 3, CWD, CHT, { c: T.accent });
  numsInto(els, 704, 194, [[1], [2], [3]], CWD, CHT);
  els.push(text(704, 284, 'a', { size: 11, stroke: T.inkSubtle, align: 'center', width: 34 }));
  els.push(text(748, 224, '=', { size: 16, stroke: T.inkMuted, align: 'center', width: 24 }));
  gridInto(els, 778, 194, 1, 3, CWD, CHT, { c: T.ink });
  numsInto(els, 778, 194, [[32], [50], [68]], CWD, CHT, { size: 11 });
  els.push(rect(560, 194, 3 * CWD, CHT, { stroke: T.accent, strokeWidth: 2, strokeStyle: 'dashed', roundness: null }));
  els.push(text(500, 320, '4·1 + 5·2 + 6·3 = 32   (row 1)',
    { size: 11, family: 3, stroke: T.accent, align: 'center', width: 420 }));
  els.push(text(500, 344, 'the vector stands up and walks top to bottom',
    { size: 11, stroke: T.inkSubtle, align: 'center', width: 420 }));

  els.push(rect(M, 404, 880, 72, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 418, 'Both calls succeed only because B is square, so either axis can be the contracted one.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 442, 'Make B non-square and one of the two orderings raises ValueError instead.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Order of arguments matters: np.dot(a, B) versus np.dot(B, a)',
    desc: 'Two hand-drawn panels holding the same three-by-three matrix B, with entries four through '
      + 'twelve, and the same length-three vector a of one, two, three, differing only in the order '
      + 'of the arguments. In the left panel, np.dot of a and B, the vector lies flat as a row and '
      + 'one dot product is taken per column of B; the first column is outlined and the worked sum '
      + 'one times four plus two times seven plus three times ten equals forty-eight is written out, '
      + 'giving the result forty-eight, fifty-four, sixty. In the right panel, np.dot of B and a, the '
      + 'vector stands up as a column and one dot product is taken per row of B; the first row is '
      + 'outlined and four times one plus five times two plus six times three equals thirty-two is '
      + 'written out, giving thirty-two, fifty, sixty-eight. A band beneath notes that both calls '
      + 'succeed only because B is square, and that a non-square matrix makes one of the two '
      + 'orderings raise a ValueError.',
  };
}

// ---------------------------------------------------------------- diagram 3
// The rule stated once, then split into the two halves that matter, then run on
// real numbers. The coloured digits in the worked line are placed by stepping
// through a monospace advance, which is the only reliable way to tint one
// character inside a longer expression.
function shapeRule() {
  resetSeq();
  const W = 960, H = 420;

  const els = [...heading('The shape rule, once and for all',
    'Two numbers have to be equal and then vanish. The two that are left are the answer.')];

  // Each shape as its two axes. Widths: 84 per pair, 32 per operator gap, so
  // the run spans 316 and starts at 322 to land centred on the 960 canvas.
  const BW = 42, BY = 102;
  const A = shapeBoxes(els, 322, BY, ['m', 'n'], [T.primary, T.accent]);
  const Bs = shapeBoxes(els, 438, BY, ['n', 'p'], [T.accent, T.primary]);
  const C = shapeBoxes(els, 554, BY, ['m', 'p'], [T.primary, T.primary]);
  els.push(text(410, BY + 8, '·', { size: 18, stroke: T.inkMuted, align: 'center', width: 24 }));
  els.push(text(526, BY + 8, '→', { size: 18, stroke: T.inkMuted, align: 'center', width: 24 }));
  [[322, '(m, n)'], [438, '(n, p)'], [554, '(m, p)']].forEach(([x, lab]) => {
    els.push(text(x, BY - 24, lab, { size: 12, stroke: T.inkSubtle, align: 'center', width: BW * 2 }));
  });
  tie(els, null, A.centres[1], Bs.centres[0], A.bottom, T.accent, 'equal, or the call raises · summed away');
  els.push(text((C.centres[0] + C.centres[1]) / 2 - 60, A.bottom + 12, 'these two survive',
    { size: 11, stroke: T.primary, align: 'center', width: 120 }));

  const HALVES = [
    {
      x: 140, c: T.accent, head: 'Inner · the two n values',
      a: 'Must be equal, or the call raises.',
      b: 'Summed over, so they do not\nappear in the result at all.',
    },
    {
      x: 500, c: T.primary, head: 'Outer · m and p',
      a: 'Never compared, never summed.',
      b: 'They pass straight through and\nbecome the result shape.',
    },
  ];
  HALVES.forEach((h) => {
    els.push(...card(h.x, 206, 320, 88, { spine: h.c }));
    els.push(text(h.x + 20, 216, fit(h.head, 13, 286, 'half head'), { size: 13, stroke: h.c }));
    els.push(text(h.x + 20, 240, fit(h.a, 11, 286, 'half line 1'), { size: 11, stroke: T.ink }));
    els.push(text(h.x + 20, 260, fit(h.b, 11, 286, 'half line 2'), { size: 11, stroke: T.inkMuted }));
  });

  els.push(text(0, 308, 'worked:   (3, 4) · (4, 3)  →  (3, 3)      the two 4s are the inner pair',
    { size: 13, stroke: T.inkMuted, align: 'center', width: W }));

  els.push(rect(M, 336, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 350, 'Inner dimensions match. Outer dimensions become the result.',
    { size: 14, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 374, 'A 1-D vector counts as a row on the left of the call and a column on the right.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The shape rule for matrix multiplication: inner dimensions match, outer dimensions survive',
    desc: 'A hand-drawn statement of the shape rule for np.dot. The generic expression m by n dotted '
      + 'with n by p giving m by p is written large across the top. Two cards below split it: the '
      + 'inner pair, the two n values, must be equal or the call raises, and they are summed over so '
      + 'they do not appear in the result; the outer pair, m and p, are never compared and never '
      + 'summed, and pass straight through to become the result shape. A worked example underneath '
      + 'reads three by four dotted with four by three giving three by three, with the two fours '
      + 'tinted and joined by a dashed link labelled equal and contracted away. A band at the bottom '
      + 'states the rule in one sentence: inner dimensions match, outer dimensions become the '
      + 'result, and notes that a one-dimensional vector counts as a row on the left of the call and '
      + 'a column on the right.',
  };
}

// ---------------------------------------------------------------- diagram 4
// The failing call above the working one, at equal weight. The point of the
// figure is that one character of difference moves the call from ValueError to
// a clean result, and stacking them is what makes that one character visible.
function batchTranspose() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('Why batching needs a transpose',
    'Inputs lead with the sample count and weights lead with the neuron count. One of them has to turn.')];

  const ROWS = [
    {
      y: 90, c: T.alert, label: 'Without the transpose',
      call: 'np.dot(inputs, weights)', concrete: 'inputs (3, 4) · weights (3, 4)',
      left: ['N', 'n'], right: ['m', 'n'], labL: '(N, n)', labR: '(m, n)',
      verdict: 'n is 4 and m is 3 · not aligned',
      outHead: 'ValueError', outA: 'shapes (3,4) and (3,4)', outB: 'not aligned: 4 != 3',
    },
    {
      y: 270, c: T.success, label: 'With the transpose',
      call: 'np.dot(inputs, weights.T)', concrete: 'inputs (3, 4) · weights.T (4, 3)',
      left: ['N', 'n'], right: ['n', 'm'], labL: '(N, n)', labR: '(n, m)',
      verdict: 'n is 4 on both sides · aligned',
      outHead: '(3, 3)', outA: 'one row per sample', outB: 'one column per neuron',
    },
  ];

  ROWS.forEach((r) => {
    els.push(...card(M, r.y, 880, 160, { spine: r.c }));
    els.push(text(62, r.y + 14, fit(r.label, 15, 240, 'row label'), { size: 15, stroke: r.c }));
    els.push(text(62, r.y + 42, fit(r.call, 12, 236 * 0.94, 'call'),
      { size: 12, family: 3, stroke: T.ink }));
    els.push(text(62, r.y + 66, fit(r.concrete, 10, 236 * 0.94, 'concrete'),
      { size: 10, family: 3, stroke: T.inkSubtle }));

    // The contracted axis is the second box of the left shape against the first
    // box of the right one, whichever letters happen to sit in them.
    const BW = 40, BH = 34, BY = r.y + 40;
    const L = shapeBoxes(els, 340, BY, r.left, [T.inkMuted, r.c], { bw: BW, bh: BH, size: 15 });
    const R = shapeBoxes(els, 452, BY, r.right, [r.c, T.inkMuted], { bw: BW, bh: BH, size: 15 });
    els.push(text(424, BY + 6, '·', { size: 16, stroke: T.inkMuted, align: 'center', width: 24 }));
    els.push(text(340, BY - 20, r.labL, { size: 11, stroke: T.inkSubtle, align: 'center', width: BW * 2 }));
    els.push(text(452, BY - 20, r.labR, { size: 11, stroke: T.inkSubtle, align: 'center', width: BW * 2 }));
    tie(els, null, L.centres[1], R.centres[0], L.bottom, r.c, r.verdict);

    els.push(rect(660, r.y + 34, 240, 92, { stroke: r.c, fill: T.surface, strokeWidth: 1.4 }));
    els.push(text(660, r.y + 46, fit(r.outHead, 15, 220, 'outcome head'),
      { size: 15, family: 3, stroke: T.ink, align: 'center', width: 240 }));
    els.push(text(660, r.y + 74, fit(r.outA, 10, 220 * 0.94, 'outcome line 1'),
      { size: 10, family: 3, stroke: T.inkMuted, align: 'center', width: 240 }));
    els.push(text(660, r.y + 94, fit(r.outB, 10, 220 * 0.94, 'outcome line 2'),
      { size: 10, family: 3, stroke: T.inkMuted, align: 'center', width: 240 }));
  });

  els.push(text(M, 452, 'The transpose is bookkeeping, not arithmetic: every output is still one input row dotted with one weight row.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 474, 'Single sample: W · x + b.        Batch: X · Wᵀ + b.',
    { size: 12, stroke: T.ink }));

  return {
    W, H, els,
    title: 'Why batching needs a transpose',
    desc: 'Two hand-drawn rows at equal weight. The upper row, marked in alert red and labelled '
      + 'without the transpose, shows np.dot of inputs and weights, with inputs of shape three by '
      + 'four and weights of shape three by four. Its shape expression, N by n dotted with m by n, '
      + 'has the inner n and m tinted and joined by a dashed link reading four is not equal to three '
      + 'and not aligned; the outcome box on the right is a ValueError. The lower row, marked in '
      + 'success green and labelled with the transpose, shows np.dot of inputs and weights '
      + 'transposed, giving shapes three by four dotted with four by three. Its expression, N by n '
      + 'dotted with n by m, has both inner values tinted and joined by a link reading four equals '
      + 'four and aligned; the outcome box is a clean three by three, one row per sample and one '
      + 'column per neuron. A footer notes that the transpose is bookkeeping rather than arithmetic, '
      + 'since every output is still one input row dotted with one weight row, and gives both forms: '
      + 'W dot x plus b for a single sample, X dot W transposed plus b for a batch.',
  };
}

// ---------------------------------------------------------------- diagram 5
// New. Section 3.1 draws a boundary around np.dot and the post has no figure
// for it, while the pitfalls list names reaching for * where np.dot belongs as
// a recurring error. Three calls that look alike, with what each does to shape
// as the thing that separates them.
function notTheSameCall() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Three calls that look alike and are not',
    'What separates them is what each one does to the shape. Only two of the three contract anything.')];

  const CALLS = [
    {
      c: T.primary, call: 'np.dot(A, B)', kind: 'contraction',
      shape: '(m, n) · (n, p)', out: '→ (m, p)',
      body: 'Sums over the shared axis.\nThe inner dimensions must\nbe equal, and they vanish.',
      note: 'this is the layer call\nrank drops by the shared axis',
    },
    {
      c: T.alert, call: 'A * B', kind: 'element-wise',
      shape: '(m, n) * (m, n)', out: '→ (m, n)',
      body: 'Multiplies position by\nposition. Shapes must match\nor broadcast. Nothing is summed.',
      note: 'never a layer call\nshape in = shape out',
    },
    {
      c: T.success, call: 'A @ B', kind: 'matmul',
      shape: '(m, n) @ (n, p)', out: '→ (m, p)',
      body: 'Identical to np.dot at 1-D\nand 2-D. At rank 3 and above\nit broadcasts instead.',
      note: 'interchangeable here\nsame result as np.dot',
    },
  ];

  const CWD = 280, GAP = 20, TOP = 94, CHT = 250;
  CALLS.forEach((c, i) => {
    const x = M + i * (CWD + GAP), inner = CWD - 36;
    els.push(...card(x, TOP, CWD, CHT, { spine: c.c }));
    els.push(text(x + 20, TOP + 14, fit(c.call, 16, inner * 0.94, 'call'),
      { size: 16, family: 3, stroke: c.c }));
    els.push(text(x + 20, TOP + 40, c.kind, { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + CWD - 16, TOP + 64));
    els.push(text(x + 20, TOP + 76, fit(c.shape, 12, inner * 0.94, 'shape'),
      { size: 12, family: 3, stroke: T.ink }));
    els.push(text(x + 20, TOP + 98, fit(c.out, 12, inner * 0.94, 'out'),
      { size: 12, family: 3, stroke: T.inkMuted }));
    els.push(rule(x + 16, x + CWD - 16, TOP + 128));
    els.push(text(x + 20, TOP + 140, fit(c.body, 11, inner, 'body'), { size: 11, stroke: T.inkMuted }));
    els.push(text(x + 20, TOP + 214, fit(c.note, 11, inner, 'note'), { size: 11, stroke: T.inkSubtle }));
  });

  // The middle card is the odd one out: it is the only call here whose output
  // carries the same shape as its inputs. The lasso is stroke-only and sits
  // inside the card, around the two shape lines alone — drawn any wider it
  // crosses into the cards on either side.
  els.push(ellipse(350, TOP + 70, 170, 48, { stroke: T.alert, strokeWidth: 2, roughness: 1.5 }));

  els.push(rect(M, 372, 880, 64, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 386, 'Every array in this series is at most 2-D, so np.dot and @ agree throughout.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 410, 'Reaching for * where np.dot belongs returns an array of the wrong shape rather than an error, which is why it survives to training time.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'np.dot, element-wise multiply, and matmul are three different calls',
    desc: 'Three hand-drawn cards separating calls that look alike. The first, np.dot of A and B, is '
      + 'a contraction: m by n dotted with n by p gives m by p, summing over the shared axis, whose '
      + 'inner dimensions must be equal and then vanish; this is the layer call. The second, A star '
      + 'B, is element-wise: m by n times m by n gives m by n, multiplying position by position with '
      + 'nothing summed, and it is circled and labelled shape in equals shape out because it is the '
      + 'only one of the three that contracts nothing; it is never a layer call. The third, A at B, '
      + 'is matmul: identical to np.dot at one and two dimensions, but broadcasting leading axes at '
      + 'rank three and above. A band at the bottom notes that every array in this series is at most '
      + 'two-dimensional so np.dot and the at operator agree throughout, and that reaching for star '
      + 'where np.dot belongs returns an array of the wrong shape rather than an error, which is why '
      + 'the mistake survives to training time.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { threeForms, orderMatters, shapeRule, batchTranspose, notTheSameCall };

if (require.main === module) {
  emit('01-three-forms', threeForms());
  emit('02-order-matters', orderMatters());
  emit('03-shape-rule', shapeRule());
  emit('04-batch-transpose', batchTranspose());
  emit('05-not-the-same-call', notTheSameCall());
}
