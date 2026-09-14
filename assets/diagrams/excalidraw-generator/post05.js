// The post 05 diagrams, hand-drawn.
//
// Scenes 01 to 03 mirror the clean vector figures one directory up. Scene 04 is
// new: sections 2.5 and 6 make the point that (3,), (1, 3) and (3, 1) print the
// same and broadcast differently, which is the root of every bug in this post,
// and no figure showed the three side by side.
//
//   01-axis-summation      960 x 460
//   02-keepdims-matters    960 x 500
//   03-broadcasting-rules  960 x 520
//   04-three-shapes        960 x 440   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

// A grid of cells with optional values. One outer rectangle plus interior
// rules; separate rectangles per cell read as noise at these sizes.
function grid(els, x, y, vals, cw, ch, o = {}) {
  const rows = vals.length, cols = vals[0].length;
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
  const size = o.size ?? 12;
  vals.forEach((row, j) => row.forEach((v, i) => {
    if (v === null || v === undefined) return;
    els.push(text(x + i * cw, y + j * ch + (ch - size) / 2 - 1, String(v), {
      size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: cw,
    }));
  }));
  return { w: cols * cw, h: rows * ch, bottom: y + rows * ch, cx: x + (cols * cw) / 2 };
}

// A row of axis-value boxes, used wherever a shape is compared against another.
function shapeRow(els, x, y, vals, colours, o = {}) {
  const bw = o.bw ?? 40, bh = o.bh ?? 34, size = o.size ?? 15;
  const centres = [];
  vals.forEach((v, i) => {
    els.push(rect(x + i * bw, y, bw, bh, {
      stroke: colours[i], fill: T.surface, strokeWidth: 1.5,
      strokeStyle: o.dashed && o.dashed[i] ? 'dashed' : 'solid', roundness: null,
    }));
    els.push(text(x + i * bw, y + (bh - size) / 2 - 1, String(v), {
      size, family: 3, stroke: o.ink ?? T.ink, align: 'center', width: bw,
    }));
    centres.push(x + i * bw + bw / 2);
  });
  return { centres, end: x + vals.length * bw, bottom: y + bh };
}

// ---------------------------------------------------------------- diagram 1
// Three panels over the same nine numbers. The slice that gets collapsed is
// outlined in the source grid, so "the axis you name is the axis that
// disappears" is something the eye can check rather than a sentence to trust.
function axisSummation() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Summing along an axis',
    'The same nine numbers, reduced three ways. The axis named in the call is the axis that disappears.')];

  const PW = 282, PX = [M, 339, 638], CW = 34, CH = 28;

  const PANELS = [
    {
      c: T.inkMuted, title: 'np.sum(a)', sub: 'axis=None · flatten everything',
      result: [[45]], shape: 'scalar', note: 'Every element in one sum.\nRarely what a network wants.',
      mark: 'all',
    },
    {
      c: T.primary, title: 'np.sum(a, axis=0)', sub: 'collapse the rows',
      result: [[12, 15, 18]], shape: '(3,)', note: 'One sum per column.\n1+4+7, 2+5+8, 3+6+9.',
      mark: 'cols',
    },
    {
      c: T.accent, title: 'np.sum(a, axis=1)', sub: 'collapse the columns',
      result: [[6, 15, 24]], shape: '(3,)', note: 'One sum per row.\n1+2+3, 4+5+6, 7+8+9.',
      mark: 'rows',
    },
  ];

  PANELS.forEach((p, i) => {
    const x = PX[i], gx = x + 90, inner = PW - 32;
    els.push(...card(x, 86, PW, 280, { spine: p.c }));
    els.push(text(x + 20, 98, fit(p.title, 14, inner * 0.94, 'panel title'),
      { size: 14, family: 3, stroke: p.c }));
    els.push(text(x + 20, 122, fit(p.sub, 11, inner, 'panel sub'), { size: 11, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + PW - 16, 148));

    grid(els, gx, 158, A, CW, CH);
    // The collapsed slices, outlined on the source rather than described.
    if (p.mark === 'cols') {
      [0, 1, 2].forEach((k) => els.push(rect(gx + k * CW, 158, CW, 3 * CH,
        { stroke: p.c, strokeWidth: 1.8, strokeStyle: 'dashed', roundness: null })));
    } else if (p.mark === 'rows') {
      [0, 1, 2].forEach((k) => els.push(rect(gx, 158 + k * CH, 3 * CW, CH,
        { stroke: p.c, strokeWidth: 1.8, strokeStyle: 'dashed', roundness: null })));
    } else {
      els.push(rect(gx, 158, 3 * CW, 3 * CH,
        { stroke: p.c, strokeWidth: 1.8, strokeStyle: 'dashed', roundness: null }));
    }

    els.push(arrow([[gx + 51, 248], [gx + 51, 272]], { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
    const rw = p.result[0].length * CW;
    grid(els, x + (PW - rw) / 2, 278, p.result, CW, CH, { c: p.c, ink: T.ink });
    els.push(text(x, 314, p.shape, { size: 12, family: 3, stroke: T.inkMuted, align: 'center', width: PW }));
    els.push(text(x + 20, 336, fit(p.note, 10, inner, 'panel note'), { size: 10, stroke: T.inkSubtle }));
  });

  els.push(rect(M, 390, 880, 60, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 402, 'The axis you name is the axis that disappears.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 426, 'Both reductions return shape (3,) here, and that 1-D shape is where the next figure’s bug begins.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Summing a 3-by-3 array along each axis',
    desc: 'Three hand-drawn panels reducing the same three-by-three array of one through nine. The '
      + 'first, np.sum of a with no axis, outlines the whole grid and produces the scalar '
      + 'forty-five. The second, axis equals nought, outlines each column and collapses the rows, '
      + 'producing twelve, fifteen, eighteen of shape three. The third, axis equals one, outlines '
      + 'each row and collapses the columns, producing six, fifteen, twenty-four, also of shape '
      + 'three. In every panel the slices being collapsed are outlined on the source grid itself. A '
      + 'band states that the axis named in the call is the axis that disappears, and notes that '
      + 'both reductions return a one-dimensional shape of three, which is where the silent '
      + 'broadcasting bug begins.',
  };
}

// ---------------------------------------------------------------- diagram 2
// The wrong result is shown in full rather than described. Its numbers are
// plausible and its shape is right, which is exactly why the bug survives to
// training time, and a figure that only said "wrong" would lose that.
function keepdimsMatters() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('One keyword between a right answer and a wrong one',
    'Subtracting the per-row maximum. Both calls run, both return a 3-by-3, and only one is correct.')];

  const CW = 40, CH = 30;

  const panel = (x, o) => {
    els.push(...card(x, 86, 420, 310, { spine: o.c }));
    els.push(text(x + 20, 98, o.title, { size: 15, stroke: o.c }));
    els.push(text(x + 20, 124, fit(o.call, 11, 380 * 0.94, 'call'),
      { size: 11, family: 3, stroke: T.ink }));
    els.push(rule(x + 16, x + 404, 152));

    els.push(text(x + 24, 162, 'the max is', { size: 10, stroke: T.inkSubtle }));
    grid(els, x + 24, 182, o.maxVals, 34, 28, { c: o.c });
    els.push(text(x + 24, o.maxShapeY, o.maxShape, { size: 12, family: 3, stroke: o.c }));
    els.push(text(x + 24, o.maxShapeY + 22, fit(o.broadcast, 10, 170, 'broadcast note'),
      { size: 10, stroke: T.inkSubtle }));

    els.push(text(x + 236, 162, 'a − max', { size: 10, stroke: T.inkSubtle }));
    grid(els, x + 236, 182, o.result, CW, CH, { c: o.c, size: 11 });

    els.push(rule(x + 16, x + 404, 300));
    els.push(text(x + 24, 310, fit(o.verdict, 12, 380, 'verdict'), { size: 12, stroke: o.c }));
    els.push(text(x + 24, 336, fit(o.why, 10, 380, 'why'), { size: 10, stroke: T.inkMuted }));
  };

  panel(M, {
    c: T.alert, title: 'Without keepdims',
    call: 'max_vals = np.max(a, axis=1)',
    maxVals: [[3, 6, 9]], maxShape: '(3,)', maxShapeY: 218,
    broadcast: 'a 1-D shape is padded to\n(1, 3) and stretches down',
    result: [[-2, -4, -6], [1, -1, -3], [4, 2, 0]],
    verdict: 'Silently wrong.',
    why: '3 comes off column 0 of every row, not off row 0.\nNo error is raised, and the result even has the right shape.',
  });

  panel(500, {
    c: T.success, title: 'With keepdims=True',
    call: 'max_vals = np.max(a, axis=1, keepdims=True)',
    maxVals: [[3], [6], [9]], maxShape: '(3, 1)', maxShapeY: 274,
    broadcast: 'a column stretches across',
    result: [[-2, -1, 0], [-2, -1, 0], [-2, -1, 0]],
    verdict: 'Correct.',
    why: '3 is taken off row 0, 6 off row 1, 9 off row 2.\nEvery row now ends at zero, which is what softmax needs.',
  });

  els.push(rect(M, 424, 880, 60, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 436, 'When reducing along an axis and then operating against the original 2-D array, always pass keepdims=True.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 460, 'It costs nothing when it is not needed, and the failure it prevents raises no error.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why keepdims=True matters: a silent wrong answer against a correct one',
    desc: 'Two hand-drawn panels subtracting the per-row maximum from a three-by-three array. The '
      + 'left panel, in alert red and labelled without keepdims, calls np.max with axis one; the '
      + 'maximum comes back as three, six, nine of shape three, a one-dimensional array that is '
      + 'padded to one by three and stretches downward. Its result is shown in full: minus two, '
      + 'minus four, minus six on the first row, one, minus one, minus three on the second, four, '
      + 'two, nought on the third, and it is labelled silently wrong because three is taken off '
      + 'column nought of every row rather than off row nought. The right panel, in success green '
      + 'and labelled with keepdims equals True, gets a column of shape three by one, which '
      + 'stretches across the columns, and every row of its result ends at zero. A band advises '
      + 'always passing keepdims when reducing along an axis and then operating against the original '
      + 'array, since it costs nothing and the failure it prevents raises no error.',
  };
}

// ---------------------------------------------------------------- diagram 3
// Four cases, three that broadcast and one that does not. Each pair of aligned
// axes gets its own verdict, because the rule is applied per axis and a figure
// that only judged the whole pair would hide where the failure actually is.
function broadcastingRules() {
  resetSeq();
  const W = 960, H = 520;

  const els = [...heading('The broadcasting rules, four cases',
    'Align trailing axes, stretch anything of size 1, and require the rest to be equal.')];

  const CASES = [
    {
      x: M, y: 86, title: '(3, 3)  +  (3, 1)', ok: true,
      a: [3, 3], b: [3, 1], pad: false,
      verdicts: ['equal', 'stretch'], out: '(3, 3)',
      note: 'The single column is\nreplicated across all\nthree columns.',
    },
    {
      x: 500, y: 86, title: '(3, 3)  +  (1, 3)', ok: true,
      a: [3, 3], b: [1, 3], pad: false,
      verdicts: ['stretch', 'equal'], out: '(3, 3)',
      note: 'The single row is\nreplicated down all\nthree rows.',
    },
    {
      x: M, y: 272, title: '(3, 3)  +  (3,)', ok: true,
      a: [3, 3], b: [1, 3], pad: true,
      verdicts: ['stretch', 'equal'], out: '(3, 3)',
      note: 'A 1-D shape is left-padded\nto (1, 3). It can never\nbecome a column.',
    },
    {
      x: 500, y: 272, title: '(3, 3)  +  (2, 3)', ok: false,
      a: [3, 3], b: [2, 3], pad: false,
      verdicts: ['fail', 'equal'], out: 'ValueError',
      note: 'Leading axes are 3 and 2.\nNeither is 1, so there is\nnothing to stretch.',
    },
  ];

  CASES.forEach((c) => {
    const CH = 168;
    els.push(...card(c.x, c.y, 420, CH, { spine: c.ok ? T.success : T.alert }));
    els.push(text(c.x + 20, c.y + 12, c.title, { size: 15, family: 3, stroke: T.ink }));
    els.push(text(c.x + 300, c.y + 14, c.ok ? 'broadcasts' : 'raises',
      { size: 11, stroke: c.ok ? T.success : T.alert, align: 'center', width: 104 }));
    els.push(rule(c.x + 16, c.x + 404, c.y + 40));

    const bx = c.x + 40;
    const colourFor = (v) => (v === 'fail' ? T.alert : v === 'stretch' ? T.accent : T.inkMuted);
    shapeRow(els, bx, c.y + 54, c.a, c.verdicts.map(colourFor));
    shapeRow(els, bx, c.y + 96, c.b, c.verdicts.map(colourFor),
      { dashed: c.pad ? [true, false] : [false, false] });
    els.push(text(c.x + 14, c.y + 62, 'a', { size: 11, stroke: T.inkSubtle }));
    els.push(text(c.x + 14, c.y + 104, 'b', { size: 11, stroke: T.inkSubtle }));

    // One verdict under each aligned pair, because the rule is applied per axis.
    c.verdicts.forEach((v, i) => {
      els.push(text(bx + i * 40 - 24, c.y + 134, v === 'fail' ? '3 ≠ 2' : v,
        { size: 10, stroke: colourFor(v), align: 'center', width: 88 }));
    });
    if (c.pad) {
      els.push(text(bx - 6, c.y + 134, '', { size: 10, stroke: T.inkSubtle }));
      els.push(text(c.x + 176, c.y + 100, 'padded', { size: 10, stroke: T.inkSubtle }));
    }

    els.push(arrow([[c.x + 190, c.y + 88], [c.x + 226, c.y + 88]],
      { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
    els.push(text(c.x + 236, c.y + 76, c.out,
      { size: 15, family: 3, stroke: c.ok ? T.success : T.alert }));
    els.push(text(c.x + 236, c.y + 104, fit(c.note, 10, 168, 'case note'),
      { size: 10, stroke: T.inkSubtle, width: 168 }));
  });

  els.push(rect(M, 464, 880, 50, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 478, 'Align the trailing axes, stretch any axis of size 1, and require every other pair to be equal.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'NumPy broadcasting rules with four worked examples',
    desc: 'Four hand-drawn cases, each showing two shapes as rows of axis boxes with a verdict under '
      + 'every aligned pair. A three-by-three plus a three-by-one broadcasts: the trailing pair '
      + 'stretches and the leading pair is equal, giving three by three. A three-by-three plus a '
      + 'one-by-three broadcasts the other way, the leading pair stretching. A three-by-three plus a '
      + 'bare shape-three array broadcasts too, its missing leading axis drawn as a dashed padded '
      + 'box, because a one-dimensional shape is left-padded to one by three and can never become a '
      + 'column. The fourth, a three-by-three plus a two-by-three, raises a ValueError: the leading '
      + 'axes are three and two, neither of them one, so there is nothing to stretch. A band states '
      + 'the rule: align the trailing axes, stretch any axis of size one, and require every other '
      + 'pair to be equal.',
  };
}

// ---------------------------------------------------------------- diagram 4
// New. Sections 2.5 and 6 both say that these three shapes print alike and
// behave differently, and that is the root of every bug in the post. Figure 02
// shows one consequence; this shows the taxonomy underneath it.
function threeShapes() {
  resetSeq();
  const W = 960, H = 440;

  const els = [...heading('Three shapes that print the same',
    'Two of them broadcast as a row and only one as a column. print() cannot tell them apart.')];

  const KINDS = [
    {
      c: T.alert, call: 'np.max(a, axis=1)', shown: '[3 6 9]', shape: '(3,)',
      as: 'a row', dir: 'down',
      note: 'Left-padded to (1, 3).\nNever a column, whichever\naxis produced it.',
    },
    {
      c: T.inkMuted, call: 'np.max(a, axis=0, keepdims=True)', shown: '[[7 8 9]]', shape: '(1, 3)',
      as: 'a row', dir: 'down',
      note: 'Already rank 2, so no\npadding is needed. Same\nbehaviour as the first.',
    },
    {
      c: T.success, call: 'np.max(a, axis=1, keepdims=True)', shown: '[[3]\n [6]\n [9]]', shape: '(3, 1)',
      as: 'a column', dir: 'right',
      note: 'The only one of the three\nthat stretches sideways,\nand the one softmax needs.',
    },
  ];

  const CWD = 280, TOP = 90, CHT = 244;
  KINDS.forEach((k, i) => {
    const x = M + i * 300, inner = CWD - 40;
    els.push(...card(x, TOP, CWD, CHT, { spine: k.c }));
    els.push(text(x + 20, TOP + 12, fit(k.call, 10, inner * 0.94, 'call'),
      { size: 10, family: 3, stroke: k.c }));
    els.push(rule(x + 16, x + CWD - 16, TOP + 36));

    els.push(text(x + 20, TOP + 44, 'prints as', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 20, TOP + 60, k.shown, { size: 12, family: 3, stroke: T.ink }));
    els.push(text(x + 168, TOP + 44, 'shape', { size: 10, stroke: T.inkSubtle }));
    els.push(text(x + 168, TOP + 60, k.shape, { size: 15, family: 3, stroke: k.c }));

    els.push(rule(x + 16, x + CWD - 16, TOP + 122));
    els.push(text(x + 20, TOP + 130, `broadcasts as ${k.as}`, { size: 11, stroke: k.c }));

    // A 3x3 target with the slice that gets replicated shaded, and arrows in
    // the direction it stretches.
    const gx = x + 20, gy = TOP + 152, cw = 18, ch = 16;
    grid(els, gx, gy, [[null, null, null], [null, null, null], [null, null, null]], cw, ch,
      { c: T.border, strokeWidth: 1.2 });
    if (k.dir === 'down') {
      els.push(rect(gx, gy, 3 * cw, ch, { stroke: k.c, fill: k.c, strokeWidth: 1.4, opacity: 30, roundness: null }));
      [0, 1, 2].forEach((j) => els.push(arrow([[gx + j * cw + cw / 2, gy + ch + 3], [gx + j * cw + cw / 2, gy + 3 * ch - 3]],
        { stroke: k.c, strokeWidth: 1, roughness: 0.4 })));
    } else {
      els.push(rect(gx, gy, cw, 3 * ch, { stroke: k.c, fill: k.c, strokeWidth: 1.4, opacity: 30, roundness: null }));
      [0, 1, 2].forEach((j) => els.push(arrow([[gx + cw + 3, gy + j * ch + ch / 2], [gx + 3 * cw - 3, gy + j * ch + ch / 2]],
        { stroke: k.c, strokeWidth: 1, roughness: 0.4 })));
    }
    els.push(text(x + 96, TOP + 156, fit(k.note, 10, inner - 76, 'kind note'), { size: 10, stroke: T.inkSubtle }));
  });

  els.push(rect(M, 358, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 370, 'A shape of (n,) and a shape of (1, n) are interchangeable. A shape of (n, 1) is not either of them.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 394, 'Every silent bug in this post is a reduction that returned the first when the third was wanted.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'A 1-D array, a row vector, and a column vector are three different things',
    desc: 'Three hand-drawn cards comparing shapes that print almost identically. The first, from '
      + 'np.max with axis one and no keepdims, prints as three six nine and has shape three; it is '
      + 'left-padded to one by three and broadcasts as a row, shown by shading the top row of a '
      + 'three-by-three target and drawing arrows downward. The second, from np.max with axis nought '
      + 'and keepdims, prints as a nested seven eight nine and has shape one by three; already rank '
      + 'two, it needs no padding and behaves identically to the first. The third, from np.max with '
      + 'axis one and keepdims, prints as a stacked three, six, nine and has shape three by one; it '
      + 'is the only one that broadcasts as a column, shown by shading the left column and drawing '
      + 'arrows rightward, and it is the one softmax needs. A band states that a shape of n and a '
      + 'shape of one by n are interchangeable while a shape of n by one is neither, and that every '
      + 'silent bug in the post is a reduction that returned the first when the third was wanted.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { axisSummation, keepdimsMatters, broadcastingRules, threeShapes };

if (require.main === module) {
  emit('01-axis-summation', axisSummation());
  emit('02-keepdims-matters', keepdimsMatters());
  emit('03-broadcasting-rules', broadcastingRules());
  emit('04-three-shapes', threeShapes());
}
