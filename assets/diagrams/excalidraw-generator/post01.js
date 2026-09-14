// The post 01 diagrams, hand-drawn.
//
// Scenes 01 to 05 are the same arguments as the clean vector figures one
// directory up, at the same canvas sizes, so the two are interchangeable in the
// post. What changes is the register: a sketch rather than a diagram, for
// slides and talks. Scene 06 is new — section 11 makes the structural claim of
// the whole series and the post had no figure for it.
//
// Canvas sizes are inherited from the published figures on purpose:
//   01-neuron-anatomy            960 x 540
//   02-layer-as-stacked-neurons  960 x 500
//   03-three-implementations     960 x 520
//   04-shape-diary               960 x 440
//   05-batch-broadcasting        960 x 500
//   06-what-gets-added           960 x 460   (new)
// Keeping them identical is what lets a figure be swapped for its alternate
// without touching a single Markdown image tag.

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// Every number in these scenes is the post's own worked example: inputs
// [1, 2, 3, 2.5], three neurons, biases [2, 3, 0.5], printed [4.8, 1.21, 2.385].
// Nothing here is illustrative — a reader can check the figure against the code
// block, which is the only reason a figure is allowed to carry numbers at all.
const OUT = '[4.8, 1.21, 2.385]';

const SUB = ['₁', '₂', '₃', '₄'];

// ---------------------------------------------------------------- diagram 1
// The hero. Four inputs fan into one summation node, the bias joins from above
// rather than from the left, and four annotation cards down the right margin
// name each part. The bias enters on its own axis because it is the one term
// with no input to multiply, and a diagram that lets it arrive with the others
// loses exactly that.
function neuronAnatomy() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('A neuron is a weighted sum plus a bias',
    'Four inputs, four weights, one bias, one number out. Every parameter in a network is one of the middle two.')];

  const IN_X = 100, SUM_X = 352, SUM_Y = 242, R = 46;
  const ys = [152, 212, 272, 332];

  ys.forEach((y, i) => {
    els.push(circle(IN_X, y, 20, { stroke: T.primary, fill: T.surface, strokeWidth: 1.5 }));
    els.push(text(IN_X - 20, y - 10, `x${SUB[i]}`, { size: 15, stroke: T.ink, align: 'center', width: 40 }));
    els.push(line([[IN_X + 20, y], [SUM_X - R + 2, SUM_Y]],
      { stroke: T.primary, strokeWidth: 1.2, roughness: 0.6 }));
    // The weight rides above the shoulder of its own input node, where the fan
    // is widest. Placed at the midpoint it would sit on the line, and four
    // labels 27px apart would read as one smudge.
    els.push(text(IN_X + 30, y - 32, `w${SUB[i]}`, { size: 13, stroke: T.primary }));
  });
  els.push(text(IN_X - 60, 366, 'inputs · shape (n,)',
    { size: 11, stroke: T.inkSubtle, align: 'center', width: 120 }));

  // The bias, arriving from above on a dashed line.
  els.push(circle(SUM_X, 126, 20, { stroke: T.success, fill: T.surface, strokeWidth: 1.5 }));
  els.push(text(SUM_X - 20, 116, 'b', { size: 15, stroke: T.ink, align: 'center', width: 40 }));
  els.push(line([[SUM_X, 146], [SUM_X, SUM_Y - R]],
    { stroke: T.success, strokeWidth: 1.2, strokeStyle: 'dashed', roughness: 0.5 }));
  els.push(text(SUM_X + 26, 158, 'added once,', { size: 11, stroke: T.success }));
  els.push(text(SUM_X + 26, 174, 'not per input', { size: 11, stroke: T.success }));

  // The summation node.
  els.push(circle(SUM_X, SUM_Y, R, { stroke: T.accent, fill: T.surface, strokeWidth: 1.8 }));
  els.push(text(SUM_X - R, SUM_Y - 26, 'Σ', { size: 26, stroke: T.ink, align: 'center', width: R * 2 }));
  els.push(text(SUM_X - R, SUM_Y + 8, fit('w·x + b', 11, R * 2, 'sum node caption'),
    { size: 11, stroke: T.inkMuted, align: 'center', width: R * 2 }));

  // Output.
  els.push(arrow([[SUM_X + R, SUM_Y], [486, SUM_Y]], { stroke: T.accent, strokeWidth: 1.8, roughness: 0.5 }));
  els.push(rect(492, 220, 112, 44, { stroke: T.accent, fill: T.surface, strokeWidth: 1.5 }));
  els.push(text(492, 228, fit('ŷ = 4.8', 17, 112, 'output pill'),
    { size: 17, stroke: T.ink, align: 'center', width: 112 }));
  els.push(text(492, 272, 'one scalar out', { size: 11, stroke: T.inkSubtle, align: 'center', width: 112 }));

  // Annotation cards: the same four rows as the component table in section 3,
  // in the same order, so figure and table can be read against each other.
  const CX = 636, CW = 284, CH = 78, PAD = 20;
  const CARDS = [
    { c: T.primary, head: 'Inputs · xᵢ · (n,)', a: 'The data fed in.', b: 'Changes every forward pass.' },
    { c: T.primary, head: 'Weights · wᵢ · (n,)', a: 'Importance of each input.', b: 'Learned, frozen at inference.' },
    { c: T.success, head: 'Bias · b · scalar', a: 'A constant offset, one per neuron.', b: 'Learned, frozen at inference.' },
    { c: T.accent, head: 'Output · ŷ · scalar', a: 'The weighted sum plus the bias.', b: 'Recomputed every forward pass.' },
  ];
  CARDS.forEach((c, i) => {
    const y = 92 + i * 92;
    const inner = CW - PAD - 12;
    els.push(...card(CX, y, CW, CH, { spine: c.c }));
    els.push(text(CX + PAD, y + 10, fit(c.head, 13, inner, `card ${i + 1} head`), { size: 13, stroke: T.ink }));
    els.push(text(CX + PAD, y + 32, fit(c.a, 11, inner, `card ${i + 1} line 1`), { size: 11, stroke: T.inkMuted }));
    els.push(text(CX + PAD, y + 50, fit(c.b, 11, inner, `card ${i + 1} line 2`), { size: 11, stroke: T.inkSubtle }));
  });

  els.push(text(40, 414,
    fit('output  =  w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b  =  4.8', 15, 560, 'worked formula'),
    { size: 15, stroke: T.ink, align: 'center', width: 560 }));

  els.push(text(M, 486, 'Rosenblatt published this arithmetic in 1958. Sixty-eight years later the forward pass is unchanged;',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 506, 'what grew around it is the activation, the loss, the optimiser, and the depth of the stack.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'A neuron is a weighted sum plus a bias',
    desc: 'A single neuron drawn by hand at hero scale. Four input nodes labelled x1 through x4 sit on '
      + 'the left, each sending a weighted connection labelled w1 through w4 into a central summation '
      + 'node. A separate bias term b joins from above on a dashed line, annotated as added once '
      + 'rather than per input. The summation node emits one scalar output of 4.8 on the right. Four '
      + 'annotation cards down the right margin name the parts in the same order as the component '
      + 'table in the post: inputs and weights of shape n, a scalar bias, and a scalar output, each '
      + 'with its role and its lifetime. The worked arithmetic is written out beneath the drawing, '
      + 'and a footer notes that Rosenblatt published this arithmetic in 1958 and the forward pass '
      + 'has not changed since.',
  };
}

// ---------------------------------------------------------------- diagram 2
// Two panels sharing one input vector. Each neuron's fan gets its own colour,
// which is what turns "each neuron owns its own weights" from a caption into
// something visible: three colours leaving the same four dots.
function layerAsStackedNeurons() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('A layer is several neurons sharing the same inputs',
    'Same four inputs into both panels. What multiplies by three is the weights and the biases, not the data.')];

  const panel = (x, title, note) => {
    els.push(rect(x, 86, 420, 286, { stroke: T.border, fill: T.surface, strokeWidth: 1.2 }));
    els.push(text(x + 20, 98, fit(title, 15, 380, 'panel title'), { size: 15, stroke: T.ink }));
    els.push(text(x + 20, 120, note, { size: 11, stroke: T.inkSubtle }));
  };

  const inputs = (x) => {
    const ys = [162, 200, 238, 276];
    ys.forEach((y, i) => {
      els.push(circle(x, y, 11, { stroke: T.inkMuted, fill: T.surface, strokeWidth: 1.2 }));
      els.push(text(x - 11, y - 7, String(i + 1), { size: 10, stroke: T.inkMuted, align: 'center', width: 22 }));
    });
    return ys;
  };

  // --- left panel: one neuron
  panel(M, 'One neuron', 'four inputs · one output');
  const lys = inputs(90);
  lys.forEach((y) => els.push(line([[101, y], [266, 219]],
    { stroke: T.primary, strokeWidth: 1.1, roughness: 0.5 })));
  els.push(circle(300, 219, 34, { stroke: T.primary, fill: T.surface, strokeWidth: 1.6 }));
  els.push(text(266, 208, 'Σ', { size: 20, stroke: T.ink, align: 'center', width: 68 }));
  els.push(arrow([[334, 219], [400, 219]], { stroke: T.inkMuted, strokeWidth: 1.2, roughness: 0.5 }));
  els.push(text(M, 306, '5 parameters', { size: 17, stroke: T.ink, align: 'center', width: 420 }));
  els.push(text(M, 334, '4 weights  +  1 bias', { size: 11, stroke: T.inkMuted, align: 'center', width: 420 }));

  // --- right panel: three neurons
  // Radius 24 on a 53px pitch rather than 26 on 59: the bottom node has to
  // clear the parameter count beneath it, and at 26 it sat on the text.
  panel(500, 'A layer of three neurons', 'four inputs · three outputs');
  const rys = inputs(550);
  const NEU = [{ y: 166, c: T.primary }, { y: 219, c: T.accent }, { y: 272, c: T.success }];
  // Twelve fan lines first, nodes after, so the lines terminate under the node
  // outlines instead of crossing them.
  NEU.forEach((n) => {
    rys.forEach((y) => els.push(line([[561, y], [736, n.y]],
      { stroke: n.c, strokeWidth: 1, roughness: 0.5 })));
  });
  NEU.forEach((n, i) => {
    els.push(circle(760, n.y, 24, { stroke: n.c, fill: T.surface, strokeWidth: 1.6 }));
    els.push(text(734, n.y - 11, String(i + 1), { size: 15, stroke: T.ink, align: 'center', width: 52 }));
    els.push(arrow([[784, n.y], [860, n.y]], { stroke: T.inkMuted, strokeWidth: 1.2, roughness: 0.5 }));
  });
  els.push(text(500, 306, '15 parameters', { size: 17, stroke: T.ink, align: 'center', width: 420 }));
  els.push(text(500, 334, '12 weights  +  3 biases', { size: 11, stroke: T.inkMuted, align: 'center', width: 420 }));

  // --- the general rule
  els.push(rect(M, 396, 880, 68, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 410,
    fit('One neuron with n inputs has n + 1 parameters.  A layer of m such neurons has m (n + 1).', 13, 880, 'rule line 1'),
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 434,
    fit('Only the bias count equals the neuron count. The weight count is neurons × inputs.', 11, 880, 'rule line 2'),
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'A layer is several neurons sharing the same inputs',
    desc: 'Two hand-drawn panels side by side, both fed by the same four numbered input dots. The left '
      + 'panel holds a single neuron receiving four weighted connections and emitting one output, '
      + 'counted underneath as five parameters: four weights plus one bias. The right panel holds '
      + 'three neurons, each drawn in its own colour and each receiving its own fan of four '
      + 'connections from the same four inputs, emitting three outputs, counted underneath as '
      + 'fifteen parameters: twelve weights plus three biases. Colouring each neuron’s fan '
      + 'separately shows that the neurons share the data but not the weights. A band across the '
      + 'bottom states the general rule: one neuron with n inputs has n plus one parameters, and a '
      + 'layer of m such neurons has m times n plus one, with only the bias count equal to the '
      + 'neuron count.',
  };
}

// ---------------------------------------------------------------- diagram 3
// Three columns of code with an identical result line at the foot of each. The
// repeated result is the argument: the arithmetic is invariant and only the
// representation changes, so the three columns have to agree digit for digit.
function threeImplementations() {
  resetSeq();
  const W = 960, H = 520;

  const els = [...heading('The same layer, three implementations',
    'Identical numbers out of all three. What changes is how much code says it, and how fast it runs.')];

  const COLS = [
    {
      c: T.alert, title: 'By hand', sub: 'hardcoded · does not scale',
      code: 'outputs = [\n'
        + '  x[0]*w[0][0] + x[1]*w[0][1]\n'
        + '  + x[2]*w[0][2] + x[3]*w[0][3]\n'
        + '  + b[0],\n\n'
        + '  x[0]*w[1][0] + x[1]*w[1][1]\n'
        + '  + x[2]*w[1][2] + x[3]*w[1][3]\n'
        + '  + b[1],\n\n'
        + '  # ... again for neuron 3\n'
        + ']',
      metric: ['lines of code   ~13', 'scales?   no', 'speed   moot at n = 3'],
    },
    {
      c: T.warn, title: 'Two loops', sub: 'scales · clean · still slow',
      code: 'outputs = []\n'
        + 'for w_row, b in zip(w, biases):\n'
        + '    out = 0\n'
        + '    for xi, wi in zip(x, w_row):\n'
        + '        out += xi * wi\n'
        + '    out += b\n'
        + '    outputs.append(out)',
      metric: ['lines of code   7', 'scales?   yes', 'speed   one op per bytecode'],
    },
    {
      c: T.success, title: 'NumPy', sub: 'scales · concise · vectorised C',
      code: 'import numpy as np\n\n'
        + 'outputs = np.dot(w, x) + biases\n\n\n'
        + '# w is (3, 4), x is (4,)\n'
        + '# -> outputs is (3,)',
      metric: ['lines of code   1', 'scales?   yes', 'speed   compiled, whole array'],
    },
  ];

  const CW = 280, GAP = 20, TOP = 86, CH = 300;
  COLS.forEach((col, i) => {
    const x = M + i * (CW + GAP);
    els.push(...card(x, TOP, CW, CH));
    els.push(line([[x + 5, TOP + 10], [x + 5, TOP + 46]], { stroke: col.c, strokeWidth: 4, roughness: 0.6 }));
    els.push(text(x + 20, TOP + 10, col.title, { size: 15, stroke: col.c }));
    els.push(text(x + 20, TOP + 32, fit(col.sub, 10, CW - 34, 'column caption'), { size: 10, stroke: T.inkSubtle }));
    els.push(rule(x + 14, x + CW - 14, TOP + 56));
    // Mono advances wider than the heading font, so code is checked at 0.62em
    // by asking fit() for a proportionally smaller box.
    els.push(text(x + 16, TOP + 68, fit(col.code, 10, (CW - 30) * 0.94, 'code block'),
      { size: 10, stroke: T.ink, family: 3 }));
    // The shared result, repeated verbatim under all three.
    els.push(rule(x + 14, x + CW - 14, TOP + 248));
    els.push(text(x, TOP + 258, OUT, { size: 12, stroke: T.ink, family: 3, align: 'center', width: CW }));
    els.push(text(x, TOP + 278, 'same numbers, every time',
      { size: 10, stroke: T.inkSubtle, align: 'center', width: CW }));
  });

  // The comparison band, drawn before the metrics so they sit on top of it.
  els.push(rect(M, 406, 880, 84, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  COLS.forEach((col, i) => {
    const x = M + i * (CW + GAP);
    els.push(text(x + 16, 424, fit(col.metric.join('\n'), 11, CW - 30, 'metric block'),
      { size: 11, stroke: T.inkMuted }));
  });

  return {
    W, H, els,
    title: 'The same layer computed three ways: by hand, with loops, with NumPy',
    desc: 'Three hand-drawn columns of code, each computing the output of the same three-neuron layer. '
      + 'The first column, marked in alert red and titled by hand, writes out every input-weight '
      + 'product for every neuron and is captioned hardcoded and does not scale. The second, in warn '
      + 'tan, uses two nested for-loops over neurons and inputs and is captioned as scaling and clean '
      + 'but still slow. The third, in success green, is the single line outputs equals np.dot of w '
      + 'and x plus biases, captioned as scaling, concise, and vectorised C. The identical result '
      + 'list 4.8, 1.21, 2.385 is repeated at the foot of all three columns to show the arithmetic is '
      + 'unchanged. A band underneath compares the three on lines of code, whether they scale, and '
      + 'speed: roughly thirteen lines that do not scale, seven lines running one operation per '
      + 'bytecode, and one line compiled over the whole array.',
  };
}

// ---------------------------------------------------------------- diagram 4
// The shape table from section 10, as a table. The batch row is tinted because
// it is the only one where the transpose appears, and the transpose is the
// thing readers get wrong.
function shapeDiary() {
  resetSeq();
  const W = 960, H = 440;

  const els = [...heading('Shape diary: single neuron, single layer, whole batch',
    'The arithmetic is identical in all three rows. Only the call signature moves.')];

  const COLS = [
    { x: M, w: 210, head: 'Setting' },
    { x: 250, w: 95, head: 'X' },
    { x: 345, w: 105, head: 'W' },
    { x: 450, w: 95, head: 'b' },
    { x: 545, w: 235, head: 'Operation' },
    { x: 780, w: 140, head: 'Output' },
  ];
  const HEAD_Y = 90, HEAD_H = 34, ROW_H = 66;
  const ROWS = [
    { a: 'Single neuron,', b: 'single sample', cells: ['(n,)', '(n,)', 'scalar', 'np.dot(W, X) + b', 'scalar'] },
    { a: 'Layer of m,', b: 'single sample', cells: ['(n,)', '(m, n)', '(m,)', 'np.dot(W, X) + b', '(m,)'] },
    { a: 'Layer of m,', b: 'batch of N', cells: ['(N, n)', '(m, n)', '(m,)', 'np.dot(X, W.T) + b', '(N, m)'], mark: true },
  ];

  els.push(rect(M, HEAD_Y, 880, HEAD_H,
    { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2, roundness: null }));
  COLS.forEach((c) => els.push(text(c.x + 12, HEAD_Y + 9, c.head, { size: 12, stroke: T.inkMuted })));

  ROWS.forEach((r, i) => {
    const y = HEAD_Y + HEAD_H + i * ROW_H;
    // The tint is carried by element opacity, which wraps fill and stroke
    // together, so the row reads as a wash and the text on it stays full strength.
    els.push(rect(M, y, 880, ROW_H, {
      stroke: T.border, fill: r.mark ? T.accent : T.surface,
      strokeWidth: 1.2, opacity: r.mark ? 18 : 100, roundness: null,
    }));
    // The marked row is called out by a spine, not by recolouring its text.
    // Accent ink on an accent wash measures under 3:1 in both themes, and the
    // two cells worth emphasising are the two that most need to stay readable.
    if (r.mark) {
      els.push(line([[M, y], [M, y + ROW_H]], { stroke: T.accent, strokeWidth: 4, roughness: 0.4 }));
    }
    els.push(text(COLS[0].x + 12, y + 16, fit(r.a, 13, COLS[0].w - 24, 'setting line 1'), { size: 13, stroke: T.ink }));
    els.push(text(COLS[0].x + 12, y + 36, fit(r.b, 13, COLS[0].w - 24, 'setting line 2'), { size: 13, stroke: T.ink }));
    r.cells.forEach((cell, j) => {
      const c = COLS[j + 1];
      const isOp = j === 3;
      els.push(text(c.x + 12, y + 24, fit(cell, isOp ? 12 : 13, c.w - 20, `cell ${c.head}`), {
        size: isOp ? 12 : 13, family: 3, stroke: T.ink,
      }));
    });
  });

  // Column rules drawn once through the whole table at low roughness: six
  // separate cell rectangles per row would read as a sketch of a table rather
  // than as a table.
  const BOT = HEAD_Y + HEAD_H + ROWS.length * ROW_H;
  COLS.slice(1).forEach((c) => {
    els.push(line([[c.x, HEAD_Y], [c.x, BOT]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  });

  els.push(text(M, 356, 'Row three is the only one that transposes, and the transpose changes the call signature, not the arithmetic.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 378, 'Print the shape after every operation. Surprises here account for most of the shape-mismatch errors in the series.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Shape diary: how the arrays change from a single neuron to a batched layer',
    desc: 'A hand-drawn three-row table tracking the shapes of the input X, the weight matrix W, the '
      + 'bias b, the operation, and the output. Row one covers a single neuron on a single sample: X '
      + 'of shape n, W of shape n, a scalar bias, np.dot of W and X plus b, and a scalar out. Row two '
      + 'covers a layer of m neurons on a single sample: X of shape n, W of shape m by n, a bias of '
      + 'shape m, the same call, and an output of shape m. Row three, tinted to mark it out, covers a '
      + 'layer of m neurons on a batch of N samples: X of shape N by n, W of shape m by n used '
      + 'transposed, a bias of shape m, np.dot of X and W transposed plus b, and an output of shape N '
      + 'by m. Captions note that row three is the only one that transposes, that the transpose '
      + 'changes the call signature rather than the arithmetic, and that printing shapes after every '
      + 'operation catches most shape-mismatch errors.',
  };
}

// ---------------------------------------------------------------- diagram 5
// The batched pass as a shape equation on top, and the same pass as arithmetic
// underneath. The grids carry no numbers on purpose: the top half is about
// shapes flowing and the band below is where the values are checked, so neither
// half has to do both jobs.
function batchBroadcasting() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('One call, three samples, three neurons',
    'Transposing W realigns the inner dimensions; broadcasting then puts each neuron’s bias in its own column.')];

  const CW = 34, CH = 30, MID = 193;

  // A grid of empty cells: one outer rectangle plus interior rules, rather than
  // rows x cols separate rectangles, which at this size reads as noise.
  const grid = (x, cols, rows, o) => {
    const w = cols * CW, h = rows * CH, y = MID - h / 2;
    els.push(rect(x, y, w, h, { stroke: o.c, fill: T.surface, strokeWidth: 1.4, roundness: null }));
    for (let i = 1; i < cols; i++) {
      els.push(line([[x + i * CW, y], [x + i * CW, y + h]],
        { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
    }
    for (let j = 1; j < rows; j++) {
      els.push(line([[x, y + j * CH], [x + w, y + j * CH]],
        { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
    }
    els.push(text(x, 106, fit(o.label, 13, w + 24, 'grid label'),
      { size: 13, stroke: o.c, align: 'center', width: w }));
    els.push(text(x, 262, o.shape, { size: 12, stroke: T.ink, family: 3, align: 'center', width: w }));
  };

  const op = (x, s) => els.push(text(x - 12, 182, s, { size: 20, stroke: T.inkMuted, align: 'center', width: 24 }));

  grid(144, 4, 3, { c: T.primary, label: 'inputs', shape: '(3, 4)' });
  op(296, '×');
  grid(312, 3, 4, { c: T.primary, label: 'Wᵀ', shape: '(4, 3)' });
  op(430, '=');
  grid(446, 3, 3, { c: T.inkMuted, label: 'raw', shape: '(3, 3)' });
  op(564, '+');
  grid(580, 3, 1, { c: T.success, label: 'biases', shape: '(3,)' });
  op(698, '=');
  grid(714, 3, 3, { c: T.accent, label: 'outputs', shape: '(3, 3)' });

  els.push(text(144, 284, 'inner dimensions 4 and 4 agree only after the transpose',
    { size: 11, stroke: T.inkSubtle }));

  // The same pass in numbers. Every value is the post's printed output, with
  // the raw column back-computed by subtracting the bias, so the band can be
  // checked against the code block rather than taken on trust.
  els.push(rect(M, 312, 880, 106, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(56, 322, 'Broadcasting, written out · one bias vector, added to all three rows',
    { size: 12, stroke: T.ink }));
  const NUMS = [
    'sample 1   [  2.80,  -1.790,   1.885 ]  +  [ 2, 3, 0.5 ]  =  [ 4.80,   1.210,  2.385 ]',
    'sample 2   [  6.90,  -4.810,  -0.300 ]  +  [ 2, 3, 0.5 ]  =  [ 8.90,  -1.810,  0.200 ]',
    'sample 3   [ -0.59,  -1.949,  -0.474 ]  +  [ 2, 3, 0.5 ]  =  [ 1.41,   1.051,  0.026 ]',
  ];
  NUMS.forEach((r, i) => els.push(text(56, 348 + i * 22, fit(r, 11, 848 * 0.94, `numbers row ${i + 1}`),
    { size: 11, stroke: T.inkMuted, family: 3 })));

  els.push(text(M, 438, 'Each row of the result is one sample’s three activations; each column belongs to one neuron.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 460, 'The bias vector has three entries and the result has three columns, which is why one vector covers every row.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'Batched forward pass: three samples through three neurons in one call',
    desc: 'A hand-drawn left-to-right shape equation. A three-by-four grid of inputs is multiplied by '
      + 'a four-by-three grid of transposed weights, giving a three-by-three raw result; a '
      + 'three-element bias vector is then added, giving the three-by-three output. Every grid '
      + 'carries its shape underneath, and a note points out that the inner dimensions of four and '
      + 'four agree only after the transpose. A band below writes the same pass out in numbers, one '
      + 'line per sample: the raw activations plus the bias vector two, three, nought-point-five, '
      + 'giving the printed outputs 4.8, 1.21, 2.385 for the first sample, 8.9, minus 1.81, 0.2 for '
      + 'the second, and 1.41, 1.051, 0.026 for the third. Captions note that each row of the result '
      + 'is one sample’s three activations and each column belongs to one neuron, and that a '
      + 'three-entry bias vector covers every row because the result has three columns.',
  };
}

// ---------------------------------------------------------------- diagram 6
// New. Section 11 makes the structural claim of the whole series and the post
// had no figure for it. The core sits at the same visual weight as the three
// additions, because the claim is that the core is not the small part.
function whatGetsAdded() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('What the rest of the series adds to this one formula',
    'Three additions, and then detail. The forward pass built in Part 01 is never replaced.')];

  const TOP = 104, CH = 152;

  // No spine on the core card: its whole border is already primary at 1.8, and
  // a spine 5px inside that reads as one thick smear rather than two edges.
  els.push(...card(M, TOP, 256, CH, { stroke: T.primary, strokeWidth: 1.8 }));
  els.push(text(M + 20, TOP + 14, 'The core · Part 01', { size: 12, stroke: T.primary }));
  els.push(text(M + 20, TOP + 42, fit('output = Σ wᵢxᵢ + b', 16, 216, 'core formula'), { size: 16, stroke: T.ink }));
  els.push(rule(M + 20, M + 236, TOP + 86));
  els.push(text(M + 20, TOP + 98, 'Every learnable parameter', { size: 11, stroke: T.inkMuted }));
  els.push(text(M + 20, TOP + 116, 'is a weight or a bias.', { size: 11, stroke: T.inkMuted }));

  els.push(arrow([[300, TOP + CH / 2], [334, TOP + CH / 2]],
    { stroke: T.inkMuted, strokeWidth: 1.4, roughness: 0.5 }));

  const ADDS = [
    {
      c: T.accent, head: 'Activation functions\n(ReLU, Softmax)',
      why: 'Without them, stacking\nlayers collapses back to\na single linear layer.',
      when: 'arrives in Part 06',
    },
    {
      c: T.warn, head: 'Loss functions\n(cross-entropy)',
      why: 'A way to score the\npredicted outputs\nagainst the truth.',
      when: 'arrives in Part 08',
    },
    {
      c: T.success, head: 'Backpropagation\n+ an optimiser',
      why: 'A way to adjust W and b\nso the score the loss\nreports goes down.',
      when: 'arrives in Parts 09–27',
    },
  ];
  ADDS.forEach((a, i) => {
    const x = 340 + i * 196, CWD = 188, inner = CWD - 30;
    els.push(...card(x, TOP, CWD, CH, { spine: a.c }));
    els.push(text(x + 18, TOP + 14, fit(a.head, 13, inner, `add ${i + 1} head`), { size: 13, stroke: T.ink }));
    els.push(rule(x + 18, x + CWD - 18, TOP + 58));
    els.push(text(x + 18, TOP + 68, fit(a.why, 11, inner, `add ${i + 1} body`), { size: 11, stroke: T.inkMuted }));
    els.push(text(x + 18, TOP + 124, fit(a.when, 11, inner, `add ${i + 1} when`), { size: 11, stroke: a.c }));
  });

  els.push(rect(M, 286, 880, 90, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(56, 298, 'With all three in place', { size: 12, stroke: T.ink }));
  els.push(text(56, 322, `the same forward pass that printed ${OUT} becomes the inner loop of a network`,
    { size: 12, stroke: T.inkMuted }));
  els.push(text(56, 344, 'that classifies spirals, recognises handwritten digits, or sorts sentiment.',
    { size: 12, stroke: T.inkMuted }));

  els.push(text(M, 404, 'Everything else across the remaining posts is detail on these three.',
    { size: 12, stroke: T.inkSubtle }));

  return {
    W, H, els,
    title: 'What the rest of the series adds to the Part 01 formula',
    desc: 'A hand-drawn accretion diagram. On the left, a card at full weight holds the Part 01 core: '
      + 'output equals the sum of w times x plus b, with the note that every learnable parameter is '
      + 'either a weight or a bias. An arrow leads right to three cards of equal size. The first '
      + 'adds activation functions, ReLU and Softmax, arriving in Part 06, because without them '
      + 'stacking layers collapses back to a single linear layer. The second adds loss functions, '
      + 'cross-entropy, arriving in Part 08, as a way to score predicted outputs against the truth. '
      + 'The third adds backpropagation and an optimiser, arriving across Parts 09 to 27, as a way to '
      + 'adjust the weights and biases so the loss goes down. A band beneath states that with all '
      + 'three in place the same forward pass that printed 4.8, 1.21, 2.385 becomes the inner loop of '
      + 'a network that classifies spirals, recognises handwritten digits, or sorts sentiment, and a '
      + 'footer notes that everything else in the series is detail on these three.',
  };
}

// ---------------------------------------------------------------------------
module.exports = {
  neuronAnatomy, layerAsStackedNeurons, threeImplementations,
  shapeDiary, batchBroadcasting, whatGetsAdded,
};

if (require.main === module) {
  emit('01-neuron-anatomy', neuronAnatomy());
  emit('02-layer-as-stacked-neurons', layerAsStackedNeurons());
  emit('03-three-implementations', threeImplementations());
  emit('04-shape-diary', shapeDiary());
  emit('05-batch-broadcasting', batchBroadcasting());
  emit('06-what-gets-added', whatGetsAdded());
}
