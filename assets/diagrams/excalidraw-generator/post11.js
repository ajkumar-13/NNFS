// The post 11 diagrams, hand-drawn.
//
// Scenes 01 and 02 mirror the clean vector figures one directory up. Scene 03 is
// new: sections 4 and 7 give a mechanical three-step procedure and work it on a
// polynomial, and both existing figures are about structure rather than about
// how the rule is actually applied.
//
//   01-chain-rule           960 x 460
//   02-chain-in-a-network   960 x 480
//   03-three-step-pattern   960 x 470   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// A box on a flow row, with an optional second line.
function node(els, x, y, w, h, label, sub, o = {}) {
  els.push(...card(x, y, w, h, { stroke: o.c ?? T.ink, spine: o.spine }));
  els.push(text(x, y + (sub ? 14 : (h - (o.size ?? 16)) / 2 - 2), label,
    { size: o.size ?? 16, family: o.family ?? 3, stroke: o.c ?? T.ink, align: 'center', width: w }));
  if (sub) {
    els.push(text(x, y + 42, fit(sub, 10, w - 20, 'node sub'),
      { size: 10, stroke: T.inkMuted, align: 'center', width: w }));
  }
  return { end: x + w, cx: x + w / 2 };
}

// ---------------------------------------------------------------- diagram 1
// The forward row and the backward row sit in the same columns, so each local
// derivative is directly under the box that owns it. That vertical alignment is
// the argument: a factor of the product belongs to exactly one function.
function chainRule() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('The chain rule',
    'Two functions in a row. The derivative through both is the product of the two local slopes.')];

  const FY = 108, FH = 66;
  node(els, 50, FY, 70, FH, 'x', null, { c: T.primary });
  node(els, 146, FY, 110, FH, 'g', null, { c: T.accent });
  node(els, 282, FY, 70, FH, 'z', null, { c: T.inkMuted });
  node(els, 378, FY, 110, FH, 'f', null, { c: T.accent });
  node(els, 514, FY, 70, FH, 'y', null, { c: T.success });
  [[120, 146], [256, 282], [352, 378], [488, 514]].forEach(([a, b]) => {
    els.push(arrow([[a + 2, FY + FH / 2], [b - 2, FY + FH / 2]],
      { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
  });
  els.push(text(50, FY - 22, 'forward', { size: 10, stroke: T.inkSubtle }));
  els.push(text(146, 190, 'z = g(x)', { size: 11, family: 3, stroke: T.inkSubtle, align: 'center', width: 110 }));
  els.push(text(378, 190, 'y = f(z)', { size: 11, family: 3, stroke: T.inkSubtle, align: 'center', width: 110 }));

  // Backward: the same columns, one local derivative each.
  const BY = 232;
  els.push(...card(146, BY, 110, 58, { spine: T.accent }));
  els.push(text(146, BY + 18, 'dz/dx', { size: 15, family: 3, stroke: T.ink, align: 'center', width: 110 }));
  els.push(...card(378, BY, 110, 58, { spine: T.accent }));
  els.push(text(378, BY + 18, 'dy/dz', { size: 15, family: 3, stroke: T.ink, align: 'center', width: 110 }));
  els.push(arrow([[372, BY + 29], [262, BY + 29]], { stroke: T.primary, strokeWidth: 1.4, roughness: 0.4 }));
  els.push(text(262, BY - 22, 'backward', { size: 10, stroke: T.primary, align: 'center', width: 110 }));

  els.push(text(40, 314, 'dy/dx  =  dy/dz  ·  dz/dx',
    { size: 16, family: 3, stroke: T.ink, align: 'center', width: 560 }));

  els.push(...card(620, 108, 300, 240, { spine: T.success }));
  els.push(text(642, 120, 'Litres per hour', { size: 15, stroke: T.success }));
  els.push(rule(634, 906, 148));
  const ROWS = [
    ['x', 'time, in hours', ''],
    ['z = 60x', 'distance, in km', 'dz/dx = 60'],
    ['y = z / 30', 'fuel, in litres', 'dy/dz = 1/30'],
  ];
  ROWS.forEach((r, i) => {
    const y = 160 + i * 46;
    els.push(text(642, y, r[0], { size: 12, family: 3, stroke: T.ink }));
    els.push(text(642, y + 18, r[1], { size: 10, stroke: T.inkSubtle }));
    if (r[2]) els.push(text(786, y + 4, r[2], { size: 11, family: 3, stroke: T.accent }));
  });
  els.push(rule(634, 906, 302));
  els.push(text(642, 312, 'dy/dx = (1/30) × 60 = 2', { size: 13, family: 3, stroke: T.success }));

  els.push(text(M, 380, 'y never mentions x. The chain rule supplies the link anyway, by multiplying "fuel per kilometre" by "kilometres per hour".',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 404, 'Each slope is read at its own point: f′ at the value z that g produced, and g′ at the original x.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The chain rule: derivatives compose by multiplication',
    desc: 'A hand-drawn flow diagram. The forward row runs left to right: x enters the box g '
      + 'producing z equals g of x, which enters the box f producing y equals f of z. Directly '
      + 'beneath each box sits its local derivative, dz by dx under g and dy by dz under f, with a '
      + 'right-to-left arrow labelled backward running between them, and the statement that dy by dx '
      + 'equals dy by dz times dz by dx. A side panel works a numerical example: x is time in hours, '
      + 'z equals sixty x is distance in kilometres so dz by dx is sixty, and y equals z over thirty '
      + 'is fuel in litres so dy by dz is one thirtieth; the product gives two litres per hour. A '
      + 'footer notes that y never mentions x, and that the chain rule supplies the link by '
      + 'multiplying fuel per kilometre by kilometres per hour, with each slope read at its own '
      + 'point.',
  };
}

// ---------------------------------------------------------------- diagram 2
// Four factors, each sitting under the function that owns it. The backward row
// is deliberately not re-ordered left to right: keeping the columns aligned is
// what shows that reading the chain rule is reading the architecture backwards.
function chainInANetwork() {
  resetSeq();
  const W = 960, H = 480;

  const els = [...heading('The same rule, unrolled across a network',
    'One factor per function. Every factor is something a single class can work out on its own.')];

  const FY = 100, FH = 76;
  const NODES = [
    { x: 40, w: 70, label: 'x', sub: null, c: T.primary },
    { x: 136, w: 160, label: 'Layer 1', sub: 'W₁, b₁', c: T.accent },
    { x: 322, w: 120, label: 'ReLU', sub: null, c: T.success },
    { x: 468, w: 160, label: 'Layer 2', sub: 'W₂, b₂', c: T.accent },
    { x: 654, w: 170, label: 'Softmax + CE', sub: 'against y', c: T.primary },
    { x: 850, w: 70, label: 'L', sub: null, c: T.alert },
  ];
  NODES.forEach((n, i) => {
    node(els, n.x, FY, n.w, FH, n.label, n.sub, { c: n.c, family: n.sub ? 2 : 3, size: n.sub ? 14 : 17 });
    if (i < NODES.length - 1) {
      const b = NODES[i + 1];
      els.push(arrow([[n.x + n.w + 2, FY + FH / 2], [b.x - 2, FY + FH / 2]],
        { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
    }
  });
  [[309, 'z₁'], [455, 'a₁'], [641, 'z₂']].forEach(([cx, lab]) => {
    els.push(text(cx - 30, 82, lab, { size: 11, family: 3, stroke: T.inkSubtle, align: 'center', width: 60 }));
  });

  const BY = 244, BH = 62;
  const FACTORS = [
    { x: 141, w: 150, t: '∂z₁/∂W₁', from: 'Layer_Dense' },
    { x: 306, w: 150, t: '∂a₁/∂z₁', from: 'Activation_ReLU' },
    { x: 473, w: 150, t: '∂z₂/∂a₁', from: 'Layer_Dense' },
    { x: 664, w: 150, t: '∂L/∂z₂', from: 'Loss (+ softmax)' },
  ];
  FACTORS.forEach((f) => {
    els.push(...card(f.x, BY, f.w, BH, { spine: T.primary }));
    els.push(text(f.x, BY + 12, f.t, { size: 15, family: 3, stroke: T.ink, align: 'center', width: f.w }));
    els.push(text(f.x, BY + 38, fit(f.from, 9, f.w - 20, 'owner'),
      { size: 9, stroke: T.inkSubtle, align: 'center', width: f.w }));
    els.push(line([[f.x + f.w / 2, FY + FH + 4], [f.x + f.w / 2, BY - 4]],
      { stroke: T.border, strokeWidth: 1, strokeStyle: 'dotted', roughness: 0.3 }));
  });
  [298, 464, 643].forEach((cx) => els.push(text(cx - 10, BY + 16, '·',
    { size: 16, stroke: T.inkMuted, align: 'center', width: 20 })));

  els.push(arrow([[820, 218], [150, 218]], { stroke: T.primary, strokeWidth: 1.4, roughness: 0.3 }));
  els.push(text(400, 194, 'the backward pass walks the chain in reverse',
    { size: 11, stroke: T.primary, align: 'center', width: 200 }));

  els.push(text(M, 334, '∂L/∂W₁  =  ∂L/∂z₂  ·  ∂z₂/∂a₁  ·  ∂a₁/∂z₁  ·  ∂z₁/∂W₁',
    { size: 15, family: 3, stroke: T.ink, align: 'center', width: 880 }));

  els.push(rect(M, 372, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 384, 'Every class needs to know only its own local derivative. Gluing them together is the chain rule’s job.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 408, 'Parts 12 to 21 derive one factor at a time. This post supplies only the rule that multiplies them.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'The chain rule applied to a two-layer neural network',
    desc: 'A hand-drawn diagram in two rows. The upper row is the forward pass: x enters Layer 1 '
      + 'holding W1 and b1 producing z1, then ReLU producing a1, then Layer 2 holding W2 and b2 '
      + 'producing z2, then softmax with cross-entropy against the labels, producing the scalar loss '
      + 'L. The lower row holds four local-derivative cards, each connected by a dotted line to the '
      + 'function directly above it that owns it: partial z1 by partial W1 from Layer_Dense, partial '
      + 'a1 by partial z1 from Activation_ReLU, partial z2 by partial a1 from Layer_Dense again, and '
      + 'partial L by partial z2 from the loss with softmax folded in. A long right-to-left arrow '
      + 'between the rows is labelled as the backward pass walking the chain in reverse. The full '
      + 'product is written out beneath. A band notes that every class needs to know only its own '
      + 'local derivative, that gluing them together is the chain rule’s job, and that Parts 12 to '
      + '21 derive one factor at a time.',
  };
}

// ---------------------------------------------------------------- diagram 3
// New. Sections 4 and 7 give a mechanical procedure and work it on a
// polynomial. Both existing figures draw the shape of the rule; neither shows
// it being applied, which is the part a reader has to reproduce themselves.
function threeStepPattern() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Applying the rule is mechanical',
    'Three steps, worked on 3(2x²)⁵. None of the individual derivatives is hard; the rule strings them together.')];

  const STEPS = [
    {
      c: T.primary, n: '1', head: 'Split it',
      what: 'Name the outer function and\nthe inner one it is applied to.',
      work: 'inner   g(x) = 2x²\nouter   f(z) = 3z⁵',
      note: 'Step 1 is just reading the expression.',
    },
    {
      c: T.accent, n: '2', head: 'Differentiate each',
      what: 'One ordinary derivative per\nfunction, ignoring the other.',
      work: 'g′(x) = 4x\nf′(z) = 15z⁴',
      note: 'Step 2 is where the nine\nremaining posts do their work.',
    },
    {
      c: T.success, n: '3', head: 'Multiply',
      what: 'Evaluate f′ at what g produced,\nthen multiply by g′.',
      work: '15(2x²)⁴ · 4x\n= 15 · 16x⁸ · 4x\n= 960x⁹',
      note: 'Step 3 is arithmetic.',
    },
  ];

  const CWD = 280, TOP = 94, CHT = 254;
  STEPS.forEach((s, i) => {
    const x = M + i * 300, inner = CWD - 40;
    els.push(...card(x, TOP, CWD, CHT, { spine: s.c }));
    els.push(text(x + 20, TOP + 12, s.n, { size: 20, stroke: s.c }));
    els.push(text(x + 46, TOP + 16, fit(s.head, 14, inner - 30, 'step head'), { size: 14, stroke: s.c }));
    els.push(text(x + 20, TOP + 50, fit(s.what, 10, inner, 'step what'), { size: 10, stroke: T.inkSubtle }));
    els.push(rule(x + 16, x + CWD - 16, TOP + 92));
    els.push(text(x + 20, TOP + 104, fit(s.work, 13, inner * 0.94, 'step work'),
      { size: 13, family: 3, stroke: T.ink }));
    els.push(rule(x + 16, x + CWD - 16, TOP + 196));
    els.push(text(x + 20, TOP + 206, fit(s.note, 10, inner, 'step note'), { size: 10, stroke: T.inkMuted }));
    if (i < 2) {
      els.push(arrow([[x + CWD + 2, TOP + CHT / 2], [x + CWD + 18, TOP + CHT / 2]],
        { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
    }
  });

  els.push(rect(M, 380, 880, 72, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(56, 392, 'The same three steps on the network:', { size: 12, stroke: T.ink }));
  els.push(text(56, 414, 'Step 1  read the architecture      Step 2  derive each layer’s local derivative      Step 3  multiply the factors',
    { size: 11, stroke: T.inkMuted }));
  els.push(text(56, 434, 'Only Step 2 is actual work, and it is the work Parts 12 to 21 do, one class at a time.',
    { size: 11, stroke: T.inkSubtle }));

  return {
    W, H, els,
    title: 'The chain rule as a three-step procedure, worked on a polynomial',
    desc: 'Three hand-drawn step cards working the derivative of three times two x squared, all to '
      + 'the fifth. Step one, split it: name the inner function g of x equals two x squared and the '
      + 'outer function f of z equals three z to the fifth, which is just reading the expression. '
      + 'Step two, differentiate each locally: g prime of x is four x and f prime of z is fifteen z '
      + 'to the fourth, one ordinary derivative per function, and this is where the remaining posts '
      + 'do their work. Step three, multiply: evaluate f prime at what g produced and multiply by g '
      + 'prime, giving fifteen times two x squared to the fourth times four x, which expands to '
      + 'fifteen times sixteen x to the eighth times four x, equalling nine hundred and sixty x to '
      + 'the ninth. A band maps the same three steps onto the network: read the architecture, derive '
      + 'each layer’s local derivative, multiply the factors, noting that only the second is actual '
      + 'work and it is what Parts 12 to 21 do one class at a time.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { chainRule, chainInANetwork, threeStepPattern };

if (require.main === module) {
  emit('01-chain-rule', chainRule());
  emit('02-chain-in-a-network', chainInANetwork());
  emit('03-three-step-pattern', threeStepPattern());
}
