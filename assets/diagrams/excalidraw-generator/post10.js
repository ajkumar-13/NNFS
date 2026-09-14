// The post 10 diagrams, hand-drawn.
//
// Scenes 01 and 02 mirror the clean vector figures one directory up. Scene 03 is
// new: section 2.3 ends by turning the three readings of a slope into three
// update decisions, which is the whole reason the post exists, and the step from
// "this is the slope" to "so the weight moves this way" had no figure.
//
//   01-derivative-as-slope  960 x 500
//   02-partial-to-gradient  960 x 460
//   03-slope-to-update      960 x 460   (new)

const { T, rect, circle, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// Three tangents on one parabola. Drawing them on the same curve is what lets
// the slopes be compared: on three separate axes they would be three unrelated
// straight lines.
function derivativeAsSlope() {
  resetSeq();
  const W = 960, H = 500;

  const els = [...heading('A derivative is the slope of the tangent',
    'One curve, three points, three slopes. The number says how much f moves per unit of nudge to x.')];

  const CX = 330, Y0 = 380, XS = 83, YS = 260 / 9;
  const px = (x) => CX + x * XS;
  const py = (y) => Y0 - y * YS;

  els.push(line([[px(-3.1), Y0], [px(3.1), Y0]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(line([[CX, Y0 + 12], [CX, py(9.4)]], { stroke: T.border, strokeWidth: 1, roughness: 0.3 }));
  els.push(text(px(2.9), Y0 + 12, 'x', { size: 12, stroke: T.inkSubtle }));
  els.push(text(CX + 10, py(9.2), 'f(x) = x²', { size: 12, stroke: T.inkSubtle }));
  [-2, -1, 1, 2].forEach((x) => {
    els.push(line([[px(x), Y0 - 4], [px(x), Y0 + 4]], { stroke: T.border, strokeWidth: 1, roughness: 0.2 }));
    els.push(text(px(x) - 16, Y0 + 10, String(x), { size: 10, stroke: T.inkSubtle, align: 'center', width: 32 }));
  });

  const curve = [];
  for (let i = 0; i <= 48; i++) {
    const x = -3 + (i / 48) * 6;
    curve.push([px(x), py(x * x)]);
  }
  els.push(line(curve, { stroke: T.primary, strokeWidth: 2.4, roughness: 0.3 }));

  const TANGENTS = [
    { a: -1.5, m: -3, c: T.alert, lab: 'slope −3', dx: -66, dy: -34 },
    { a: 0.5, m: 1, c: T.warn, lab: 'slope 1', dx: 12, dy: 14 },
    { a: 2.0, m: 4, c: T.success, lab: 'slope 4', dx: 10, dy: -26 },
  ];
  TANGENTS.forEach((t) => {
    const f = (x) => t.a * t.a + t.m * (x - t.a);
    els.push(line([[px(t.a - 0.95), py(f(t.a - 0.95))], [px(t.a + 0.95), py(f(t.a + 0.95))]],
      { stroke: t.c, strokeWidth: 1.8, roughness: 0.3 }));
    els.push(circle(px(t.a), py(t.a * t.a), 5, { stroke: t.c, fill: T.surface, strokeWidth: 1.6 }));
    els.push(text(px(t.a) + t.dx, py(t.a * t.a) + t.dy, t.lab, { size: 11, stroke: t.c }));
  });

  els.push(...card(620, 110, 300, 280, { spine: T.primary }));
  els.push(text(642, 122, 'The power rule', { size: 15, stroke: T.primary }));
  els.push(text(642, 148, 'd/dx [a xⁿ]  =  n · a · xⁿ⁻¹', { size: 12, family: 3, stroke: T.ink }));
  els.push(rule(634, 906, 178));
  els.push(text(642, 188, 'Three readings of the number', { size: 12, stroke: T.ink }));
  const READ = [
    { c: T.success, a: 'large positive', b: 'x matters, and raising it raises f' },
    { c: T.warn, a: 'near zero', b: 'x barely matters here at all' },
    { c: T.alert, a: 'negative', b: 'x matters, and raising it lowers f' },
  ];
  READ.forEach((r, i) => {
    const y = 214 + i * 52;
    els.push(line([[642, y + 2], [642, y + 30]], { stroke: r.c, strokeWidth: 3.5, roughness: 0.5 }));
    els.push(text(656, y, r.a, { size: 12, stroke: r.c }));
    els.push(text(656, y + 20, fit(r.b, 10, 240, 'reading note'), { size: 10, stroke: T.inkMuted }));
  });

  els.push(text(M, 420, 'The slope is a rate, not a step. A finite nudge of Δx moves f by roughly f′(x) · Δx, which is why the learning rate exists.',
    { size: 12, stroke: T.inkMuted }));
  els.push(text(M, 442, 'When f is the loss and x is a weight, these three readings become three different things to do with that weight.',
    { size: 12, stroke: T.inkMuted }));

  return {
    W, H, els,
    title: 'The derivative is the slope of the tangent line',
    desc: 'A hand-drawn plot of f of x equals x squared with three tangent lines. At x equals minus '
      + 'one point five the tangent has slope minus three and falls steeply; at x equals nought '
      + 'point five it has slope one and rises gently; at x equals two it has slope four and rises '
      + 'steeply. Each point of contact is marked with a dot and each tangent is labelled with its '
      + 'slope. A side panel gives the power rule, that the derivative of a x to the n is n times a '
      + 'times x to the n minus one, and lists three readings of the resulting number: large '
      + 'positive means x matters and raising it raises f, near zero means x barely matters here, '
      + 'and negative means x matters and raising it lowers f. A footer notes that the slope is a '
      + 'rate rather than a step, so a finite nudge moves f by roughly the slope times the nudge, '
      + 'which is why the learning rate exists.',
  };
}

// ---------------------------------------------------------------- diagram 2
// Three branches that differ only in which variable is left free. Drawing them
// identically apart from that one word is the argument: a partial derivative is
// an ordinary derivative with the other variables held still.
function partialToGradient() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('Partial derivatives, assembled into a gradient',
    'One derivative per variable, each taken with the others held still, stacked into a single vector.')];

  els.push(...card(50, 186, 190, 88, { stroke: T.primary, strokeWidth: 1.6 }));
  els.push(text(50, 206, 'f(x, y, z)', { size: 18, family: 3, stroke: T.primary, align: 'center', width: 190 }));
  els.push(text(50, 238, 'one function, three inputs',
    { size: 10, stroke: T.inkSubtle, align: 'center', width: 190 }));

  const BRANCH = [
    { y: 106, v: 'x', frozen: 'hold y and z still', c: T.accent },
    { y: 196, v: 'y', frozen: 'hold x and z still', c: T.accent },
    { y: 286, v: 'z', frozen: 'hold x and y still', c: T.accent },
  ];
  BRANCH.forEach((b) => {
    els.push(...card(330, b.y, 280, 74, { spine: b.c }));
    els.push(text(352, b.y + 12, `∂f / ∂${b.v}`, { size: 17, family: 3, stroke: T.ink }));
    els.push(text(352, b.y + 44, b.frozen, { size: 10, stroke: T.inkSubtle }));
    els.push(line([[244, 230], [292, 230], [292, b.y + 37], [326, b.y + 37]],
      { stroke: T.inkMuted, strokeWidth: 1.2, roughness: 0.4 }));
    els.push(arrow([[614, b.y + 37], [676, b.y + 37]],
      { stroke: T.inkMuted, strokeWidth: 1.3, roughness: 0.4 }));
  });

  els.push(rect(682, 106, 150, 254, { stroke: T.success, fill: T.surface, strokeWidth: 1.6, roundness: null }));
  [1, 2].forEach((k) => els.push(line([[682, 106 + k * 84.67], [832, 106 + k * 84.67]],
    { stroke: T.border, strokeWidth: 1, roughness: 0.3 })));
  ['∂f / ∂x', '∂f / ∂y', '∂f / ∂z'].forEach((s, i) => {
    els.push(text(682, 106 + i * 84.67 + 34, s,
      { size: 14, family: 3, stroke: T.ink, align: 'center', width: 150 }));
  });
  els.push(text(682, 80, '∇f', { size: 20, stroke: T.success, align: 'center', width: 150 }));
  els.push(text(682, 368, 'the gradient', { size: 11, stroke: T.success, align: 'center', width: 150 }));

  els.push(rect(M, 396, 880, 50, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 410, '∇f points along the steepest ascent, so −∇f is the direction gradient descent moves in.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'From partial derivatives to gradients',
    desc: 'A hand-drawn flow diagram. On the left, a function f of x, y and z, described as one '
      + 'function with three inputs. Three branches lead from it, each computing one partial '
      + 'derivative: partial f by partial x while holding y and z still, partial f by partial y '
      + 'while holding x and z still, and partial f by partial z while holding x and y still. The '
      + 'three branches are drawn identically apart from which variable is left free. They feed into '
      + 'a single column vector on the right, labelled the gradient, holding the three partials '
      + 'stacked. A band beneath states that the gradient points along the steepest ascent, so its '
      + 'negative is the direction gradient descent moves in.',
  };
}

// ---------------------------------------------------------------- diagram 3
// New. Section 2.3 closes by turning the three readings of a slope into three
// update decisions, and that step — from "here is the slope" to "so this weight
// moves that way" — is the one the rest of the series depends on.
function slopeToUpdate() {
  resetSeq();
  const W = 960, H = 460;

  const els = [...heading('What a slope tells a weight to do',
    'The same loss curve and the same update rule at three points. The sign decides the direction; the size decides the distance.')];

  const CASES = [
    {
      x: M, c: T.alert, title: 'Slope is positive', w: 2.0, slope: 2.0, wNew: 1.8,
      reading: 'Raising w raises the loss.',
      verdict: 'The rule subtracts, so w falls.',
    },
    {
      x: 340, c: T.warn, title: 'Slope is near zero', w: 1.0, slope: 0.0, wNew: 1.0,
      reading: 'w barely affects the loss here.',
      verdict: 'The rule leaves it where it is.',
    },
    {
      x: 640, c: T.success, title: 'Slope is negative', w: 0.0, slope: -2.0, wNew: 0.2,
      reading: 'Raising w lowers the loss.',
      verdict: 'Subtracting a negative pushes w up.',
    },
  ];

  const CWD = 280, TOP = 90, CHT = 250;
  CASES.forEach((c) => {
    const inner = CWD - 40;
    els.push(...card(c.x, TOP, CWD, CHT, { spine: c.c }));
    els.push(text(c.x + 20, TOP + 12, c.title, { size: 14, stroke: c.c }));
    els.push(text(c.x + 20, TOP + 34, fit(c.reading, 10, inner, 'reading'), { size: 10, stroke: T.inkSubtle }));

    // L(w) = (w - 1)^2 over w in [-0.6, 2.6], the same curve in all three cards.
    const gx = c.x + 26, gw = 228, gy0 = TOP + 148;
    const pw = (w) => gx + ((w + 0.6) / 3.2) * gw;
    const pl = (L) => gy0 - 10 - (L / 2.6) * 70;
    els.push(line([[gx - 6, gy0], [gx + gw + 6, gy0]], { stroke: T.border, strokeWidth: 1, roughness: 0.25 }));
    const pts = [];
    for (let i = 0; i <= 36; i++) {
      const w = -0.6 + (i / 36) * 3.2;
      pts.push([pw(w), pl((w - 1) * (w - 1))]);
    }
    els.push(line(pts, { stroke: T.inkMuted, strokeWidth: 1.6, roughness: 0.3 }));

    const L = (c.w - 1) * (c.w - 1);
    const tan = (w) => L + c.slope * (w - c.w);
    els.push(line([[pw(c.w - 0.62), pl(tan(c.w - 0.62))], [pw(c.w + 0.62), pl(tan(c.w + 0.62))]],
      { stroke: c.c, strokeWidth: 2, roughness: 0.3 }));
    els.push(circle(pw(c.w), pl(L), 5, { stroke: c.c, fill: T.surface, strokeWidth: 1.6 }));
    els.push(text(pw(c.w) - 30, gy0 + 6, `w = ${c.w.toFixed(1)}`,
      { size: 10, family: 3, stroke: T.inkSubtle, align: 'center', width: 60 }));

    els.push(rule(c.x + 16, c.x + CWD - 16, TOP + 172));
    els.push(text(c.x + 20, TOP + 182, `∂L/∂w  =  ${c.slope > 0 ? '+' : ''}${c.slope.toFixed(1)}`,
      { size: 12, family: 3, stroke: c.c }));
    els.push(text(c.x + 20, TOP + 202,
      `w ← ${c.w.toFixed(1)} − 0.1 × (${c.slope.toFixed(1)}) = ${c.wNew.toFixed(1)}`,
      { size: 11, family: 3, stroke: T.ink }));
    els.push(text(c.x + 20, TOP + 226, fit(c.verdict, 10, inner, 'verdict'), { size: 10, stroke: T.inkMuted }));
  });

  els.push(rect(M, 370, 880, 64, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 382, 'All three moves are the same line of code: w ← w − α · ∂L/∂w, with α = 0.1.',
    { size: 13, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 406, 'Every one of them steps towards w = 1, where the loss is lowest, without anything in the rule knowing that is where it is going.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'How the sign and size of a derivative decide a weight update',
    desc: 'Three hand-drawn cards, each showing the same loss curve against a single weight w, with a '
      + 'tangent drawn at a different point. In the first the slope is positive two at w equals two, '
      + 'meaning raising w raises the loss, so the update rule subtracts and w falls to one point '
      + 'eight. In the second the slope is nought at w equals one, meaning w barely affects the loss '
      + 'there, so the rule leaves it where it is. In the third the slope is minus two at w equals '
      + 'nought, meaning raising w lowers the loss, so subtracting a negative pushes w up to nought '
      + 'point two. Each card spells out the arithmetic, w becomes w minus nought point one times '
      + 'the slope. A band notes that all three moves are the same line of code with a learning rate '
      + 'of nought point one, and that every one of them steps towards w equals one where the loss '
      + 'is lowest, without anything in the rule knowing that is where it is going.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { derivativeAsSlope, partialToGradient, slopeToUpdate };

if (require.main === module) {
  emit('01-derivative-as-slope', derivativeAsSlope());
  emit('02-partial-to-gradient', partialToGradient());
  emit('03-slope-to-update', slopeToUpdate());
}
