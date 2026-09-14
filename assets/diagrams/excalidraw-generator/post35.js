// The post 35 diagrams, hand-drawn.
//
// Scene 01 mirrors the clean vector figure one directory up. Scene 02 is new:
// section 1 closes the series by pointing out that the loop written in Part 21
// and the contract written in Part 23 do not change for any of the things on
// the reading list. The map shows what is out there; this shows why none of it
// requires starting again.
//
//   01-whats-next-map          960 x 540
//   02-the-skeleton-is-fixed   960 x 470   (new)

const { T, rect, text, line, arrow, resetSeq } = require('./lib');
const { M, heading, emit, fit, card, rule } = require('./scaffold');

// ---------------------------------------------------------------- diagram 1
// The "you are here" column is the same height as the three ahead of it. The
// series is not a preliminary to the real material; it is one column of it.
function whatsNextMap() {
  resetSeq();
  const W = 960, H = 540;

  const els = [...heading('The map from here',
    'Three directions out of a working MLP, sorted by what each one actually adds.')];

  const COLS = [
    {
      // A flat list, not glossed pairs: this column is an inventory.
      x: M, w: 200, c: T.success, name: 'You are here', sub: 'Posts 01 to 34',
      flat: ['Dense layers', 'ReLU · sigmoid · softmax', 'Cross-entropy · BCE · MSE',
        'SGD through Adam', 'L1 · L2 · dropout', 'Mini-batching · k-fold', 'He initialisation',
        'Four worked projects'],
    },
    {
      x: 262, w: 200, c: T.primary, name: 'New layer types', sub: 'a different forward pass',
      pairs: [['Convolution', 'images, audio, anything spatial'], ['Recurrent · LSTM · GRU', 'sequences and time series'],
        ['Attention · transformers', 'the modern default'], ['Batch norm · residuals', 'what makes depth trainable']],
    },
    {
      x: 484, w: 200, c: T.accent, name: 'New infrastructure', sub: 'a different training run',
      pairs: [['Cosine · warmup', 'schedules beyond 1/(1+dt)'], ['Mixed precision', 'fp16 and bf16 arithmetic'],
        ['Distributed training', 'across many devices'], ['Gradient checkpointing', 'memory traded for compute']],
    },
    {
      x: 706, w: 214, c: T.warn, name: 'New framings', sub: 'a different question',
      pairs: [['Self-supervised', 'labels from the data itself'], ['Transfer · fine-tuning', 'start from someone else’s'],
        ['Reinforcement learning', 'reward instead of labels'], ['Diffusion', 'generation, not prediction']],
    },
  ];

  COLS.forEach((c) => {
    const inner = c.w - 30;
    els.push(...card(c.x, 86, c.w, 344, { spine: c.c }));
    els.push(text(c.x + 16, 98, fit(c.name, 13, inner, 'col name'), { size: 13, stroke: c.c }));
    els.push(text(c.x + 16, 120, fit(c.sub, 9, inner, 'col sub'), { size: 9, stroke: T.inkSubtle }));
    els.push(rule(c.x + 14, c.x + c.w - 14, 142));

    // Both layouts fill the same 272px, so the four columns end level.
    if (c.flat) {
      c.flat.forEach((it, i) => els.push(text(c.x + 16, 158 + i * 34,
        fit(it, 11, inner, 'col item'), { size: 11, stroke: T.ink })));
    } else {
      c.pairs.forEach((p, i) => {
        const y = 158 + i * 68;
        els.push(text(c.x + 16, y, fit(p[0], 11, inner, 'pair head'), { size: 11, stroke: T.ink }));
        els.push(text(c.x + 16, y + 18, fit(p[1], 9, inner, 'pair gloss'), { size: 9, stroke: T.inkSubtle }));
      });
    }
  });

  els.push(rect(M, 456, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 468, 'The first column is the same height as the other three on purpose.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 492, 'What this series built is not a preliminary to the real material. It is one column of it, and the one the rest is written on top of.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'What to learn after this series, organised by what each addition contributes',
    desc: 'A hand-drawn four-column map. The first column, headed you are here and covering Posts 01 '
      + 'to 34, lists dense layers, ReLU with sigmoid and softmax, cross-entropy with BCE and MSE, '
      + 'SGD through Adam, L1 and L2 and dropout, mini-batching and k-fold, He initialisation, and '
      + 'four worked projects. The second, new layer types meaning a different forward pass, lists '
      + 'convolution for images and audio, recurrent layers with LSTM and GRU for sequences, '
      + 'attention and transformers as the modern default, and batch normalisation with residual '
      + 'connections. The third, new infrastructure meaning a different training run, lists cosine '
      + 'and warmup schedules, mixed precision, distributed training and gradient checkpointing. The '
      + 'fourth, new framings meaning a different question, lists self-supervised learning, transfer '
      + 'and fine-tuning, reinforcement learning and diffusion. All four columns are drawn at the '
      + 'same height, and a band notes that this is deliberate: what the series built is one column '
      + 'of the material rather than a preliminary to it.',
  };
}

// ---------------------------------------------------------------- diagram 2
// New. The closing claim of the series is that nothing on the reading list
// requires starting again: the loop from Part 21 and the contract from Part 23
// both hold. Putting the loop in the middle with everything else plugging into
// it is the shape of that claim.
function theSkeletonIsFixed() {
  resetSeq();
  const W = 960, H = 470;

  const els = [...heading('Everything ahead plugs into what you already wrote',
    'The loop from Part 21 and the optimiser contract from Part 23 do not change for any of it.')];

  els.push(...card(340, 96, 280, 262, { stroke: T.primary, strokeWidth: 1.8 }));
  els.push(text(340, 108, 'the training loop', { size: 12, stroke: T.primary, align: 'center', width: 280 }));
  els.push(text(340, 128, 'unchanged since Part 21', { size: 9, stroke: T.inkSubtle, align: 'center', width: 280 }));
  els.push(rule(356, 604, 150));
  const LOOP = [
    'for epoch in range(E):',
    '  for batch in batches:',
    '    forward(batch)',
    '    loss = criterion(...)',
    '    backward()',
    '    opt.pre_update_params()',
    '    opt.update_params(layer)',
    '    opt.post_update_params()',
  ];
  LOOP.forEach((l, i) => els.push(text(360, 162 + i * 20, fit(l, 10, 244 * 0.94, 'loop line'),
    { size: 10, family: 3, stroke: i >= 5 ? T.accent : T.ink })));
  els.push(text(340, 330, 'nothing here needs editing', { size: 9, stroke: T.inkSubtle, align: 'center', width: 280 }));

  const side = (x, o) => {
    els.push(...card(x, 96, 280, 262, { spine: o.c }));
    els.push(text(x + 18, 108, o.title, { size: 13, stroke: o.c }));
    els.push(text(x + 18, 130, o.rule, { size: 10, family: 3, stroke: T.ink }));
    els.push(rule(x + 14, x + 266, 154));
    els.push(text(x + 18, 164, 'already built', { size: 9, stroke: T.inkSubtle }));
    els.push(text(x + 18, 180, fit(o.built, 10, 244, 'built'), { size: 10, stroke: T.inkMuted }));
    els.push(rule(x + 14, x + 266, o.mid));
    els.push(text(x + 18, o.mid + 12, 'and everything after it', { size: 9, stroke: T.inkSubtle }));
    els.push(text(x + 18, o.mid + 28, fit(o.next, 10, 244, 'next'), { size: 10, stroke: o.c }));
    els.push(rule(x + 14, x + 266, 314));
    els.push(text(x + 18, 326, fit(o.close, 9, 244, 'close'), { size: 9, stroke: T.inkMuted }));
  };

  side(M, {
    c: T.success, title: 'Any layer', rule: '.forward(x)   .backward(dvalues)',
    built: 'Dense · ReLU · Sigmoid\nSoftmax · Dropout',
    mid: 226,
    next: 'Convolution · LSTM · GRU\nAttention · BatchNorm\nAnything with those two methods.',
    close: 'The loop calls those two by name\nand asks the layer nothing else.',
  });

  side(640, {
    c: T.accent, title: 'Any optimiser', rule: 'pre · update · post',
    built: 'SGD · decay · momentum\nAdaGrad · RMSProp · Adam',
    mid: 226,
    next: 'AdamW · Lion · Adafactor\nWarmup and cosine schedules\nAnything with those three hooks.',
    close: 'The loop calls those three in order\nand asks the optimiser nothing else.',
  });

  els.push(arrow([[322, 216], [336, 216]], { stroke: T.success, strokeWidth: 1.4, roughness: 0.4 }));
  els.push(arrow([[636, 216], [622, 216]], { stroke: T.accent, strokeWidth: 1.4, roughness: 0.4 }));

  els.push(rect(M, 382, 880, 62, { stroke: T.border, fill: T.neutral1, strokeWidth: 1.2 }));
  els.push(text(M, 394, 'A transformer block is a forward and a backward. AdamW is three hooks. Neither needs a line of the loop to move.',
    { size: 12, stroke: T.ink, align: 'center', width: 880 }));
  els.push(text(M, 418, 'That is what the twenty-one parameters of Part 09 grew into: not a finished model, but the frame everything else is written against.',
    { size: 11, stroke: T.inkMuted, align: 'center', width: 880 }));

  return {
    W, H, els,
    title: 'Why nothing on the reading list requires starting again',
    desc: 'A hand-drawn figure with the training loop in the centre, labelled unchanged since Part '
      + '21, listing the epoch and batch loops, the forward pass, the loss, the backward pass and '
      + 'the optimiser’s three hooks, with a note that nothing in it needs editing. A card on the '
      + 'left, headed any layer and giving the interface as a forward taking x and a backward taking '
      + 'dvalues, lists what is already built — Dense, ReLU, Sigmoid, Softmax and Dropout — and what '
      + 'comes after: convolution, LSTM, GRU, attention, batch normalisation, and anything with '
      + 'those two methods. A card on the right, headed any optimiser and giving the interface as '
      + 'pre, update and post, lists SGD through Adam as built and AdamW, Lion, Adafactor, warmup '
      + 'and cosine schedules as what follows, being anything with those three hooks. Arrows run '
      + 'from both cards into the loop. A band states that a transformer block is a forward and a '
      + 'backward and AdamW is three hooks, that neither needs a line of the loop to move, and that '
      + 'this is what the twenty-one parameters of Part 09 grew into: not a finished model but the '
      + 'frame everything else is written against.',
  };
}

// ---------------------------------------------------------------------------
module.exports = { whatsNextMap, theSkeletonIsFixed };

if (require.main === module) {
  emit('01-whats-next-map', whatsNextMap());
  emit('02-the-skeleton-is-fixed', theSkeletonIsFixed());
}
