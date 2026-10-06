"""Post 35 hero (sections 2 to 4): the three kinds of addition the post names, one card per addition.

Run from anywhere:  python posts/35-whats-next/diagrams/src/01-whats-next-map.py
Writes posts/35-whats-next/diagrams/01-whats-next-map.svg.

Nothing here is a measurement and the figure shows no numbers except the years of the works it cites. Every
heading, card name, line and source is the post's own wording, asserted against index.md (markdown emphasis and
backticks removed): the column headings are the headings of sections 2, 3 and 4; the card names are the
subsection headings, shortened where they carry a colon; the line under each name is the post's phrase for what
the addition adds (the summary table of section 9 for the three layers, the sections themselves for the others);
the sources are the works each section names. The dashed slot closing the framings column is section 4's
sentence "the networks, losses and optimisers stay".

Layout: three columns of neutral cards (no role colours: nothing here is an input, a weight or a gradient), the
columns' bottoms aligned: 3 layer cards of 148, 5 infrastructure cards of 84 and 4 framing cards of 84 plus the
dashed slot, 12 apart.
"""
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, text_width  # noqa: E402

POST = Path(__file__).resolve().parents[2]
RAW = (POST / "index.md").read_text(encoding="utf-8")
INDEX = re.sub(r"[`*]", "", RAW)


def said(*pieces):
    """Every piece is the post's own wording."""
    for p in pieces:
        assert p in INDEX, p
    return pieces[0] if len(pieces) == 1 else pieces


# -- the content, column by column: (name, what it adds (lines), source lines [(text, style)])
COLS = [
    (said("New layer types"), [
        (said("Convolutional layers"), ["one filter's weights reused", "at every position of a grid"],
         [(said("LeCun et al. (1998)"), "note"), (said("cnn-from-scratch"), "code")]),
        (said("Recurrent layers"), ["one layer's weights reused at", "every timestep, with a carried state"],
         [(said("Hochreiter and Schmidhuber (1997)"), "note"), (said("rnn-from-scratch"), "code")]),
        (said("Attention and transformers"), ["every position reads every other,", "weighted by a softmax of scores"],
         [(said("Vaswani et al. (2017)"), "note"), ("Karpathy, Let's build GPT", "note")]),
    ]),
    (said("New training infrastructure"), [
        (said("Normalisation"), [said("activations rescaled")], [(said("Ioffe and Szegedy (2015)"), "note")]),
        (said("Residual connections"), [said("adds the input of a block to its output")],
         [(said("He et al. (2016)"), "note")]),
        ("Learning-rate schedules", [("another formula in ", "pre_update_params")],
         [(said("Loshchilov and Hutter (2017)"), "note")]),
        (said("Mixed precision"), [said("forward and backward passes in 16 bits")],
         [("Micikevicius et al. (2018)", "note")]),
        (said("Distributed training"), [said("every device holds a copy of the model")],
         [(said("DistributedDataParallel"), "code")]),
    ]),
    (said("New problem framings"), [
        (said("Self-supervised learning"), [said("takes its labels from the input")],
         [("BERT, GPT-2, Masked Autoencoders", "note")]),
        (said("Transfer learning"), [said("a pretrained network")], [(said("LoRA (Hu et al., 2022)"), "note")]),
        (said("Reinforcement learning"), [said("no labels; reward-weighted gradients")],
         [("Sutton and Barto (2018)", "note")]),
        (said("Diffusion models"), [said("a denoising objective")], [(said("Ho, Jain and Abbeel (2020)"), "note")]),
    ]),
]
# the wording that is split over lines or joined from several places of the post
said("one filter's weights reused at every position of a grid",
     "one layer's weights reused at every timestep, with a carried state",
     "every position reads every other, weighted by a softmax of scores",
     "another formula in pre_update_params", "Learning-rate schedules beyond",
     "Micikevicius et al., 2018", "Sutton and Barto", "MIT Press, 2018", "BERT", "GPT-2", "Masked Autoencoders", "Karpathy", "Let's build GPT",
     "the networks, losses and optimisers stay")
assert "### 3.3. Learning-rate schedules" in RAW and "### 4.2. Transfer learning and fine-tuning" in RAW
assert [len(c[1]) for c in COLS] == [3, 5, 4]

fig = Figure(
    "01-whats-next-map", "Everything after this series is an addition",
    "Three columns of cards. New layer types: convolutional layers, one filter's weights reused at every "
    "position of a grid, LeCun et al. (1998) and cnn-from-scratch; recurrent layers, one layer's weights reused at "
    "every timestep with a carried state, Hochreiter and Schmidhuber (1997) and rnn-from-scratch; attention and "
    "transformers, every position reads every other weighted by a softmax of scores, Vaswani et al. (2017) and "
    "Karpathy's Let's build GPT. New training infrastructure: normalisation, activations rescaled, Ioffe and "
    "Szegedy (2015); residual connections, adds the input of a block to its output, He et al. (2016); "
    "learning-rate schedules, another formula in pre_update_params, Loshchilov and Hutter (2017); mixed precision, "
    "forward and backward passes in 16 bits, Micikevicius et al. (2018); distributed training, every device holds a "
    "copy of the model, DistributedDataParallel. New problem framings: self-supervised learning, takes its labels "
    "from the input, BERT, GPT-2 and Masked Autoencoders; transfer learning, a pretrained network, LoRA (Hu et al., "
    "2022); reinforcement learning, no labels and reward-weighted gradients, Sutton and Barto (2018); diffusion models, a "
    "denoising objective, Ho, Jain and Abbeel (2020); and a dashed slot: the networks, losses and optimisers stay.",
    subtitle="What sections 2 to 4 add to the forward pass, backward pass and optimiser step of posts 01 to 34.",
    height=720, data_w=True)

XS, WS = (40, 328, 648), (264, 296, 272)            # columns 24 apart; the middle one holds the longest lines
HEAD_Y, TOP, GAP = 136, 152, 12
H_CARD = {0: 148, 1: 84, 2: 84}
PAD = 12
BOTTOM = TOP + 3 * 148 + 2 * GAP
assert BOTTOM == TOP + 5 * 84 + 4 * GAP == 620 and BOTTOM <= 656


def lines_fit(x, w, s, style):
    size, weight = (14, 400) if style != "head" else (16, 600)
    # the calibrated sans estimate carries 5 percent slack; figcheck measures the rendered width (data_w)
    tw = text_width(s, 14, mono=True) if style == "code" else text_width(s, size, weight=weight) / 1.05
    assert tw <= w - 2 * PAD, (s, tw, w)


for k, (heading, cards) in enumerate(COLS):
    x, w = XS[k], WS[k]
    fig.text(x, HEAD_Y, heading, "head")
    h = H_CARD[k]
    for i, (name, adds, sources) in enumerate(cards):
        y = TOP + i * (h + GAP)
        fig.card(x, y, w, h)
        tx = x + PAD
        lines_fit(x, w, name, "head")
        fig.text(tx, y + (28 if k == 0 else 24), name, "head")
        if k == 0:                                    # name, two lines of what it adds, the paper, from scratch
            ys_add, ys_src = (y + 52, y + 72), (y + 104, y + 124)
        else:                                         # name, one line, one source
            ys_add, ys_src = (y + 48,), (y + 72,)
        assert len(adds) == len(ys_add) and len(sources) == len(ys_src)
        for s, yy in zip(adds, ys_add):
            if isinstance(s, tuple):                  # sans words, then a code name drawn as its own text
                words, code = s
                cw = 4 * round(text_width(words, 14, weight=400) / 1.05 / 4)
                assert cw + text_width(code, 14, mono=True) <= w - 2 * PAD
                fig.text(tx, yy, words, "label")
                with fig.data():
                    fig.text(tx + cw, yy, code, "code", snap=False)
            else:
                lines_fit(x, w, s, "label")
                fig.text(tx, yy, s, "label")
        for (s, style), yy in zip(sources, ys_src):
            lines_fit(x, w, s, style)
            fig.text(tx, yy, s, style)

# the framings column closes with what stays: a dashed slot the size of a card
slot = Box(XS[2], TOP + 4 * (84 + GAP), WS[2], 84)
assert slot.bottom == BOTTOM
fig.outline(slot, "ink-muted", width=1, dash="lead", radius=8)
fig.text(slot.x + PAD, slot.y + 36, "The networks, losses and", "note")
fig.text(slot.x + PAD, slot.y + 56, "optimisers stay.", "note")

fig.caption("Under each name: what it adds, then where to read; layer cards add a from-scratch tutorial.")
fig.write()
