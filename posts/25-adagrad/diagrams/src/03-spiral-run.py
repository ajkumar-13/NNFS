"""Post 25, section 7: the documented run, Optimizer_Adagrad(learning_rate=1.0, decay=1e-4) on seed 0.

Run from anywhere:  python posts/25-adagrad/diagrams/src/03-spiral-run.py   (about 10 seconds)
Writes posts/25-adagrad/diagrams/03-spiral-run.svg.
snippets/adagrad.py is imported here (runpy, without its main block) and its train function makes the post's own
documented run, the one section 7 prints, with a watch hook that records the loss and the effective rates of all
387 parameters after every epoch's update, computed as the snippet's own watch computes them. Every row of the
section 7 table and the two split lines are rebuilt from those records with the snippet's print formats and
asserted against the listings in index.md (which the series lint checks against the snippet's output).

Layout: two charts on one log epoch axis, the loss above and the effective rates below.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sub, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(POST / "snippets" / "adagrad.py"), run_name="snippet")
INDEX = (POST / "index.md").read_text(encoding="utf-8")

PRINTED = (0, 1, 2, 10, 100, 1000, 2000, 5000, 9000, 10000)
LOSS, MEDIAN, SMALLEST, ROWS, SPLIT = [], [], [], {}, {}


def watch(epoch, loss, accuracy, dead, layers, optimizer):
    """The snippet's watch, recording instead of printing."""
    cache = np.concatenate([c.ravel() for layer in layers for c in (layer.weight_cache, layer.bias_cache)])
    gradient = np.concatenate([g.ravel() for layer in layers for g in (layer.dweights, layer.dbiases)])
    rate = optimizer.current_learning_rate / (np.sqrt(cache) + optimizer.epsilon)
    step = rate * np.abs(gradient)
    LOSS.append(float(loss))
    MEDIAN.append(float(np.median(rate)))
    SMALLEST.append(float(rate.min()))
    if epoch in PRINTED:
        ROWS[epoch] = (f"{epoch:6d}  {loss:.4f}  {accuracy:.3f}  {dead:3d}   {optimizer.current_learning_rate:.4f}  "
                       f"{np.median(cache):.2e}  {cache.max():.2e}  {np.median(rate):10.2f}  {rate.min():8.3f}  "
                       f"{int(np.sum(rate < 1)):6d}   {step.mean():.2e}")
    if epoch in (100, 10000):
        gone = np.max(layers[0].output, axis=0) <= 0
        theirs = np.concatenate([np.tile(gone, 2), gone, np.repeat(gone, 3), np.zeros(3, dtype=bool)])
        SPLIT[epoch] = (int(theirs.sum()), float(np.median(rate[theirs])), int(np.sum(~theirs)),
                        float(np.median(rate[~theirs])))
        SPLIT[str(epoch)] = (f"epoch {epoch:5d}: {int(theirs.sum())} parameters of dead neurons, rate median "
                             f"{np.median(rate[theirs]):.2f}, {int(np.sum(rate[theirs] > 1))} above 1; "
                             f"the other {int(np.sum(~theirs))}, rate median {np.median(rate[~theirs]):.2f}, "
                             f"{int(np.sum(rate[~theirs] > 1))} above 1")
    if epoch == 10:
        ROWS["dead10"] = dead


final = s["train"](s["Optimizer_Adagrad"](learning_rate=1.0, decay=1e-4), seed=0, watch=watch)
for e in PRINTED:
    assert ROWS[e] in INDEX, ROWS[e]
for e in ("100", "10000"):
    assert SPLIT[e] in INDEX, SPLIT[e]
assert f"final: loss {final[0]:.4f}, accuracy {final[1]:.4f}, dead neurons {final[2]} of 64" in INDEX
assert ROWS["dead10"] == 32 and final[2] == 34
assert len(LOSS) == 10001

# the values printed in the figure, in the post's roundings
assert (f"{LOSS[0]:.4f}", f"{MEDIAN[0]:.0f}", f"{LOSS[2]:.4f}", f"{LOSS[10]:.4f}", f"{LOSS[10000]:.4f}") == \
       ("1.0986", "10685", "9.9083", "1.1331", "0.3847")
assert (f"{MEDIAN[10000]:.2f}", f"{SMALLEST[10000]:.3f}") == ("1.20", "0.045")
assert (f"{LOSS[1]:.4f}", f"{MEDIAN[1]:.2f}", f"{SMALLEST[1]:.3f}") == ("6.2613", "18.42", "1.171")
assert SPLIT[10000][:1] == (204,) and SPLIT[10000][2] == 183
assert (f"{SPLIT[10000][1]:.2f}", f"{SPLIT[10000][3]:.2f}") == ("3.71", "0.41")
assert abs(LOSS[0] - np.log(3)) < 1e-4
assert 0.3 < min(LOSS) and max(LOSS) < 20 and min(MEDIAN[1:]) > 1e-2 and max(MEDIAN[1:]) < 1e2 and min(SMALLEST[1:]) > 1e-2

# Every epoch is drawn, thinned only below the resolution of the chart: the loss alternates up and down by a few
# thousandths from about epoch 20 on, so a sample of epochs would turn that into false spikes. Each run of
# epochs that falls within one quarter of a unit of the log axis keeps its first, smallest, largest and last value.
def envelope(values, x0=96, w=616):          # the plot: x 96, width 880 - 56 - 208
    keep, bucket = [], None
    for e in range(1, 10001):
        b = int(4 * (x0 + w * np.log10(e) / 4))
        if b != bucket:
            if bucket is not None:
                keep.extend(sorted({run[0], run[-1], min(run, key=values.__getitem__),
                                    max(run, key=values.__getitem__)}))
            bucket, run = b, []
        run.append(e)
    keep.extend(sorted({run[0], run[-1], min(run, key=values.__getitem__), max(run, key=values.__getitem__)}))
    return keep

EL, ES, EM = envelope(LOSS), envelope(SMALLEST), envelope(MEDIAN)
DOTS = PRINTED[1:]

fig = Figure(
    "03-spiral-run", "A violent start, then rates that only fall",
    "Two charts of the documented run on seed 0, learning rate 1 and decay 10 to the minus 4, on one logarithmic "
    "epoch axis from 1 to 10,000; epoch 0, with the loss 1.0986 and a median rate of 10,685, lies left of it. Top, "
    "the loss: 6.2613 at epoch 1 and 9.9083 at epoch 2, back to 1.1331 at epoch 10, when 32 of the 64 hidden "
    "neurons are dead, then falling below ln 3 to 0.3847 at epoch 10,000. Bottom, the effective rates of the 387 "
    "parameters after each epoch's update on a logarithmic axis: the median falls from 18.42 at epoch 1 to 1.20 "
    "and the smallest from 1.171 to 0.045. At epoch 10,000 the 204 parameters of dead neurons have a median "
    "rate of 3.71 and the other 183 one of 0.41. Dots mark the epochs the script prints.",
    subtitle=rich("Seed 0, ", var("α"), " = 1, decay ", sup("10", num(-4), italic=False),
                  ". Epoch 0, loss 1.0986 and median rate 10,685, lies left of the axes."),
    height=720, data_w=True)

LABEL_W = 208
X_AX = (1, 1e4, [10.0 ** k for k in range(5)])

# -- the loss
fig.text(40, 120, "Loss, log scale", "head")
ax = fig.line_chart(Box(40, 128, 880, 224), [dict(xs=EL, ys=[LOSS[e] for e in EL], color="error", points=False)],
                    x=X_AX, y=(0.3, 20, [0.3, 1, 3, 10]), x_log=True, y_log=True, labels=False, label_w=LABEL_W,
                    fmt_y=lambda v: f"{v:g}", ref_lines=[dict(y=float(np.log(3)), label="ln 3 = 1.0986", at="right")])
for e in DOTS:
    ax.point(e, LOSS[e], "circle", "error", size=8)
ax.text(1, LOSS[1], "6.2613", "value", color="error", dx=10, dy=20)
ax.text(2, LOSS[2], "9.9083", "value", color="error", anchor="middle", dy=-12)
ax.text(10, LOSS[10], "epoch 10: 1.1331, and 32 of 64 hidden neurons dead", "note", dx=12, dy=-12)
ax.text(1e4, LOSS[10000], "0.3847", "value", color="error", dx=12, dy=5)

# -- the effective rates
fig.text(40, 388, rich("Effective learning rates ", var("α"), "/(√", var("G"), " + ", var("ε"), "), log scale"), "head")
ax = fig.line_chart(Box(40, 408, 880, 248),
                    [dict(xs=ES, ys=[SMALLEST[e] for e in ES], color="ink-muted", points=False),
                     dict(xs=EM, ys=[MEDIAN[e] for e in EM], color="ink", points=False)],
                    x=X_AX, y=(1e-2, 1e2, [10.0 ** k for k in range(-2, 3)]), x_log=True, y_log=True,
                    labels=False, label_w=LABEL_W, x_label="epoch, log scale")
for e in DOTS:
    ax.point(e, MEDIAN[e], "circle", "ink", size=8)
    ax.point(e, SMALLEST[e], "triangle", "ink-muted", size=10)
ax.text(1, MEDIAN[1], "median of 387, 18.42", "note", dx=12, dy=-6)
ax.text(1, SMALLEST[1], "smallest, 1.171", "note", color="ink-muted", dx=12, dy=-8)
ax.text(1e4, MEDIAN[10000], "median of 387, 1.20", "note", dx=12, dy=5)
ax.text(1e4, SMALLEST[10000], "smallest, 0.045", "note", color="ink-muted", dx=12, dy=5)
ax.point(1e4, SPLIT[10000][1], "diamond", "ink", size=10, hollow=True)
ax.point(1e4, SPLIT[10000][3], "diamond", "ink", size=10, hollow=True)
ax.text(1e4, SPLIT[10000][1], "204 of dead neurons, 3.71", "note", dx=12, dy=5)
ax.text(1e4, SPLIT[10000][3], "the other 183, 0.41", "note", dx=12, dy=5)

fig.caption("Dots: the epochs the script prints. Diamonds: the medians of the two groups at epoch 10,000.")
fig.write()
