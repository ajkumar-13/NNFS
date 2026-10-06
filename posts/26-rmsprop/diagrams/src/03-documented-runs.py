"""Post 26, sections 7.1 and 11: the two documented runs on seed 0, every epoch, loss and mean update size.

Run from anywhere:  python posts/26-rmsprop/diagrams/src/03-documented-runs.py   (about 40 seconds)
Writes posts/26-rmsprop/diagrams/03-documented-runs.svg.
snippets/rmsprop.py is imported here (runpy, without its main block) and its train function makes the post's two
documented runs, Optimizer_Adagrad(learning_rate=1.0, decay=1e-4) and Optimizer_RMSprop(learning_rate=0.02,
decay=1e-5, rho=0.999), with a watch that computes, after every epoch's update, what the snippet's documented_run
computes at its seven watched epochs: the scale current_learning_rate / (sqrt(cache) + epsilon) and the mean of
scale * |gradient| over the 387 parameters. The seven rows of each run and the lines around the highest late loss are
rebuilt with the snippet's print formats and asserted against the listings of sections 7.1 and 11 in index.md (which
the series lint checks against the snippet's output).

Layout: two charts on one linear epoch axis, the loss above and the mean update size below.
"""
import contextlib
import io
import runpy
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, var, sup, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(POST / "snippets" / "rmsprop.py"), run_name="snippet")
assert 'documented_run("AdaGrad", Optimizer_Adagrad(learning_rate=1.0, decay=1e-4))' in INDEX
assert 'documented_run("RMSProp", Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999))' in INDEX
WATCHED = (0, 1, 10, 100, 1000, 5000, 10000)


def run(optimizer):
    """Every epoch's loss, accuracy and mean |update|, and the printed rows at the watched epochs."""
    update, rows = [], {}

    def watch(epoch, dense1, dense2):
        cache = s["all_parameters"](dense1, dense2, "cache")
        gradient = s["all_parameters"](dense1, dense2, "d")
        scale = optimizer.current_learning_rate / (np.sqrt(cache) + optimizer.epsilon)
        update.append(float(np.mean(np.abs(scale * gradient))))
        if epoch in WATCHED:
            rows[epoch] = (optimizer.current_learning_rate, np.median(cache), cache.max(), np.median(scale),
                           update[-1])

    losses, accuracies = s["train"](optimizer, watch=watch)
    for epoch, (rate, median, largest, scale, upd) in rows.items():
        line = (f"{epoch:6d}   {losses[epoch]:.4f}   {accuracies[epoch]:.4f}     {rate:.5f}   {median:.3e}      "
                f"{largest:.3e}       {scale:10.3f}     {upd:.2e}")
        assert line in INDEX, line
    return losses, accuracies, np.array(update)


A_LOSS, A_ACC, A_UPD = run(s["Optimizer_Adagrad"](learning_rate=1.0, decay=1e-4))
R_LOSS, R_ACC, R_UPD = run(s["Optimizer_RMSprop"](learning_rate=0.02, decay=1e-5, rho=0.999))
assert len(A_LOSS) == len(R_LOSS) == len(A_UPD) == len(R_UPD) == 10001

# -- section 11: the highest late loss of the RMSProp run and the epochs around it
PEAK = 9001 + int(np.argmax(R_LOSS[-1000:]))
assert PEAK == 9130 and "around the highest late loss, epoch 9130:" in INDEX
for e in (9115, 9121, 9127, 9130, 9133, 9136, 9139):      # the seven of the snippet's nine lines section 11 shows
    assert e in range(PEAK - 15, PEAK + 10, 3)
    line = f"{e:6d}   {R_LOSS[e]:.4f}   {R_ACC[e]:.4f}"
    assert line in INDEX, line
LATE_MED = np.median(R_LOSS[-1000:])
assert int(np.sum(R_LOSS[-1000:] > 2 * LATE_MED)) == 19 and "above twice its late median on 19" in INDEX

# -- the values the figure prints, in the post's roundings
assert (f"{A_LOSS[-1]:.4f}", f"{R_LOSS[-1]:.4f}", f"{R_LOSS[PEAK]:.4f}") == ("0.3847", "0.2379", "1.6829")
assert (f"{A_UPD[100]:.2e}", f"{A_UPD[10000]:.2e}", f"{R_UPD[100]:.2e}", f"{R_UPD[10000]:.2e}") == \
       ("1.18e-02", "9.42e-04", "1.05e-02", "6.08e-03")
A_FACTOR, R_FACTOR = A_UPD[100] / A_UPD[10000], R_UPD[100] / R_UPD[10000]
assert (f"{A_FACTOR:.1f}", f"{R_FACTOR:.1f}", f"{R_UPD[10000] / A_UPD[10000]:.1f}") == ("12.5", "1.7", "6.5")
assert "a factor of 12.5" in INDEX and "by a factor of 1.7" in INDEX and "it is 6.5 times AdaGrad's" in INDEX
assert (f"{A_LOSS[1]:.4f}", f"{R_LOSS[1]:.4f}") == ("6.2613", "2.8219")
assert 0.2 < min(R_LOSS.min(), A_LOSS.min()) and max(R_LOSS.max(), A_LOSS.max()) < 20
assert 1e-4 < min(A_UPD.min(), R_UPD.min()) and max(A_UPD.max(), R_UPD.max()) < 1


# Every epoch is drawn, thinned only below the resolution of the chart: each run of epochs that falls within one
# unit of the epoch axis keeps its first, smallest, largest and last value, so no spike is lost.
def envelope(values, x0, w):
    keep, run_, bucket = [], [], None
    for e in range(10001):
        b = int(x0 + w * e / 10000)
        if b != bucket and run_:
            keep.extend(sorted({run_[0], run_[-1], min(run_, key=values.__getitem__),
                                max(run_, key=values.__getitem__)}))
            run_ = []
        bucket = b
        run_.append(e)
    keep.extend(sorted({run_[0], run_[-1], min(run_, key=values.__getitem__), max(run_, key=values.__getitem__)}))
    return keep


ADA_C, RMS_C = "ink-muted", "blue"
LABEL_W = 144
PLOT_X, PLOT_W = 40 + 56, 880 - 56 - LABEL_W
DOTS = (1000, 5000, 10000)


def sci(v):
    m, e = f"{v:.2e}".split("e")
    return rich(m, " × ", sup("10", num(int(e)), italic=False))


fig = Figure(
    "03-documented-runs", "AdaGrad's steps shrink; RMSProp's keep their size and spike",
    "Two charts of the documented runs on seed 0 over 10,001 epochs, every epoch drawn: AdaGrad with learning rate 1 "
    "and decay 10 to the minus 4 in grey, RMSProp with learning rate 0.02, decay 10 to the minus 5 and rho 0.999 in "
    "blue. Top, the loss on a log axis: after a jump at the start, to 6.2613 and 2.8219 at epoch 1, AdaGrad falls "
    "smoothly to 0.3847 and RMSProp to 0.2379, but RMSProp's loss spikes late in the run, highest at epoch 9,130 "
    "with 1.6829. Bottom, the mean distance a parameter moves per update on a log axis: AdaGrad's falls from 1.18 "
    "times 10 to the minus 2 at epoch 100 to 9.42 times 10 to the minus 4 at epoch 10,000, 12.5 times smaller; "
    "RMSProp's from 1.05 times 10 to the minus 2 to 6.08 times 10 to the minus 3, 1.7 times smaller and 6.5 times "
    "AdaGrad's at the end. Dots mark epochs 100, 1,000, 5,000 and 10,000, which the script prints.",
    subtitle=rich("Seed 0. AdaGrad: ", var("α"), " = 1, decay ", sup("10", num(-4), italic=False), ". RMSProp: ",
                  var("α"), " = 0.02, decay ", sup("10", num(-5), italic=False), ", ", var("ρ"), " = 0.999."),
    height=720, data_w=True)

X_AX = (0, 10000, [0, 2000, 4000, 6000, 8000, 10000])

# -- the loss
fig.text(40, 120, "Loss, log scale", "head")
EA, ER = envelope(A_LOSS, PLOT_X, PLOT_W), envelope(R_LOSS, PLOT_X, PLOT_W)
ax = fig.line_chart(Box(40, 128, 880, 216),
                    [dict(xs=EA, ys=[float(A_LOSS[e]) for e in EA], color=ADA_C, points=False),
                     dict(xs=ER, ys=[float(R_LOSS[e]) for e in ER], color=RMS_C, points=False)],
                    x=X_AX, y=(0.15, 20, [0.3, 1, 3, 10]), y_log=True, labels=False, label_w=LABEL_W, bands=[(9000, 10000, RMS_C)],
                    fmt_y=lambda v: f"{v:g}")
for e in DOTS:
    ax.point(e, A_LOSS[e], "square", ADA_C, size=8)
    ax.point(e, R_LOSS[e], "circle", RMS_C, size=8)
ax.text(10000, A_LOSS[-1], "AdaGrad, 0.3847", "note", color=ADA_C, dx=12, dy=1)
ax.text(10000, R_LOSS[-1], "RMSProp, 0.2379", "note", color=RMS_C, dx=12, dy=9)
ax.point(PEAK, R_LOSS[PEAK], "diamond", RMS_C, size=10, hollow=True)
ax.text(PEAK, 5, "epoch 9,130: 1.6829, the highest of the shaded last 1,000 epochs", "note", color=RMS_C, dx=8, anchor="end")
assert R_LOSS[1000:].max() < 4                    # the late spikes stay under the label
ax.text(0, 13, "epoch 1: AdaGrad 6.2613, RMSProp 2.8219", "note", dx=16, dy=5)

# -- the mean update
fig.text(40, 384, "Mean distance moved per update, log scale", "head")
EA, ER = envelope(A_UPD, PLOT_X, PLOT_W), envelope(R_UPD, PLOT_X, PLOT_W)
bx = fig.line_chart(Box(40, 404, 880, 220),
                    [dict(xs=EA, ys=[float(A_UPD[e]) for e in EA], color=ADA_C, points=False),
                     dict(xs=ER, ys=[float(R_UPD[e]) for e in ER], color=RMS_C, points=False)],
                    x=X_AX, y=(1e-4, 1, [10.0 ** k for k in range(-4, 1)]), y_log=True, labels=False,
                    label_w=LABEL_W, x_label="epoch")
for e in DOTS:
    bx.point(e, A_UPD[e], "square", ADA_C, size=8)
    bx.point(e, R_UPD[e], "circle", RMS_C, size=8)
bx.text(10000, A_UPD[-1], sci(A_UPD[-1]), "value", color=ADA_C, dx=12, dy=5)
bx.text(10000, R_UPD[-1], sci(R_UPD[-1]), "value", color=RMS_C, dx=12, dy=5)
bx.text(0, 0.6, rich("AdaGrad: ", sci(A_UPD[100]), " at epoch 100, 12.5 times smaller at 10,000"), "note",
        color=ADA_C, dx=16, dy=5)
bx.text(0, 0.6, rich("RMSProp: ", sci(R_UPD[100]), " at epoch 100, 1.7 times smaller at 10,000"), "note",
        color=RMS_C, dx=16, dy=27)
assert max(A_UPD[10:].max(), R_UPD[10:].max()) < 0.075        # the notes sit above every epoch from 10 on

fig.legend(PLOT_X, 652, [dict(color=ADA_C, label="AdaGrad, post 25", mark="square"),
                                   dict(color=RMS_C, label="RMSProp", mark="circle"),
                                   dict(color=RMS_C, label="highest late loss", mark="diamond", hollow=True)],
           direction="row", gap=40)
fig.caption("Dots: epochs the script prints. Every epoch is drawn.")
fig.write()
