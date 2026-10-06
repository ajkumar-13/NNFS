"""Post 25, section 7: final accuracy on seeds 0 to 4 for gradient descent, momentum and AdaGrad.

Run from anywhere:  python posts/25-adagrad/diagrams/src/04-five-seeds.py   (about a minute)
Writes posts/25-adagrad/diagrams/04-five-seeds.svg.
The three seed scripts of the post, snippets/seeds_sgd.py, seeds_momentum.py and seeds_adagrad.py, are run here
as they are (three processes side by side, each 35 to 50 seconds), and their printed rows give every value drawn.
Each row is asserted against the cell of section 7's table in index.md, and the counts the section states (AdaGrad
above gradient descent on all five seeds, above momentum on three) are asserted on the printed accuracies.

Layout: one dot plot, a row per seed, the three final accuracies on one axis.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\admin\Desktop\series-standard\tools")
from figkit import Figure, Box, rich, num  # noqa: E402

POST = Path(__file__).resolve().parents[2]
INDEX = (POST / "index.md").read_text(encoding="utf-8")
NAMES = ("sgd", "momentum", "adagrad")
procs = {n: subprocess.Popen([sys.executable, str(POST / "snippets" / f"seeds_{n}.py")], cwd=str(POST.parents[1]),
                             stdout=subprocess.PIPE, text=True) for n in NAMES}
RUNS = {}
for n, p in procs.items():
    out, _ = p.communicate()
    assert p.returncode == 0, n
    rows = re.findall(r"^\s*(\d)\s+(\d\.\d{4})\s+(\d\.\d{4})\s+(\d+) of 64$", out, re.M)
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3, 4], (n, out)
    RUNS[n] = [(r[1], r[2], int(r[3])) for r in rows]

# -- every printed row is the cell of section 7's table
for seed in range(5):
    cells = " | ".join(f"{loss}, {acc}, {dead}" for loss, acc, dead in (RUNS[n][seed] for n in NAMES))
    assert f"| {seed} | {cells} |" in INDEX, (seed, cells)

ACC = {n: [100 * float(r[1]) for r in RUNS[n]] for n in NAMES}
assert all(a > s for a, s in zip(ACC["adagrad"], ACC["sgd"]))                  # above gradient descent on all five
assert [a > m for a, m in zip(ACC["adagrad"], ACC["momentum"])] == [False, True, True, True, False]
assert (f"{min(ACC['adagrad']):.1f}", f"{max(ACC['adagrad']):.1f}") == ("75.7", "92.7")
assert f"{ACC['momentum'][0] - ACC['adagrad'][0]:.1f}" == "11.7"
assert "Seed 0 alone would say that AdaGrad loses to momentum by 11.7 points; five seeds do not support that" in INDEX
assert "AdaGrad's accuracies lie between 75.7 and 92.7 percent and momentum's between 68.7 and 98.0" in INDEX

def pct(n):
    v = [f"{a:.1f}" for a in ACC[n]]
    return ", ".join(v[:-1]) + " and " + v[-1]


COLORS = ("ink-muted", "ink", "blue")
SHAPES = ("circle", "square", "diamond")
fig = Figure(
    "04-five-seeds", "Above gradient descent on all five seeds, momentum on three",
    "A dot plot of the final accuracy on the spiral after 10,001 epochs, one row per seed, on an axis from 50 to "
    f"100 percent. Gradient descent with learning rate 1 reaches {pct('sgd')} percent on seeds 0 to 4; momentum "
    f"with decay 0.001 and momentum 0.9 reaches {pct('momentum')}; AdaGrad with decay 0.0001 reaches "
    f"{pct('adagrad')}. AdaGrad is above gradient descent on all five seeds and above momentum on seeds 1, 2 and 3.",
    subtitle="Final accuracy on the spiral after 10,001 epochs, seeds 0 to 4, each optimiser at its documented setting.",
    data_w=True)

rows = [(f"seed {k}", [ACC[n][k] for n in NAMES]) for k in range(5)]
ax = fig.dot_plot(Box(40, 112, 880, 296), rows, 50, 100, [50, 60, 70, 80, 90, 100], colors=COLORS, shapes=SHAPES,
                  fmt=lambda v: num(v), unit="%", label_w=96, pad_right=72, axis_label="final accuracy")
with fig.data():
    for k in range(5):
        a = ACC["adagrad"][k]
        fig.text(ax.sx(a), ax.sy(k) - 12, f"{a:.1f}", "value", anchor="middle", color="blue")
fig.legend(136, 448, [dict(color=c, label=l, mark=m) for c, l, m in
                      zip(COLORS, ("gradient descent, post 22", "momentum, post 24", "AdaGrad, this post"), SHAPES)],
           direction="row", gap=40)
fig.caption("Seed 0 alone would put momentum 11.7 points ahead; over five seeds neither of the two leads.")
fig.write()
