"""Shared by figures 03, 04 and 05 of post 30: run the post's five-seed training scripts and parse what they print.

runs(names) starts snippets/<name>.py for every name at once (each takes 30 to 50 seconds; run side by side, five
take about a minute and a half), waits, and parses every printed row and the per-seed summary lines. Nothing is
added to the scripts and no other seed is run. Only the scripts of section 7 and section 8 are used, never
snippets/jobs/.

Layout work only: with FIG30_RUNS_DIR set to a folder holding <name>.txt, a saved stdout of the same script, that
text is parsed instead of a new run.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

POST = Path(__file__).resolve().parents[2]
ROOT = POST.parents[1]
INDEX = (POST / "index.md").read_text(encoding="utf-8")

ROW = re.compile(r"^ +(\d) +(\d\.\d{4}) +(\d\.\d{4}) +(\d+\.\d\d) +(\d\.\d{4}) +(\d\.\d{4}) +(\d\.\d{4}) +(\d+\.\d\d) +"
                 r"(\d+\.\d\d) +(\d+) +(\d+)$", re.M)
KEYS = ("train", "test", "gap", "train_loss", "test_loss", "penalty", "sum_abs", "sum_sq", "near", "exact")


def _ints(s):
    return [int(v) for v in s.split(", ")]


def parse(out):
    rows = []
    for m in ROW.finditer(out):
        g = m.groups()
        r = dict(zip(KEYS, g[1:]))
        r["seed"] = int(g[0])
        for k in ("train", "test"):                         # 300 points each: an accuracy is a count over 300
            count = round(300 * float(r[k]))
            assert f"{count / 300:.4f}" == r[k], r
            r[k + "_pct"] = count / 3                       # 0.9533 -> 286 of 300 -> 95.33 percent
        rows.append(r)
    assert [r["seed"] for r in rows] == [0, 1, 2, 3, 4], out
    for r in rows:                                          # the gap column is training minus test, in points
        assert f"{r['train_pct'] - r['test_pct']:.2f}" == r["gap"], r
    m = re.search(r"^dead neurons of dense1 \(no output on any training point\), of 64, per seed: \[([\d, ]+)\]$",
                  out, re.M)
    dead = _ints(m.group(1))
    m = re.search(r"^weights of dense1 with \|w\| < 0\.001, per seed: in dead neurons \[([\d, ]+)\], \d+ of their "
                  r"\d+ weights; in live neurons \[([\d, ]+)\], \d+ of their \d+ weights$", out, re.M)
    near_dead, near_live = _ints(m.group(1)), _ints(m.group(2))
    m = re.search(r"test accuracy (\d+\.\d\d) to (\d+\.\d\d) \(mean (\d+\.\d\d)\) percent", out)
    test_mean = m.group(3)
    for r, d, nd, nl in zip(rows, dead, near_dead, near_live):
        r["dead"], r["near_dead"], r["near_live"] = d, nd, nl
        assert nd + nl == int(r["near"]), r                 # the split adds up to the row's count
    return dict(rows=rows, test_mean=test_mean, out=out)


def runs(names):
    cache = os.environ.get("FIG30_RUNS_DIR")
    outs = {}
    if cache:
        for n in names:
            outs[n] = (Path(cache) / f"{n}.txt").read_text(encoding="utf-8")
    else:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        procs = {n: subprocess.Popen([sys.executable, str(POST / "snippets" / f"{n}.py")], cwd=str(ROOT),
                                     stdout=subprocess.PIPE, text=True, env=env) for n in names}
        for n, p in procs.items():
            out, _ = p.communicate()
            assert p.returncode == 0, n
            outs[n] = out
    return {n: parse(o) for n, o in outs.items()}


def rng(vals, fmt="{:.2f}"):
    """'a to b' over the five seeds, as the tables of sections 7 and 8 print a range."""
    return f"{fmt.format(min(vals))} to {fmt.format(max(vals))}"
