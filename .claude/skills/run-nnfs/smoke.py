"""Smoke-drive the NNFS repo: the four projects + the cumulative notebook.

Run FROM THE REPO ROOT (checkpoints and dataset caches are cwd-relative):

    python .claude/skills/run-nnfs/smoke.py             # all steps
    python .claude/skills/run-nnfs/smoke.py mnist        # one step
    python .claude/skills/run-nnfs/smoke.py --fresh      # force retrain (temp checkpoints)

Steps: moons, mnist, fashion, housing, notebook

Per project: if the canonical checkpoint is missing (clean machine), a short
smoke training run produces it; then evaluate.py runs against it and the
parsed metric must clear a threshold. Thresholds are set for the SHORT
training runs, so fully-trained checkpoints clear them easily.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PROJECTS = {
    # step: (project dir, checkpoint, smoke-train args, eval args,
    #        metric regex (group 1 = float), threshold, metric name)
    "moons": (
        "projects/02-binary-classifier", "moons_weights.pkl",
        ["--epochs", "500"], [],
        r"\[test\] loss=[\d.]+\s+accuracy=([\d.]+)", 0.90, "test accuracy",
    ),
    "mnist": (
        "projects/01-mnist-from-scratch", "mnist_weights.pkl",
        ["--epochs", "1", "--backend", "manual"], ["--backend", "manual"],
        r"accuracy\s+([\d.]+)", 0.90, "test accuracy",
    ),
    "fashion": (
        "projects/03-fashion-mnist", "fashion_mnist_weights.pkl",
        ["--epochs", "1", "--backend", "manual"], ["--backend", "manual"],
        r"accuracy\s+([\d.]+)", 0.75, "test accuracy",
    ),
    "housing": (
        "projects/04-california-housing-regression", "cal_housing_weights.pkl",
        ["--epochs", "50"], [],
        r"R\^?2[^\d-]*(-?[\d.]+)", 0.50, "R^2",
    ),
}


def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    # PYTHONUTF8: the evaluate scripts print Unicode (e.g. U+2192) and crash
    # on Windows cp1252 consoles without it.
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, **kw)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"exit {r.returncode} after {dt:.0f}s: {cmd}")
    print(f"    ok ({dt:.0f}s)")
    return r.stdout


def step_project(name, fresh):
    proj, ckpt, train_args, eval_args, pat, thresh, metric = PROJECTS[name]
    py = sys.executable
    ckpt_path = ROOT / ckpt

    if fresh:
        ckpt = str(Path(tempfile.gettempdir()) / f"nnfs_smoke_{ckpt}")
        ckpt_path = Path(ckpt)
        ckpt_path.unlink(missing_ok=True)

    if not ckpt_path.exists():
        print(f"  checkpoint {ckpt} missing -> smoke training")
        run([py, f"{proj}/train.py", "--checkpoint", ckpt, *train_args])
    else:
        print(f"  using existing checkpoint {ckpt}")

    out = run([py, f"{proj}/evaluate.py", "--checkpoint", ckpt, *eval_args])

    m = re.search(pat, out, re.IGNORECASE)
    if not m:
        print(out[-1500:])
        raise RuntimeError(f"could not parse {metric!r} from evaluate output")
    val = float(m.group(1))
    print(f"  {metric} = {val}  (threshold {thresh})")
    if val < thresh:
        raise RuntimeError(f"{metric} {val} below threshold {thresh}")


def step_notebook(_fresh):
    """Execute every code cell of cumulative_notebook.ipynb in-process."""
    nb = json.loads((ROOT / "cumulative_notebook.ipynb").read_text(encoding="utf-8"))
    g = {}
    t0 = time.time()
    n = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        with redirect_stdout(io.StringIO()) as buf:
            try:
                exec(compile(src, f"cell{i}", "exec"), g)
            except Exception as e:
                print(buf.getvalue()[-1500:])
                raise RuntimeError(f"notebook cell {i} failed: {e}") from e
        n += 1
    print(f"  {n} code cells executed ({time.time() - t0:.0f}s)")
    acc = g.get("test_accuracy")
    print(f"  final test accuracy = {acc:.4f}  (threshold 0.5)")
    if acc is None or acc < 0.5:
        raise RuntimeError(f"notebook final test accuracy {acc} below 0.5")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("steps", nargs="*",
                   choices=[*PROJECTS, "notebook"], default=None,
                   help="subset of steps (default: all)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoints; retrain into temp files")
    args = p.parse_args()
    steps = args.steps or [*PROJECTS, "notebook"]

    failed = []
    for s in steps:
        print(f"\n=== {s} ===")
        try:
            (step_notebook if s == "notebook" else
             lambda fresh, s=s: step_project(s, fresh))(args.fresh)
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(s)
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'ALL STEPS PASSED'} "
          f"({len(steps) - len(failed)}/{len(steps)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
