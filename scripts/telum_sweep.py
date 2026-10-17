"""Where does the Telum II accelerator pay off? Sweep Snap ML tree inference on
the CPU (compressed trees) and on the on-chip AI accelerator (NNPA via zDNN)
across batch sizes, thread counts and model sizes.

    python scripts/telum_sweep.py            # batch size x n_jobs on the cached 300-tree model
    python scripts/telum_sweep.py --shapes   # also trains smaller models (about 2 min extra)

Needs artifacts/telum_data.npz and artifacts/snapml_model.pkl (scripts/telum_benchmark.py train).
Writes results/telum_sweep.json and figures/telum_crossover.png.
"""

import json
import pickle
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "telum_data.npz"
MODEL = ROOT / "artifacts" / "snapml_model.pkl"
OUT = ROOT / "results" / "telum_sweep.json"
FIG = ROOT / "figures" / "telum_crossover.png"
FORMATS = ("compress_trees", "zdnn_tensors")
BATCHES = (1, 8, 32, 128, 512, 2048, 8192, 65536, None)   # None = all test payments
THREADS = (1, 2, 3)


def log(msg):
    print(msg, flush=True)


def prepared(model_bytes, fmt, n_jobs):
    m = pickle.loads(model_bytes)
    m.optimize_trees(tree_format=fmt)
    m.set_params(n_jobs=n_jobs)
    return m


def time_batch(model, X, bs):
    """Median microseconds per payment over 3-5 timed repeats."""
    xb = X if bs is None else X[:bs]
    model.predict_proba(xb)                                   # warm-up
    calls = max(1, min(2000, 20000 // len(xb)))
    runs = []
    for _ in range(3 if len(xb) > 10000 else 5):
        t0 = time.perf_counter()
        for _ in range(calls):
            model.predict_proba(xb)
        runs.append((time.perf_counter() - t0) / (calls * len(xb)) * 1e6)
    return statistics.median(runs)


def sweep(model_bytes, X, batches=BATCHES, threads=THREADS):
    rows = []
    for fmt in FORMATS:
        for nj in threads:
            try:
                m = prepared(model_bytes, fmt, nj)
            except Exception as e:                            # keep going if one path is unavailable
                log(f"{fmt} n_jobs={nj}: {type(e).__name__}: {e}")
                continue
            for bs in batches:
                us = time_batch(m, X, bs)
                n = len(X) if bs is None else bs
                rows.append({"format": fmt, "n_jobs": nj, "batch": n, "us_per_payment": round(us, 3),
                             "payments_per_s": round(1e6 / us)})
                log(f"{fmt:14} n_jobs={nj} batch={n:>7}: {us:9.3f} us/payment  ({1e6/us:12,.0f}/s)")
    return rows


def crossover(rows):
    """For each (n_jobs, batch), which path is faster and by how much."""
    out = []
    key = lambda r: (r["n_jobs"], r["batch"])
    cpu = {key(r): r for r in rows if r["format"] == "compress_trees"}
    nnpa = {key(r): r for r in rows if r["format"] == "zdnn_tensors"}
    for k in sorted(cpu):
        if k in nnpa:
            ratio = cpu[k]["us_per_payment"] / nnpa[k]["us_per_payment"]
            out.append({"n_jobs": k[0], "batch": k[1], "nnpa_speedup": round(ratio, 3),
                        "faster": "NNPA" if ratio > 1.1 else "CPU" if ratio < 0.9 else "tie (within 10%)"})
    return out


def plot(rows):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    styles = {"compress_trees": ("#0f62fe", "CPU compressed trees"), "zdnn_tensors": ("#da1e28", "Telum II NNPA (zDNN)")}
    for fmt, (color, label) in styles.items():
        for nj, ls in zip(THREADS, (":", "--", "-")):
            pts = sorted((r["batch"], r["us_per_payment"]) for r in rows if r["format"] == fmt and r["n_jobs"] == nj)
            if pts:
                ax.plot(*zip(*pts), ls, color=color, marker="o", ms=3, label=f"{label}, {nj} thread{'s' * (nj > 1)}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("payments per scoring call (batch size)"); ax.set_ylabel("microseconds per payment (lower is better)")
    ax.set_title(f"Snap ML tree inference on {platform.machine()}: CPU vs Telum II accelerator")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); FIG.parent.mkdir(exist_ok=True); fig.savefig(FIG, dpi=150)
    log(f"saved {FIG}")


def shapes(d):
    from snapml import BoostingMachineClassifier
    X = d["Xte"]
    out = []
    for rounds, depth in ((50, 4), (100, 6), (300, 8)):
        m = BoostingMachineClassifier(num_round=rounds, max_depth=depth, learning_rate=0.1, n_jobs=3,
                                      class_weight="balanced", random_state=42)
        t0 = time.time(); m.fit(d["Xtr"], d["ytr"]); fit_s = time.time() - t0
        rows = sweep(pickle.dumps(m), X, batches=(1, 2048, None), threads=(3,))
        for r in rows:
            r.update(trees=rounds, depth=depth, fit_s=round(fit_s, 1))
        out += rows
    return out


def main():
    d = np.load(DATA)
    X = d["Xte"]
    model_bytes = MODEL.read_bytes()
    res = {"machine": platform.machine(), "test_payments": int(len(X))}
    res["sweep"] = sweep(model_bytes, X)
    res["crossover"] = crossover(res["sweep"])
    if "--shapes" in sys.argv:
        res["shapes"] = shapes(d)
        res["shapes_crossover"] = []
        for t, dep in ((50, 4), (100, 6), (300, 8)):
            sub = [r for r in res["shapes"] if r["trees"] == t]
            for c in crossover(sub):
                res["shapes_crossover"].append({"trees": t, "depth": dep, **c})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    plot(res["sweep"])
    wins = [c for c in res["crossover"] + res.get("shapes_crossover", []) if c["faster"] == "NNPA"]
    log("\nNNPA faster than CPU in: " + (json.dumps(wins) if wins else "no tested configuration"))
    best = max(res["crossover"], key=lambda c: c["nnpa_speedup"])
    log(f"closest to crossover: {json.dumps(best)}")


if __name__ == "__main__":
    main()
