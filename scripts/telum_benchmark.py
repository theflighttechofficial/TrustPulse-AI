"""Benchmark tree-ensemble inference on the CPU and on the IBM Telum on-chip AI
accelerator (NNPA, through IBM zDNN) with IBM Snap ML.

Runs outside Jupyter so a crash in a native library cannot take the notebook
kernel down, and each inference path runs in its own process.

    python scripts/telum_benchmark.py train              # train once, cache the model
    python scripts/telum_benchmark.py bench compress_trees
    python scripts/telum_benchmark.py bench zdnn_tensors
    python scripts/telum_benchmark.py compare            # accuracy + agreement summary

Input: artifacts/telum_data.npz (Xtr, ytr, Xva, yva, Xte, yte), written by the notebook.
Output: results/telum_benchmark.json
"""

import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "telum_data.npz"
MODEL = ROOT / "artifacts" / "snapml_model.pkl"
OUT = ROOT / "results" / "telum_benchmark.json"
SCORES = ROOT / "artifacts"


def log(msg):
    print(msg, flush=True)


def load():
    z = np.load(DATA)
    return {k: z[k] for k in z.files}


def results():
    return json.loads(OUT.read_text()) if OUT.exists() else {"machine": platform.machine()}


def save(res):
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))


def train():
    from snapml import BoostingMachineClassifier
    d = load()
    model = BoostingMachineClassifier(num_round=300, max_depth=8, learning_rate=0.1, n_jobs=3,
                                      class_weight="balanced", random_state=42, verbose=False)
    t0 = time.time()
    model.fit(d["Xtr"], d["ytr"])
    secs = time.time() - t0
    log(f"trained in {secs:.1f}s on {platform.machine()}")
    with open(MODEL, "wb") as f:
        pickle.dump(model, f)
    res = results()
    res["train_seconds"] = round(secs, 1)
    res["train_rows"] = int(len(d["ytr"]))
    save(res)


def bench(fmt):
    import inspect
    d = load()
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    Xva, Xte = d["Xva"], d["Xte"]
    x1, xb = Xte[:1], Xte[:10000]

    sig = inspect.signature(model.optimize_trees)
    log(f"optimize_trees{sig}")
    kw = {"tree_format": fmt}
    if "X" in sig.parameters:
        kw["X"] = xb
    log(f"[{fmt}] optimize_trees ...")
    t0 = time.time()
    model.optimize_trees(**kw)
    prep = time.time() - t0
    log(f"[{fmt}] prepared in {prep:.2f}s; scoring ...")

    model.predict_proba(xb)
    t0 = time.perf_counter()
    p_te = model.predict_proba(Xte, n_jobs=3)[:, 1]
    full = time.perf_counter() - t0
    p_va = model.predict_proba(Xva, n_jobs=3)[:, 1]
    t0 = time.perf_counter()
    for _ in range(10):
        model.predict_proba(xb, n_jobs=3)
    batch = (time.perf_counter() - t0) / 10
    t0 = time.perf_counter()
    for _ in range(300):
        model.predict_proba(x1, n_jobs=1)
    single = (time.perf_counter() - t0) / 300

    np.savez(SCORES / f"scores_{fmt}.npz", p_te=p_te, p_va=p_va)
    row = {"prepare_s": round(prep, 3), "all_test_s": round(full, 3), "test_payments": int(len(Xte)),
           "payments_per_s": round(len(Xte) / full), "batch_10k_ms": round(batch * 1e3, 2),
           "one_payment_ms": round(single * 1e3, 4)}
    log(f"[{fmt}] {json.dumps(row)}")
    res = results()
    res.setdefault("inference", {})[fmt] = row
    save(res)


def compare():
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = load()
    yte = d["yte"]
    res = results()
    acc, scores = {}, {}
    for fmt in ("compress_trees", "zdnn_tensors"):
        path = SCORES / f"scores_{fmt}.npz"
        if not path.exists():
            continue
        s = np.load(path)
        scores[fmt] = s
        thr = np.quantile(s["p_va"], 1 - 0.005)
        a = s["p_te"] >= thr
        acc[fmt] = {"pr_auc": round(float(average_precision_score(yte, s["p_te"])), 4),
                    "roc_auc": round(float(roc_auc_score(yte, s["p_te"])), 4),
                    "alert_rate": round(float(a.mean()), 5),
                    "recall": round(float(yte[a].sum() / yte.sum()), 4),
                    "precision": round(float(yte[a].mean()), 4)}
    res["accuracy_test"] = acc
    if len(scores) == 2:
        c, z = scores["compress_trees"], scores["zdnn_tensors"]
        thr = np.quantile(c["p_va"], 1 - 0.005)
        res["cpu_vs_nnpa"] = {"max_score_diff": float(np.abs(c["p_te"] - z["p_te"]).max()),
                              "same_review_decision": float(((c["p_te"] >= thr) == (z["p_te"] >= thr)).mean())}
        cpu, nnpa = res["inference"]["compress_trees"], res["inference"]["zdnn_tensors"]
        res["speedup_nnpa_vs_cpu"] = {k: round(cpu[k] / nnpa[k], 2)
                                      for k in ("all_test_s", "batch_10k_ms", "one_payment_ms")}
    save(res)
    log(json.dumps(res, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        train()
    elif cmd == "bench" and len(sys.argv) > 2:
        bench(sys.argv[2])
    elif cmd == "compare":
        compare()
    else:
        sys.exit(__doc__)
