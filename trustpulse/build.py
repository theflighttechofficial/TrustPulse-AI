"""Notebook-side helpers: export the trained model to the real-time engine and
prove the engine reproduces the batch pipeline.

Needs pandas and scikit-learn (the notebook environment). The serving side
(``engine.py`` and ``app/``) does not.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import ACCT, FEATURES, PAIR_SHIFT, DecisionEngine, StreamState, TreeEnsemble


def ts_minutes(ts: pd.Series) -> np.ndarray:
    """Timestamps as integer minutes since the Unix epoch."""
    return ts.values.astype("datetime64[m]").astype(np.int64)


# --------------------------------------------------------------------------
# Reference batch implementation (same logic as the notebook feature cell)
# --------------------------------------------------------------------------

def batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Batch feature build. ``df`` needs ts, src, dst, amt_usd, fmt_code, cur_code,
    cross_currency and must be in arrival order. Used by the tests to check the
    streaming engine against the notebook's vectorised pipeline."""
    aml = df.reset_index(drop=True).copy()
    aml["txn_id"] = np.arange(len(aml))
    n = len(aml)
    first_pair = ~aml.duplicated(["src", "dst"])
    aml["pair_prev_n"] = aml.groupby(["src", "dst"]).cumcount()
    ev = pd.DataFrame({
        "acct": np.concatenate([aml.src.values, aml.dst.values]),
        "txn": np.concatenate([aml.txn_id.values, aml.txn_id.values]),
        "is_in": np.concatenate([np.zeros(n, "int8"), np.ones(n, "int8")]),
        "amt": np.concatenate([aml.amt_usd.values, aml.amt_usd.values]),
        "first": np.concatenate([first_pair.values, first_pair.values]).astype("int8"),
        "ts": np.concatenate([aml.ts.values, aml.ts.values]),
    }).sort_values(["acct", "txn", "is_in"], kind="stable").reset_index(drop=True)

    def excl_cumsum(x):
        return x.groupby(ev.acct, sort=False).cumsum() - x

    is_out = 1 - ev.is_in
    ev["in_n"], ev["out_n"] = excl_cumsum(ev.is_in), excl_cumsum(is_out)
    ev["in_amt"], ev["out_amt"] = excl_cumsum(ev.amt * ev.is_in), excl_cumsum(ev.amt * is_out)
    ev["in_uniq"], ev["out_uniq"] = excl_cumsum(ev["first"] * ev.is_in), excl_cumsum(ev["first"] * is_out)
    g = ev.groupby("acct", sort=False).ts
    ev["age_h"] = (ev.ts - g.transform("min")).dt.total_seconds() / 3600
    ev["gap_h"] = ((ev.ts - g.shift(1)).dt.total_seconds() / 3600).fillna(-1)
    aml = (aml.join(ev[ev.is_in == 0].set_index("txn")[ACCT].add_prefix("src_"), on="txn_id")
              .join(ev[ev.is_in == 1].set_index("txn")[ACCT].add_prefix("dst_"), on="txn_id"))
    aml["log_amt"] = np.log1p(aml.amt_usd)
    aml["hour"] = aml.ts.dt.hour
    aml["self_txn"] = (aml.src == aml.dst).astype(int)
    aml["new_counterparty"] = (aml.pair_prev_n == 0).astype(int)
    aml["src_passthrough"] = aml.src_out_amt / (aml.src_in_amt + 1)
    aml["dst_passthrough"] = aml.dst_out_amt / (aml.dst_in_amt + 1)
    aml["amt_vs_src_avg"] = aml.amt_usd / (aml.src_out_amt / aml.src_out_n.clip(lower=1) + 1)
    return aml[FEATURES]


# --------------------------------------------------------------------------
# Warm state: the engine's memory at the moment the live stream starts
# --------------------------------------------------------------------------

def build_state(history: pd.DataFrame, n_accounts: int) -> StreamState:
    """Account and pair profiles after every payment in ``history`` (vectorised)."""
    src = history.src.to_numpy(np.int64)
    dst = history.dst.to_numpy(np.int64)
    amt = history.amt_usd.to_numpy(np.float64)
    ts = ts_minutes(history.ts)
    n = int(n_accounts)

    keys = src * PAIR_SHIFT + dst
    ukeys, counts = np.unique(keys, return_counts=True)

    first = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    last = np.full(n, -1, dtype=np.int64)
    for acct in (src, dst):
        np.minimum.at(first, acct, ts)
        np.maximum.at(last, acct, ts)
    first[first == np.iinfo(np.int64).max] = -1

    st = StreamState(0, ukeys, counts)
    st.out_n = np.bincount(src, minlength=n).tolist()
    st.in_n = np.bincount(dst, minlength=n).tolist()
    st.out_amt = np.bincount(src, weights=amt, minlength=n).tolist()
    st.in_amt = np.bincount(dst, weights=amt, minlength=n).tolist()
    st.out_uniq = np.bincount(ukeys // PAIR_SHIFT, minlength=n).tolist()
    st.in_uniq = np.bincount(ukeys % PAIR_SHIFT, minlength=n).tolist()
    st.first_ts = first.tolist()
    st.last_ts = last.tolist()
    return st


# --------------------------------------------------------------------------
# Export everything the live service needs
# --------------------------------------------------------------------------

def export_bundle(aml: pd.DataFrame, model, feats, medians, thr_review, thr_verify,
                  stream_start, model_dir, artifact_dir, spotlight_pattern=None,
                  pattern_starts=None) -> dict:
    """Write model_trees.json + engine_config.json (small, safe to commit) and
    engine_state.npz + stream.npz (data-derived, kept out of git)."""
    if list(feats) != FEATURES:
        raise ValueError("notebook feature order differs from the engine's FEATURES")
    model_dir, artifact_dir = Path(model_dir), Path(artifact_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    trees = TreeEnsemble.from_sklearn(model)
    trees.to_json(model_dir / "model_trees.json")

    ok = aml.amt_paid > 0
    fx = (aml.amt_usd[ok] / aml.amt_paid[ok]).groupby(aml.cur_paid[ok], observed=True).median()
    stream_start = pd.Timestamp(stream_start)
    cfg = {
        "features": FEATURES,
        "thr_review": float(thr_review),
        "thr_verify": float(thr_verify),
        "fmt_categories": [str(c) for c in aml.fmt.cat.categories],
        "cur_categories": [str(c) for c in aml.cur_paid.cat.categories],
        "fx_to_usd": {str(k): float(v) for k, v in fx.items()},
        "medians": {k: float(medians[k]) for k in FEATURES},
        "stream_start": str(stream_start),
        "trained_on": "IBM AML HI-Small, 2022-09-01..06 (sampled); thresholds from 2022-09-07..08",
        "built_on": f"{platform.machine()} / {platform.platform()}",
    }
    (model_dir / "engine_config.json").write_text(json.dumps(cfg, indent=2))

    hist = aml[aml.ts < stream_start]
    n_accounts = int(max(aml.src.max(), aml.dst.max())) + 1
    state = build_state(hist, n_accounts)
    state.save(artifact_dir / "engine_state.npz")

    live = aml[aml.ts >= stream_start]
    ptype = live.pattern_type.fillna("").astype(str).to_numpy()
    pids = live.pattern_id.to_numpy(np.int64)
    starts = {}
    if pattern_starts is not None:
        starts = {int(k): int(v) for k, v in zip(pattern_starts.index, ts_minutes(pd.Series(pattern_starts.values)))}
    pat_ids = sorted(set(pids[pids >= 0].tolist()))
    np.savez_compressed(
        artifact_dir / "stream.npz",
        ts=ts_minutes(live.ts), src=live.src.to_numpy(np.int64), dst=live.dst.to_numpy(np.int64),
        amt_usd=live.amt_usd.to_numpy(np.float64), amt_paid=live.amt_paid.to_numpy(np.float64),
        fmt_code=live.fmt.cat.codes.to_numpy(np.int64), cur_code=live.cur_paid.cat.codes.to_numpy(np.int64),
        cross_currency=live.cross_currency.to_numpy(np.int64),
        is_laundering=live.is_laundering.to_numpy(np.int64), pattern_id=pids,
        pattern_ids=np.array(pat_ids, dtype=np.int64),
        pattern_types=np.array([ptype[np.flatnonzero(pids == p)[0]] for p in pat_ids]),
        pattern_starts=np.array([starts.get(p, -1) for p in pat_ids], dtype=np.int64),
        spotlight=np.array([-1 if spotlight_pattern is None else int(spotlight_pattern)]),
    )
    info = {
        "trees": trees.n_trees, "nodes": trees.n_nodes, "max_depth": trees.depth,
        "accounts_in_state": n_accounts, "pairs_in_state": int(len(state.base_keys)),
        "history_payments": int(len(hist)), "stream_payments": int(len(live)),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(info, indent=2))
    return info


# --------------------------------------------------------------------------
# Parity and latency: the live engine against the batch notebook
# --------------------------------------------------------------------------

def parity_check(model_dir, artifact_dir, batch_rows: pd.DataFrame, model, n: int = 20000,
                 out_path=None) -> dict:
    """Stream the first ``n`` live payments through the engine one at a time and
    compare features and scores with the notebook's batch frame and sklearn."""
    artifact_dir = Path(artifact_dir)
    eng = DecisionEngine.load(model_dir, artifact_dir / "engine_state.npz")
    z = np.load(artifact_dir / "stream.npz")
    n = min(n, len(z["ts"]))
    cols = {k: z[k][:n].tolist() for k in ("ts", "src", "dst", "amt_usd", "fmt_code", "cur_code", "cross_currency")}

    feats, lat_feat, lat_score, lat_total = [], [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        x = eng.state.features(cols["ts"][i], cols["src"][i], cols["dst"][i], cols["amt_usd"][i],
                               cols["fmt_code"][i], cols["cur_code"][i], cols["cross_currency"][i])
        t1 = time.perf_counter()
        p = float(eng.model.proba(x)[0])
        band = eng.band(p)
        if band != "APPROVE":
            eng.explain(x, p)
        t2 = time.perf_counter()
        eng.state.update(cols["ts"][i], cols["src"][i], cols["dst"][i], cols["amt_usd"][i])
        t3 = time.perf_counter()
        feats.append(x)
        lat_feat.append((t1 - t0) * 1e3)
        lat_score.append((t2 - t1) * 1e3)
        lat_total.append((t3 - t0) * 1e3)

    F_stream = np.asarray(feats, dtype=np.float64)
    F_batch = batch_rows[FEATURES].to_numpy(np.float64)[:n]
    diff = np.abs(F_stream - F_batch) / np.maximum(1.0, np.abs(F_batch))
    worst = pd.Series(diff.max(axis=0), index=FEATURES).sort_values(ascending=False)

    p_engine = eng.model.proba(F_stream)
    p_sklearn = model.predict_proba(batch_rows[FEATURES].iloc[:n])[:, 1]
    bands_e = np.array([eng.band(v) for v in p_engine])
    bands_s = np.array([eng.band(v) for v in p_sklearn])

    pct = lambda a, q: float(np.percentile(a, q))
    report = {
        "machine": platform.machine(),
        "payments": n,
        "feature_max_rel_diff": float(diff.max()),
        "worst_features": worst.head(3).round(12).to_dict(),
        "score_max_abs_diff_vs_sklearn": float(np.abs(p_engine - p_sklearn).max()),
        "decision_agreement": float((bands_e == bands_s).mean()),
        "latency_ms": {
            "features_p50": round(pct(lat_feat, 50), 4),
            "score_and_explain_p50": round(pct(lat_score, 50), 4),
            "end_to_end_p50": round(pct(lat_total, 50), 4),
            "end_to_end_p99": round(pct(lat_total, 99), 4),
            "end_to_end_max": round(max(lat_total), 4),
        },
        "throughput_payments_per_s": round(n / (sum(lat_total) / 1e3), 1),
    }
    print(json.dumps(report, indent=2))
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2))
    return report
