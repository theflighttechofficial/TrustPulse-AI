"""The streaming engine must reproduce the batch notebook pipeline.

Run: python -m pytest -q tests
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from trustpulse.build import batch_features, build_state, ts_minutes
from trustpulse.engine import FEATURES, DecisionEngine, StreamState, TreeEnsemble


def synthetic_payments(n=6000, n_accounts=300, seed=7):
    rng = np.random.default_rng(seed)
    minutes = np.sort(rng.integers(0, 60 * 24 * 6, n))
    src = rng.integers(0, n_accounts, n)
    dst = rng.integers(0, n_accounts, n)
    # Some repeat pairs, some self-transfers, some same-minute payments.
    repeat = rng.random(n) < 0.3
    dst[repeat] = (src[repeat] * 7 + 3) % n_accounts
    self_txn = rng.random(n) < 0.05
    dst[self_txn] = src[self_txn]
    return pd.DataFrame({
        "ts": pd.Timestamp("2022-09-01") + pd.to_timedelta(minutes, unit="m"),
        "src": src, "dst": dst,
        "amt_usd": np.round(rng.lognormal(7, 2, n), 2),
        "fmt_code": rng.integers(0, 7, n), "cur_code": rng.integers(0, 15, n),
        "cross_currency": (rng.random(n) < 0.1).astype(int),
    })


def stream(state, df):
    rows = []
    for r in df.itertuples(index=False):
        ts = int(pd.Timestamp(r.ts).value // 60_000_000_000)
        rows.append(state.features(ts, r.src, r.dst, r.amt_usd, r.fmt_code, r.cur_code, r.cross_currency))
        state.update(ts, r.src, r.dst, r.amt_usd)
    return np.asarray(rows, dtype=np.float64)


def assert_close(a, b):
    rel = np.abs(a - b) / np.maximum(1.0, np.abs(b))
    worst = pd.Series(rel.max(axis=0), index=FEATURES).idxmax()
    assert rel.max() < 1e-9, f"max relative diff {rel.max():.3g} on {worst}"


def test_stream_from_empty_matches_batch():
    df = synthetic_payments()
    assert_close(stream(StreamState(0), df), batch_features(df).to_numpy(np.float64))


def test_warm_state_then_stream_matches_batch():
    df = synthetic_payments(seed=11)
    cut = df.ts.iloc[len(df) * 2 // 3]
    hist, live = df[df.ts < cut], df[df.ts >= cut]
    state = build_state(hist, n_accounts=300)
    expected = batch_features(df).to_numpy(np.float64)[len(hist):]
    assert_close(stream(state, live), expected)


def test_state_roundtrip(tmp_path):
    df = synthetic_payments(n=2000, seed=3)
    st = build_state(df, 300)
    st.save(tmp_path / "s.npz")
    st2 = StreamState.load(tmp_path / "s.npz")
    probe = synthetic_payments(n=300, seed=4)
    assert np.array_equal(stream(st, probe), stream(st2, probe))


@pytest.fixture(scope="module")
def trained():
    df = synthetic_payments(n=8000, seed=5)
    X = batch_features(df)
    rng = np.random.default_rng(0)
    logit = 0.002 * X.src_out_n - 1.5 * X.new_counterparty + 0.3 * X.log_amt - 4 + rng.normal(0, 1, len(X))
    y = (logit > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=31, random_state=0).fit(X, y)
    return model, X


def test_tree_export_matches_sklearn(trained, tmp_path):
    model, X = trained
    trees = TreeEnsemble.from_sklearn(model)
    trees.to_json(tmp_path / "m.json")
    trees = TreeEnsemble.from_json(tmp_path / "m.json")
    p_sk = model.predict_proba(X)[:, 1]
    assert np.abs(trees.proba(X.to_numpy()) - p_sk).max() < 1e-12
    for i in range(0, len(X), 997):                      # single-payment path
        assert abs(trees.proba(X.iloc[i].to_numpy())[0] - p_sk[i]) < 1e-12


def test_decision_engine_end_to_end(trained):
    model, X = trained
    p = model.predict_proba(X)[:, 1]
    cfg = {
        "features": FEATURES, "thr_review": float(np.quantile(p, 0.99)), "thr_verify": float(np.quantile(p, 0.95)),
        "fmt_categories": [f"F{i}" for i in range(7)], "cur_categories": [f"C{i}" for i in range(15)],
        "fx_to_usd": {"C0": 1.0}, "medians": X.median().to_dict(),
    }
    eng = DecisionEngine(StreamState(0), TreeEnsemble.from_sklearn(model), cfg)
    df = synthetic_payments(n=8000, seed=5)
    seen = {"APPROVE": 0, "VERIFY": 0, "REVIEW": 0}
    agree, off = 0, 0
    for i, r in enumerate(df.itertuples(index=False)):
        ts = int(pd.Timestamp(r.ts).value // 60_000_000_000)
        d = eng.decide({"ts": ts, "src": r.src, "dst": r.dst, "amt_usd": r.amt_usd, "fmt_code": r.fmt_code,
                        "cur_code": r.cur_code, "cross_currency": r.cross_currency,
                        "fmt": f"F{r.fmt_code}", "currency": f"C{r.cur_code}"})
        # Running sums differ from the batch "cumsum minus current" in the last bits,
        # which can tip a value sitting exactly on a split threshold.
        off += abs(d.score - p[i]) > 1e-9
        agree += d.decision == eng.band(p[i])
        seen[d.decision] += 1
        if d.decision != "APPROVE":
            assert d.reasons and set(d.drivers) == {
                "receiver inflow", "receiver outflow", "sender inflow", "sender outflow", "account timing",
                "relationship", "amount", "channel", "time of day", "self transfer"}
    assert off / len(df) < 0.005, f"{off} payments scored differently"
    assert agree / len(df) > 0.998
    assert seen["REVIEW"] > 0 and seen["VERIFY"] > 0


def test_new_accounts_grow_state():
    st = StreamState(5)
    x = st.features(1000, 50, 51, 10.0, 0, 0, 0)
    assert x[FEATURES.index("src_age_h")] == 0.0 and x[FEATURES.index("src_gap_h")] == -1.0
    st.update(1000, 50, 51, 10.0)
    assert st.n_accounts == 52 and st.pair_count(50, 51) == 1
