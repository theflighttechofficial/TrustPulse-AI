"""Build a small synthetic engine bundle so the live console can be tried
without the IBM dataset (and so CI can smoke-test the server).

    python tests/make_synthetic_bundle.py /tmp/tp_demo
    python app/server.py --model-dir /tmp/tp_demo/models --artifacts /tmp/tp_demo/engine
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trustpulse.build import batch_features, export_bundle, parity_check  # noqa: E402
from trustpulse.engine import FEATURES  # noqa: E402


def make(out: Path, n=60000, n_accounts=4000, seed=1):
    rng = np.random.default_rng(seed)
    minutes = np.sort(rng.integers(0, 60 * 24 * 10, n))
    src = rng.integers(0, n_accounts, n)
    dst = rng.integers(0, n_accounts, n)
    pattern_id = np.full(n, -1)
    ptypes = ["FAN-IN", "FAN-OUT", "CYCLE", "SCATTER-GATHER"]
    # Plant laundering chains: a mule account collects then forwards.
    pid = 0
    for start in rng.choice(np.arange(500, n - 400), 120, replace=False):
        mule = rng.integers(0, n_accounts)
        idx = np.sort(rng.choice(np.arange(start, start + 300), 8, replace=False))
        src[idx[:5]] = rng.integers(0, n_accounts, 5)
        dst[idx[:5]] = mule
        src[idx[5:]] = mule
        dst[idx[5:]] = rng.integers(0, n_accounts, 3)
        pattern_id[idx] = pid
        pid += 1
    cats_fmt = pd.Categorical(rng.choice(["ACH", "Cheque", "Credit Card", "Wire"], n))
    cats_cur = pd.Categorical(rng.choice(["Euro", "US Dollar"], n, p=[0.2, 0.8]))
    amt_paid = np.round(rng.lognormal(7, 1.5, n), 2)
    rate = np.where(cats_cur == "Euro", 1.17, 1.0)
    df = pd.DataFrame({
        "ts": pd.Timestamp("2022-09-01") + pd.to_timedelta(minutes, unit="m"),
        "src": src, "dst": dst, "amt_paid": amt_paid, "amt_usd": amt_paid * rate,
        "fmt": cats_fmt, "cur_paid": cats_cur, "cross_currency": (rng.random(n) < 0.05).astype(int),
        "pattern_id": pattern_id,
    })
    df["is_laundering"] = (df.pattern_id >= 0).astype(int)
    df["pattern_type"] = np.where(df.pattern_id >= 0, np.array(ptypes)[df.pattern_id.clip(0) % 4], None)
    df["fmt_code"] = df.fmt.cat.codes
    df["cur_code"] = df.cur_paid.cat.codes
    feats = batch_features(df)
    df = pd.concat([df, feats.drop(columns=["fmt_code", "cur_code", "cross_currency"])], axis=1)

    tr = df[df.ts < "2022-09-07"]
    va = df[(df.ts >= "2022-09-07") & (df.ts < "2022-09-09")]
    model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, class_weight="balanced",
                                           early_stopping=False, random_state=0).fit(tr[FEATURES], tr.is_laundering)
    p_va = model.predict_proba(va[FEATURES])[:, 1]
    starts = df[df.pattern_id >= 0].groupby("pattern_id").ts.min()
    export_bundle(df, model, FEATURES, tr[FEATURES].median(), np.quantile(p_va, 0.995), np.quantile(p_va, 0.98),
                  "2022-09-09", out / "models", out / "engine", spotlight_pattern=int(df[df.ts >= "2022-09-09"].pattern_id.max()),
                  pattern_starts=starts)
    parity_check(out / "models", out / "engine", df[df.ts >= "2022-09-09"], model, n=3000)


if __name__ == "__main__":
    make(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tp_demo"))
