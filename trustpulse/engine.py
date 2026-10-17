"""TrustPulse real-time decision engine.

One payment in, one decision out. The engine keeps a running profile for every
account and every sender/receiver pair, computes the same 27 features the
notebook builds in batch, scores them with the trained gradient-boosted trees,
and returns APPROVE / VERIFY / REVIEW with reasons taken from the model itself.

Dependencies: Python 3 and NumPy only. No pandas, no scikit-learn at serving
time: the trees are exported to plain arrays (see ``TreeEnsemble``), so the
service runs anywhere NumPy runs, including s390x.

Feature semantics match notebook cell "streaming account features" exactly:
every running total covers strictly earlier activity, and a self-transfer's
receiver view includes the sender-side event of the same payment, as in the
batch event table.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ACCT = ["in_n", "out_n", "in_amt", "out_amt", "in_uniq", "out_uniq", "age_h", "gap_h"]
LOCAL = ["log_amt", "hour", "fmt_code", "cur_code", "cross_currency", "self_txn"]
NET = (["pair_prev_n", "new_counterparty", "src_passthrough", "dst_passthrough", "amt_vs_src_avg"]
       + [f"{s}_{a}" for s in ("src", "dst") for a in ACCT])
FEATURES = LOCAL + NET

GROUPS = {
    "receiver inflow":  ["dst_in_n", "dst_in_uniq", "dst_in_amt"],
    "receiver outflow": ["dst_out_n", "dst_out_uniq", "dst_out_amt", "dst_passthrough"],
    "sender inflow":    ["src_in_n", "src_in_uniq", "src_in_amt"],
    "sender outflow":   ["src_out_n", "src_out_uniq", "src_out_amt", "src_passthrough"],
    "account timing":   ["src_age_h", "dst_age_h", "src_gap_h", "dst_gap_h"],
    "relationship":     ["pair_prev_n", "new_counterparty"],
    "amount":           ["log_amt", "amt_vs_src_avg"],
    "channel":          ["fmt_code", "cur_code", "cross_currency"],
    "time of day":      ["hour"],
    "self transfer":    ["self_txn"],
}

# Pair keys pack (sender, receiver) into one integer. 2**21 leaves room for
# about two million accounts, four times the IBM HI-Small population.
PAIR_SHIFT = 1 << 21


# --------------------------------------------------------------------------
# Account and pair state
# --------------------------------------------------------------------------

class StreamState:
    """Running per-account and per-pair profile, updated one payment at a time.

    Times are integer minutes since the Unix epoch. Accounts are integer ids;
    an id the state has never seen is treated as a brand-new account.
    """

    def __init__(self, n_accounts: int = 0, pair_keys=None, pair_counts=None):
        n = int(n_accounts)
        self.in_n = [0] * n
        self.out_n = [0] * n
        self.in_amt = [0.0] * n
        self.out_amt = [0.0] * n
        self.in_uniq = [0] * n
        self.out_uniq = [0] * n
        self.first_ts = [-1] * n
        self.last_ts = [-1] * n
        # Pairs seen before the stream started live in two sorted arrays
        # (compact, binary-searched); pairs touched during the stream live in a dict.
        self.base_keys = np.asarray(pair_keys if pair_keys is not None else [], dtype=np.int64)
        self.base_counts = np.asarray(pair_counts if pair_counts is not None else [], dtype=np.int64)
        self.live_pairs: dict[int, int] = {}

    # ---- persistence -------------------------------------------------------
    def save(self, path) -> None:
        np.savez_compressed(
            path,
            in_n=np.asarray(self.in_n, dtype=np.int64), out_n=np.asarray(self.out_n, dtype=np.int64),
            in_amt=np.asarray(self.in_amt, dtype=np.float64), out_amt=np.asarray(self.out_amt, dtype=np.float64),
            in_uniq=np.asarray(self.in_uniq, dtype=np.int64), out_uniq=np.asarray(self.out_uniq, dtype=np.int64),
            first_ts=np.asarray(self.first_ts, dtype=np.int64), last_ts=np.asarray(self.last_ts, dtype=np.int64),
            pair_keys=self.base_keys, pair_counts=self.base_counts,
        )

    @classmethod
    def load(cls, path) -> "StreamState":
        z = np.load(path)
        st = cls(0, z["pair_keys"], z["pair_counts"])
        st.in_n, st.out_n = z["in_n"].tolist(), z["out_n"].tolist()
        st.in_amt, st.out_amt = z["in_amt"].tolist(), z["out_amt"].tolist()
        st.in_uniq, st.out_uniq = z["in_uniq"].tolist(), z["out_uniq"].tolist()
        st.first_ts, st.last_ts = z["first_ts"].tolist(), z["last_ts"].tolist()
        return st

    @property
    def n_accounts(self) -> int:
        return len(self.in_n)

    def n_pairs(self) -> int:
        return len(self.base_keys) + sum(1 for k in self.live_pairs if not self._in_base(k))

    # ---- internals ---------------------------------------------------------
    def _grow(self, acct: int) -> None:
        extra = acct + 1 - len(self.in_n)
        if extra > 0:
            for lst, fill in ((self.in_n, 0), (self.out_n, 0), (self.in_amt, 0.0), (self.out_amt, 0.0),
                              (self.in_uniq, 0), (self.out_uniq, 0), (self.first_ts, -1), (self.last_ts, -1)):
                lst.extend([fill] * extra)

    def _in_base(self, key: int) -> bool:
        i = int(np.searchsorted(self.base_keys, key))
        return i < len(self.base_keys) and int(self.base_keys[i]) == key

    def pair_count(self, src: int, dst: int) -> int:
        key = src * PAIR_SHIFT + dst
        c = self.live_pairs.get(key)
        if c is not None:
            return c
        keys = self.base_keys
        i = int(np.searchsorted(keys, key))
        if i < len(keys) and int(keys[i]) == key:
            return int(self.base_counts[i])
        return 0

    def _profile(self, a: int):
        if a < len(self.in_n):
            return [self.in_n[a], self.out_n[a], self.in_amt[a], self.out_amt[a],
                    self.in_uniq[a], self.out_uniq[a], self.first_ts[a], self.last_ts[a]]
        return [0, 0, 0.0, 0.0, 0, 0, -1, -1]

    # ---- the two public operations ------------------------------------------
    def features(self, ts: int, src: int, dst: int, amt_usd: float,
                 fmt_code: int, cur_code: int, cross_currency: int) -> list:
        """Feature vector for a payment, in FEATURES order. Does not change state."""
        prev = self.pair_count(src, dst)
        first = 1 if prev == 0 else 0

        s = self._profile(src)
        s_age = 0.0 if s[6] < 0 else (ts - s[6]) / 60.0
        s_gap = -1.0 if s[7] < 0 else (ts - s[7]) / 60.0

        d = self._profile(dst)
        if dst == src:
            # The batch event table orders a payment's sender event before its
            # receiver event, so a self-transfer's receiver view already includes it.
            d[1] += 1
            d[3] += amt_usd
            d[5] += first
            if d[6] < 0:
                d[6] = ts
            d[7] = ts
        d_age = 0.0 if d[6] < 0 else (ts - d[6]) / 60.0
        d_gap = -1.0 if d[7] < 0 else (ts - d[7]) / 60.0

        hour = (ts // 60) % 24
        return [
            math.log1p(amt_usd), hour, fmt_code, cur_code, cross_currency, 1 if src == dst else 0,
            prev, first,
            s[3] / (s[2] + 1),
            d[3] / (d[2] + 1),
            amt_usd / (s[3] / max(s[1], 1) + 1),
            s[0], s[1], s[2], s[3], s[4], s[5], s_age, s_gap,
            d[0], d[1], d[2], d[3], d[4], d[5], d_age, d_gap,
        ]

    def update(self, ts: int, src: int, dst: int, amt_usd: float) -> None:
        """Record a payment that has been processed."""
        key = src * PAIR_SHIFT + dst
        prev = self.pair_count(src, dst)
        first = 1 if prev == 0 else 0
        self._grow(max(src, dst))

        self.out_n[src] += 1
        self.out_amt[src] += amt_usd
        self.out_uniq[src] += first
        if self.first_ts[src] < 0:
            self.first_ts[src] = ts
        self.last_ts[src] = ts

        self.in_n[dst] += 1
        self.in_amt[dst] += amt_usd
        self.in_uniq[dst] += first
        if self.first_ts[dst] < 0:
            self.first_ts[dst] = ts
        self.last_ts[dst] = ts

        self.live_pairs[key] = prev + 1


# --------------------------------------------------------------------------
# Tree ensemble exported from scikit-learn
# --------------------------------------------------------------------------

class TreeEnsemble:
    """HistGradientBoostingClassifier trees as flat arrays.

    All trees are walked together, one level per step, so scoring one payment
    or a small batch costs the same handful of NumPy operations. Leaves loop
    back to themselves, so a fixed number of steps (the deepest tree) is enough.
    """

    def __init__(self, baseline, roots, feature, threshold, missing_left, left, right, value, depth):
        self.baseline = float(baseline)
        self.roots = np.asarray(roots, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float64)
        self.missing_left = np.asarray(missing_left, dtype=bool)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.value = np.asarray(value, dtype=np.float64)
        self.depth = int(depth)

    @classmethod
    def from_sklearn(cls, model) -> "TreeEnsemble":
        if getattr(model, "n_trees_per_iteration_", 1) != 1:
            raise ValueError("only binary classifiers are supported")
        roots, feature, threshold, missing_left, left, right, value = [], [], [], [], [], [], []
        depth = 0
        offset = 0
        for (pred,) in model._predictors:
            nodes = pred.nodes
            if np.any(nodes["is_categorical"]):
                raise ValueError("categorical splits are not supported")
            n = len(nodes)
            idx = np.arange(n) + offset
            leaf = nodes["is_leaf"].astype(bool)
            roots.append(offset)
            feature.append(np.where(leaf, 0, nodes["feature_idx"]))
            # Leaves: threshold +inf sends every value left, and left points at the leaf itself.
            threshold.append(np.where(leaf, np.inf, nodes["num_threshold"]))
            missing_left.append(np.where(leaf, True, nodes["missing_go_to_left"].astype(bool)))
            left.append(np.where(leaf, idx, nodes["left"] + offset))
            right.append(np.where(leaf, idx, nodes["right"] + offset))
            value.append(np.where(leaf, nodes["value"], 0.0))
            depth = max(depth, int(nodes["depth"].max()))
            offset += n
        baseline = float(np.ravel(model._baseline_prediction)[0])
        return cls(baseline, roots, np.concatenate(feature), np.concatenate(threshold),
                   np.concatenate(missing_left), np.concatenate(left), np.concatenate(right),
                   np.concatenate(value), depth)

    def to_json(self, path) -> None:
        payload = {
            "format": "trustpulse-trees-v1", "baseline": self.baseline, "depth": self.depth,
            "roots": self.roots.tolist(), "feature": self.feature.tolist(),
            "threshold": [None if math.isinf(t) else t for t in self.threshold.tolist()],
            "missing_left": self.missing_left.astype(int).tolist(),
            "left": self.left.tolist(), "right": self.right.tolist(), "value": self.value.tolist(),
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def from_json(cls, path) -> "TreeEnsemble":
        p = json.loads(Path(path).read_text())
        thr = [math.inf if t is None else t for t in p["threshold"]]
        return cls(p["baseline"], p["roots"], p["feature"], thr, p["missing_left"],
                   p["left"], p["right"], p["value"], p["depth"])

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    @property
    def n_nodes(self) -> int:
        return len(self.feature)

    def raw(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        rows = np.arange(X.shape[0])[:, None]
        node = np.broadcast_to(self.roots, (X.shape[0], len(self.roots))).copy()
        for _ in range(self.depth):
            x = X[rows, self.feature[node]]
            go_left = np.where(np.isnan(x), self.missing_left[node], x <= self.threshold[node])
            node = np.where(go_left, self.left[node], self.right[node])
        return self.baseline + self.value[node].sum(axis=1)

    def proba(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw(X)))


# --------------------------------------------------------------------------
# Decisions and reasons
# --------------------------------------------------------------------------

def _pct(x):
    return f"{x:.0%}"


def reason_text(group: str, f: dict, txn: dict) -> str:
    if group == "receiver inflow":
        return f"receiver already took {f['dst_in_n']:.0f} payments from {f['dst_in_uniq']:.0f} accounts"
    if group == "receiver outflow":
        return f"receiver has forwarded {_pct(f['dst_passthrough'])} of what it received"
    if group == "sender inflow":
        return f"sender took {f['src_in_n']:.0f} payments from {f['src_in_uniq']:.0f} accounts"
    if group == "sender outflow":
        return (f"sender paid {f['src_out_uniq']:.0f} receivers, "
                f"forwarding {_pct(f['src_passthrough'])} of inflow")
    if group == "account timing":
        return (f"accounts active {f['src_age_h']:.0f} h / {f['dst_age_h']:.0f} h, "
                f"sender idle {max(f['src_gap_h'], 0):.1f} h")
    if group == "relationship":
        return ("first payment between these accounts" if f["new_counterparty"]
                else f"{f['pair_prev_n']:.0f} earlier payments between them")
    if group == "amount":
        ratio = f["amt_vs_src_avg"]
        size = f"{ratio:.1f}x sender's average" if ratio >= 1 else "below sender's usual size"
        return f"${txn['amt_usd']:,.0f}, {size}"
    if group == "channel":
        return f"{txn['fmt']} in {txn['currency']}" + (" (cross-currency)" if f["cross_currency"] else "")
    if group == "time of day":
        return f"sent at {int(f['hour']):02d}:00"
    if group == "self transfer":
        return "transfer to own account"
    return group


@dataclass
class Decision:
    decision: str
    score: float
    reasons: list = field(default_factory=list)
    drivers: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class DecisionEngine:
    """State + model + thresholds + explanations behind one ``decide`` call."""

    def __init__(self, state: StreamState, model: TreeEnsemble, config: dict):
        self.state = state
        self.model = model
        self.cfg = config
        self.features = config["features"]
        if self.features != FEATURES:
            raise ValueError("model feature order does not match the engine")
        self.thr_review = float(config["thr_review"])
        self.thr_verify = float(config["thr_verify"])
        self.fmt_codes = {name: i for i, name in enumerate(config["fmt_categories"])}
        self.cur_codes = {name: i for i, name in enumerate(config["cur_categories"])}
        self.fx = config["fx_to_usd"]
        med = config["medians"]
        idx = {f: i for i, f in enumerate(FEATURES)}
        self._group_idx = {g: [idx[c] for c in cols] for g, cols in GROUPS.items()}
        self._group_med = {g: [float(med[c]) for c in cols] for g, cols in GROUPS.items()}

    @classmethod
    def load(cls, model_dir, state_path) -> "DecisionEngine":
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "engine_config.json").read_text())
        model = TreeEnsemble.from_json(model_dir / "model_trees.json")
        return cls(StreamState.load(state_path), model, cfg)

    def to_usd(self, amount: float, currency: str) -> float:
        return float(self.fx.get(currency, 1.0)) * float(amount)

    def band(self, score: float) -> str:
        if score >= self.thr_review:
            return "REVIEW"
        if score >= self.thr_verify:
            return "VERIFY"
        return "APPROVE"

    def explain(self, x: list, score: float):
        """Score drop when each feature group is reset to its typical value."""
        rows = [list(x)]
        for g, cols in self._group_idx.items():
            r = list(x)
            for c, v in zip(cols, self._group_med[g]):
                r[c] = v
            rows.append(r)
        p = self.model.proba(rows)
        return {g: float(score - p[i + 1]) for i, g in enumerate(self._group_idx)}

    def decide(self, txn: dict, commit: bool = True, explain: bool = True) -> Decision:
        """txn keys: ts (epoch minutes), src, dst, amt_usd, fmt, currency, cross_currency."""
        t0 = time.perf_counter()
        x = self.state.features(txn["ts"], txn["src"], txn["dst"], txn["amt_usd"],
                                txn.get("fmt_code", self.fmt_codes.get(txn.get("fmt"), -1)),
                                txn.get("cur_code", self.cur_codes.get(txn.get("currency"), -1)),
                                int(txn.get("cross_currency", 0)))
        score = float(self.model.proba(x)[0])
        out = Decision(self.band(score), score)
        if explain and out.decision != "APPROVE":
            drops = self.explain(x, score)
            out.drivers = drops
            f = dict(zip(FEATURES, x))
            out.features = f
            top = [g for g, v in sorted(drops.items(), key=lambda kv: -kv[1]) if v > 0.01][:3]
            out.reasons = [reason_text(g, f, txn) for g in top] or ["several weak signals together"]
        if commit:
            self.state.update(txn["ts"], txn["src"], txn["dst"], txn["amt_usd"])
        out.latency_ms = (time.perf_counter() - t0) * 1000
        return out
