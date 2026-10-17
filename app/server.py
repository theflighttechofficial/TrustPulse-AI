"""TrustPulse live console: replays the held-out test days through the
real-time engine, one payment at a time, and serves a dashboard plus a REST
scoring endpoint.

Run on the LinuxONE VM:
    python app/server.py            # then open http://localhost:8050 through an SSH tunnel

Needs Flask and NumPy only.
"""

from __future__ import annotations

import argparse
import collections
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trustpulse.engine import DecisionEngine, StreamState  # noqa: E402

RATES = {"25": 25, "100": 100, "500": 500, "max": 0}


def fmt_ts(minutes: int) -> str:
    return datetime.fromtimestamp(minutes * 60, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def platform_info() -> dict:
    info = {
        "machine": platform.machine(),
        "host": platform.node(),
        "os": platform.platform(terse=True),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpus": __import__("os").cpu_count(),
        "cpu": "",
    }
    try:
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
        vendor = next((l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("vendor_id")), "")
        mtype = ""
        for line in text.splitlines():
            if "machine =" in line:
                mtype = line.split("machine =", 1)[1].split()[0].strip(",")
                break
        info["cpu"] = (vendor + (f" machine type {mtype}" if mtype else "")).strip()
        if not info["cpu"]:
            info["cpu"] = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                                if l.startswith("model name")), "")
    except OSError:
        pass
    return info


class Replayer:
    def __init__(self, model_dir: Path, artifact_dir: Path):
        self.model_dir, self.artifact_dir = model_dir, artifact_dir
        self.lock = threading.Lock()
        z = np.load(artifact_dir / "stream.npz")
        self.s = {k: z[k].tolist() for k in ("ts", "src", "dst", "amt_usd", "fmt_code", "cur_code",
                                             "cross_currency", "is_laundering", "pattern_id")}
        self.n = len(self.s["ts"])
        self.pattern_type = dict(zip(z["pattern_ids"].tolist(), z["pattern_types"].tolist()))
        self.pattern_start = dict(zip(z["pattern_ids"].tolist(), z["pattern_starts"].tolist()))
        pid = np.asarray(self.s["pattern_id"])
        ids, first_idx, totals = np.unique(pid[pid >= 0], return_index=True, return_counts=True)
        offset = np.flatnonzero(pid >= 0)
        self.pattern_total = dict(zip(ids.tolist(), totals.tolist()))
        self.pattern_first_idx = dict(zip(ids.tolist(), offset[first_idx].tolist()))
        self.spotlight = int(z["spotlight"][0])
        self.stream_start = self.s["ts"][0] if self.n else 0
        self.engine = None
        self.reset_state()
        self.rate = 100
        self.running = False
        self.ff_target = None
        threading.Thread(target=self._loop, daemon=True).start()

    # ---- state -------------------------------------------------------------
    def reset_state(self):
        t0 = time.time()
        eng = DecisionEngine.load(self.model_dir, self.artifact_dir / "engine_state.npz")
        self.fmt_names = eng.cfg["fmt_categories"]
        self.cur_names = eng.cfg["cur_categories"]
        self.engine = eng
        self.pos = 0
        self.counts = {"APPROVE": 0, "VERIFY": 0, "REVIEW": 0}
        self.truth = {"laundering": 0, "caught_verify": 0, "caught_review": 0, "false_verify": 0, "false_review": 0}
        self.latency = collections.deque(maxlen=5000)
        self.explain_latency = collections.deque(maxlen=2000)
        self.tape = collections.deque(maxlen=40)
        self.alerts = collections.deque(maxlen=400)
        self.alert_seq = 0
        self.patterns = {}
        self.skipped = 0
        self.ticks = collections.deque(maxlen=60)
        self.load_seconds = round(time.time() - t0, 1)

    # ---- one payment -------------------------------------------------------
    def _txn(self, i):
        s = self.s
        return {"ts": s["ts"][i], "src": s["src"][i], "dst": s["dst"][i], "amt_usd": s["amt_usd"][i],
                "fmt_code": s["fmt_code"][i], "cur_code": s["cur_code"][i], "cross_currency": s["cross_currency"][i],
                "fmt": self.fmt_names[s["fmt_code"][i]], "currency": self.cur_names[s["cur_code"][i]]}

    def _process(self, i):
        txn = self._txn(i)
        d = self.engine.decide(txn)
        self.counts[d.decision] += 1
        self.latency.append(d.latency_ms)
        if d.decision != "APPROVE":
            self.explain_latency.append(d.latency_ms)
        laundering = bool(self.s["is_laundering"][i])
        pid = self.s["pattern_id"][i]
        if laundering:
            self.truth["laundering"] += 1
        if d.decision != "APPROVE":
            key = ("caught_" if laundering else "false_") + d.decision.lower()
            self.truth[key] += 1
        row = {"ts": fmt_ts(txn["ts"]), "src": txn["src"], "dst": txn["dst"], "amt_usd": round(txn["amt_usd"], 2),
               "fmt": txn["fmt"], "currency": txn["currency"], "decision": d.decision, "score": round(d.score, 4),
               "latency_ms": round(d.latency_ms, 3), "truth": laundering}
        self.tape.append(row)
        if d.decision != "APPROVE":
            self.alert_seq += 1
            drivers = sorted(d.drivers.items(), key=lambda kv: -kv[1])
            f = d.features
            self.alerts.append({
                **row, "id": self.alert_seq, "reasons": d.reasons,
                "drivers": [[g, round(v, 4)] for g, v in drivers],
                "pattern_type": self.pattern_type.get(pid, "") if pid >= 0 else "",
                "profile": {
                    "sender": {"payments in": f["src_in_n"], "payments out": f["src_out_n"],
                               "distinct senders": f["src_in_uniq"], "distinct receivers": f["src_out_uniq"],
                               "forwarded share": round(f["src_passthrough"], 3), "account age h": round(f["src_age_h"], 1)},
                    "receiver": {"payments in": f["dst_in_n"], "payments out": f["dst_out_n"],
                                 "distinct senders": f["dst_in_uniq"], "distinct receivers": f["dst_out_uniq"],
                                 "forwarded share": round(f["dst_passthrough"], 3), "account age h": round(f["dst_age_h"], 1)},
                    "pair": {"earlier payments": f["pair_prev_n"]},
                },
            })
        if pid >= 0:
            p = self.patterns.get(pid)
            if p is None:
                p = self.patterns[pid] = {
                    "id": pid, "type": self.pattern_type.get(pid, "?"), "total": self.pattern_total.get(pid, 0),
                    "seen": 0, "marks": [], "flag_k": None, "flag_decision": None, "first_ts": fmt_ts(txn["ts"]),
                    "began_before": 0 <= self.pattern_start.get(pid, -1) < self.stream_start,
                }
            p["seen"] += 1
            p["marks"].append(d.decision[0])
            p["last_ts"] = fmt_ts(txn["ts"])
            p["last_index"] = i
            if d.decision != "APPROVE" and p["flag_k"] is None:
                p["flag_k"] = p["seen"]
                p["flag_decision"] = d.decision

    # ---- loop --------------------------------------------------------------
    def _loop(self):
        carry = 0.0
        while True:
            if self.ff_target is not None:
                with self.lock:
                    stop = min(self.ff_target, self.pos + 20000)
                    st, s = self.engine.state, self.s
                    for i in range(self.pos, stop):
                        st.update(s["ts"][i], s["src"][i], s["dst"][i], s["amt_usd"][i])
                    self.skipped += stop - self.pos
                    self.pos = stop
                    if self.pos >= self.ff_target:
                        self.ff_target = None
                        self.running = True
                time.sleep(0)
                continue
            if not self.running or self.pos >= self.n:
                time.sleep(0.05)
                continue
            if self.rate == 0:
                batch = 400
            else:
                carry += self.rate * 0.05
                batch = int(carry)
                carry -= batch
            t0 = time.perf_counter()
            with self.lock:
                end = min(self.n, self.pos + batch)
                for i in range(self.pos, end):
                    self._process(i)
                done = end - self.pos
                self.pos = end
                self.ticks.append((time.time(), done))
            spent = time.perf_counter() - t0
            if self.rate:
                time.sleep(max(0.0, 0.05 - spent))
            else:
                time.sleep(0.001)

    # ---- controls ----------------------------------------------------------
    def jump(self, index: int, lead: int = 150, rate: int = 25):
        target = max(0, index - lead)
        with self.lock:
            self.running = False
            if target < self.pos:
                self.reset_state()
            self.rate = rate
            self.ff_target = target

    def throughput(self):
        now = time.time()
        recent = [(t, n) for t, n in self.ticks if now - t <= 3.0]
        if len(recent) < 2:
            return 0.0
        span = max(now - recent[0][0], 0.05)
        return sum(n for _, n in recent) / span


def create_app(rep: Replayer) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT / "app" / "static"))
    info = platform_info()
    trees = rep.engine.model

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/state")
    def state():
        since = int(request.args.get("since", 0))
        with rep.lock:
            lat = np.asarray(rep.latency) if rep.latency else np.zeros(1)
            exp = np.asarray(rep.explain_latency) if rep.explain_latency else np.zeros(1)
            pats = sorted(rep.patterns.values(), key=lambda p: -p["last_index"])[:14]
            done = [p for p in rep.patterns.values() if p["seen"] >= p["total"] and not p["began_before"]]
            payload = {
                "platform": info,
                "model": {"trees": trees.n_trees, "nodes": trees.n_nodes, "features": len(rep.engine.features)},
                "running": rep.running, "rate": rep.rate,
                "fast_forward": None if rep.ff_target is None else {"pos": rep.pos, "target": rep.ff_target},
                "pos": rep.pos, "total": rep.n, "skipped": rep.skipped,
                "sim_time": fmt_ts(rep.s["ts"][max(rep.pos - 1, 0)]) if rep.n else "",
                "counts": rep.counts, "truth": rep.truth,
                "scored": sum(rep.counts.values()),
                "throughput": round(rep.throughput(), 1),
                "latency": {"p50": round(float(np.percentile(lat, 50)), 3), "p99": round(float(np.percentile(lat, 99)), 3),
                            "alert_p50": round(float(np.percentile(exp, 50)), 3)},
                "tape": list(rep.tape)[-16:][::-1],
                "alerts": [a for a in rep.alerts if a["id"] > since][-60:],
                "patterns": pats,
                "pattern_summary": {
                    "completed": len(done),
                    "flagged": sum(1 for p in done if p["flag_k"] is not None),
                    "flagged_before_final": sum(1 for p in done if p["flag_k"] is not None and p["flag_k"] < p["total"]),
                },
                "accounts": rep.engine.state.n_accounts,
                "load_seconds": rep.load_seconds,
            }
        return jsonify(payload)

    @app.get("/api/jumps")
    def jumps():
        rows = []
        for pid, idx in sorted(rep.pattern_first_idx.items(), key=lambda kv: kv[1]):
            if rep.pattern_start.get(pid, -1) >= rep.stream_start and rep.pattern_total[pid] >= 4:
                rows.append({"id": pid, "type": rep.pattern_type.get(pid, "?"), "index": idx,
                             "total": rep.pattern_total[pid], "ts": fmt_ts(rep.s["ts"][idx]),
                             "spotlight": pid == rep.spotlight})
        return jsonify(rows)

    @app.post("/api/control")
    def control():
        body = request.get_json(force=True, silent=True) or {}
        action = body.get("action")
        if action == "start":
            rep.running = True
        elif action == "pause":
            rep.running = False
        elif action == "rate":
            rep.rate = RATES.get(str(body.get("value")), 100)
        elif action == "reset":
            with rep.lock:
                rep.running = False
                rep.ff_target = None
                rep.reset_state()
        elif action == "jump":
            pid = int(body.get("pattern", rep.spotlight))
            idx = rep.pattern_first_idx.get(pid)
            if idx is None:
                return jsonify({"error": "unknown pattern"}), 400
            rep.jump(idx)
        else:
            return jsonify({"error": "unknown action"}), 400
        return jsonify({"ok": True})

    @app.post("/api/score")
    def score():
        """Score one payment. Body: src, dst, amount, currency, format, optional ts (ISO),
        cross_currency, commit (default false: the account profiles are left unchanged)."""
        b = request.get_json(force=True, silent=True) or {}
        t0 = time.perf_counter()
        with rep.lock:
            eng = rep.engine
            if "ts" in b:
                ts = int(datetime.fromisoformat(str(b["ts"])).replace(tzinfo=timezone.utc).timestamp() // 60)
            else:
                ts = rep.s["ts"][max(rep.pos - 1, 0)]
            currency = b.get("currency", "US Dollar")
            fmt = b.get("format", "ACH")
            txn = {"ts": ts, "src": int(b["src"]), "dst": int(b["dst"]),
                   "amt_usd": eng.to_usd(float(b["amount"]), currency), "fmt": fmt, "currency": currency,
                   "cross_currency": int(b.get("cross_currency", 0))}
            d = eng.decide(txn, commit=bool(b.get("commit", False)))
        return jsonify({"decision": d.decision, "score": round(d.score, 6), "reasons": d.reasons,
                        "drivers": {g: round(v, 4) for g, v in sorted(d.drivers.items(), key=lambda kv: -kv[1])},
                        "engine_ms": round(d.latency_ms, 3),
                        "request_ms": round((time.perf_counter() - t0) * 1000, 3),
                        "machine": info["machine"]})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts" / "engine"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    t0 = time.time()
    rep = Replayer(Path(args.model_dir), Path(args.artifacts))
    print(f"TrustPulse engine ready in {time.time() - t0:.1f}s on {platform.machine()}: "
          f"{rep.n:,} payments to replay, {rep.engine.state.n_accounts:,} account profiles loaded")
    print(f"Open http://localhost:{args.port}  (through: ssh -L {args.port}:localhost:{args.port} ...)")
    create_app(rep).run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
