"""Encrypt the engine's model and account-state files at rest with AES-256-GCM.

On IBM Z, OpenSSL runs AES-GCM on CPACF, the crypto unit in every core, and the
Python `cryptography` package calls OpenSSL. This script encrypts the files the
live engine loads, checks they decrypt to identical bytes, and records the
throughput.

    python scripts/cpacf_state.py             # encrypt + verify + benchmark
Writes artifacts/engine/*.enc and results/cpacf_state.json. The key stays in
artifacts/state.key (not committed); a real deployment would hold it in a key store.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from cryptography.hazmat.backends.openssl import backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "artifacts" / "state.key"
FILES = [ROOT / "artifacts" / "engine" / "engine_state.npz", ROOT / "models" / "model_trees.json"]
OUT = ROOT / "results" / "cpacf_state.json"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    if not KEY.exists():
        KEY.write_bytes(AESGCM.generate_key(bit_length=256))
        os.chmod(KEY, 0o600)
    aes = AESGCM(KEY.read_bytes())
    rows = []
    for path in FILES:
        data = path.read_bytes()
        nonce = os.urandom(12)
        t0 = time.perf_counter(); ct = aes.encrypt(nonce, data, path.name.encode()); enc = time.perf_counter() - t0
        t0 = time.perf_counter(); pt = aes.decrypt(nonce, ct, path.name.encode()); dec = time.perf_counter() - t0
        assert pt == data, "round trip failed"
        enc_path = path.with_name(path.name + ".enc")
        enc_path.write_bytes(nonce + ct)
        mb = len(data) / 1e6
        rows.append({"file": str(path.relative_to(ROOT)), "mb": round(mb, 2),
                     "encrypt_ms": round(enc * 1e3, 2), "decrypt_ms": round(dec * 1e3, 2),
                     "encrypt_gb_per_s": round(mb / 1e3 / enc, 2), "decrypt_gb_per_s": round(mb / 1e3 / dec, 2),
                     "sha256_plain": hashlib.sha256(data).hexdigest()[:16], "round_trip_ok": True})
        print(f"{path.name:22} {mb:8.2f} MB  encrypt {enc*1e3:8.2f} ms ({mb/1e3/enc:6.2f} GB/s)  "
              f"decrypt {dec*1e3:8.2f} ms ({mb/1e3/dec:6.2f} GB/s)  round trip OK", flush=True)

    speed = sh("openssl speed -seconds 1 -evp aes-256-gcm 2>/dev/null | tail -2")
    cpu = sh("grep -m1 '^features' /proc/cpuinfo")
    res = {"machine": platform.machine(), "openssl": backend.openssl_version_text(),
           "cpu_features": cpu, "msa_in_cpu_flags": " msa" in cpu,
           "files": rows, "openssl_speed_aes_256_gcm": speed}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n{res['openssl']} | CPACF (msa) flag present: {res['msa_in_cpu_flags']}")
    print("openssl speed aes-256-gcm (bytes/s per block size):\n" + speed)


if __name__ == "__main__":
    main()
