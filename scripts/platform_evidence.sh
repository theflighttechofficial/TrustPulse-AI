#!/usr/bin/env bash
# Print the LinuxONE platform facts that TrustPulse relies on, and save them to
# results/platform_evidence.txt. Run on the VM:  bash scripts/platform_evidence.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p results
{
  echo "== TrustPulse AI platform evidence, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "== Machine"
  uname -m
  grep -m1 -E '^Type|Type:' /proc/sysinfo 2>/dev/null || true
  grep -m1 -E 'Model:' /proc/sysinfo 2>/dev/null || true
  grep -E '^Control Program' /proc/sysinfo 2>/dev/null || true
  grep -E '^PRETTY_NAME' /etc/os-release
  echo "CPUs: $(nproc)"
  free -h | sed -n 1,3p
  echo
  echo "== CPU facilities"
  feats=$(grep -m1 '^features' /proc/cpuinfo)
  for f in nnpa msa vx vxe vxe2; do
    if grep -qw "$f" <<<"$feats"; then echo "$f: present"; else echo "$f: absent"; fi
  done
  echo
  echo "== Telum II accelerator software"
  dpkg-query -W -f='${Package} ${Version}\n' libzdnn0 2>/dev/null || echo "libzdnn0: not installed"
  python3 -c "import snapml; print('snapml', snapml.__version__)" 2>/dev/null || echo "snapml: not importable here"
  echo
  echo "== Python stack"
  python3 -c "import sys, numpy, pandas, sklearn; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scikit-learn', sklearn.__version__)" 2>/dev/null
  echo
  echo "== OpenSSL AES-256-GCM (CPACF), 3 s"
  openssl version
  openssl speed -elapsed -seconds 3 -evp aes-256-gcm 2>/dev/null | tail -2
  echo
  echo "== Engine tests"
  python3 -m pytest -q tests 2>&1 | tail -1
} | tee results/platform_evidence.txt
