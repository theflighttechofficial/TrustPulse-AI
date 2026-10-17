# Resources

Everything TrustPulse AI uses: open-source libraries, datasets, platform and tools. All code in this repository was written during the IBM Z Datathon 2026 event window. Each entry links to its source.

## Platform

| Resource | Use | License / terms |
|---|---|---|
| [IBM LinuxONE Community Cloud](https://community.ibm.com/zsystems/l1cc/) | Development and execution VM (Ubuntu 24.04, s390x, 3 vCPU, 8 GB RAM) | IBM LinuxONE Community Cloud terms and conditions |
| [Ubuntu 24.04 LTS (s390x)](https://ubuntu.com/download/server/s390x) | Operating system and native Python packages | Mostly GPL and other open-source licenses (Canonical) |

## Open-source libraries

| Library | Version | Use | License |
|---|---|---|---|
| [Python](https://www.python.org) | 3.12 | Language runtime | PSF License |
| [NumPy](https://numpy.org) | Ubuntu package | Arrays, numerics | BSD-3-Clause |
| [pandas](https://pandas.pydata.org) | Ubuntu package | Data loading, streaming feature build | BSD-3-Clause |
| [SciPy](https://scipy.org) | Ubuntu package | scikit-learn dependency | BSD-3-Clause |
| [scikit-learn](https://scikit-learn.org) | Ubuntu package | HistGradientBoostingClassifier, metrics | BSD-3-Clause |
| [matplotlib](https://matplotlib.org) | 3.6.3 (Ubuntu) | Evaluation figures | Matplotlib License (PSF-based) |
| [joblib](https://joblib.readthedocs.io) | Ubuntu package | Model persistence | BSD-3-Clause |
| [JupyterLab](https://jupyter.org) | 4.x | Notebook environment | BSD-3-Clause |
| [Flask](https://flask.palletsprojects.com) | 3.x | Live console and REST scoring endpoint | BSD-3-Clause |
| [IBM Snap ML (`snapml`)](https://pypi.org/project/snapml/) | 1.17.2 | Telum II accelerator benchmark (tree inference via zDNN) | IBM proprietary licence, free on PyPI |
| [IBM zDNN (`libzdnn0`)](https://github.com/IBM/zDNN) | 1.0.1 (Ubuntu) | Access to the NNPA on-chip AI accelerator | Apache-2.0 |
| [cryptography](https://cryptography.io) | Ubuntu package | AES-256-GCM encryption of engine state (OpenSSL, CPACF) | Apache-2.0 / BSD-3-Clause |
| [OpenSSL](https://www.openssl.org) | 3.0.13 (Ubuntu) | AES-GCM on CPACF | Apache-2.0 |
| [pytest](https://pytest.org) | 8.x | Streaming-vs-batch parity tests | MIT |
| [Podman](https://podman.io) | Ubuntu package | Runs the engine and console as an s390x container (`Containerfile`) | Apache-2.0 |
| [IPython / ipykernel](https://ipython.org) | pip | Notebook kernel | BSD-3-Clause |
| [Kaggle API client](https://github.com/Kaggle/kaggle-api) | 2.2.4 | Dataset download | Apache-2.0 |

Print the exact versions on the VM with (also part of `scripts/platform_evidence.sh`):

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, joblib; print(numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, matplotlib.__version__, joblib.__version__)"
jupyter lab --version
```

XGBoost was considered but not used: no s390x wheel was available on PyPI for this VM.

## Datasets

| Dataset | Source | Use | License |
|---|---|---|---|
| IBM Transactions for Anti-Money Laundering (HI-Small) | [Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | Main evaluation: network features, lead time, decision layer, live replay | CDLA-Sharing-1.0 |
| PaySim synthetic mobile-money transactions | [Kaggle: `ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1) | Per-transaction baseline | CC BY-SA 4.0 (as listed on Kaggle) |

Raw data is not redistributed in this repository; the README shows how to download it.

Citations:

- E. Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets and Benchmarks.
- E. A. Lopez-Rojas, A. Elmir, S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection", EMSS 2016.

## Tools

| Tool | Use |
|---|---|
| IBM Plex Sans / Mono (Google Fonts) | Console typography (SIL Open Font License 1.1) |
| Git and GitHub | Version control; commit history shows work done during the event |
| tmux | Keeps the Jupyter server running across SSH disconnects |
| OpenSSH | VM access and port tunnel to JupyterLab |
| Claude (Anthropic) | AI assistant used for guidance and code suggestions during the build |
