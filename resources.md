# Resources

Everything TrustPulse AI uses: open-source libraries, datasets, platform and tools. All code in this repository was written during the IBM Z Datathon 2026 event window.

## Platform

| Resource | Use | License / terms |
|---|---|---|
| IBM LinuxONE Community Cloud | Development and execution VM (Ubuntu 24.04, s390x, 3 vCPU, 8 GB RAM) | IBM LinuxONE Community Cloud terms and conditions |
| Ubuntu 24.04 LTS (s390x) | Operating system and native Python packages | Mostly GPL and other open-source licenses (Canonical) |

## Open-source libraries

| Library | Version | Use | License |
|---|---|---|---|
| Python | 3.12 | Language runtime | PSF License |
| NumPy | Ubuntu package | Arrays, numerics | BSD-3-Clause |
| pandas | Ubuntu package | Data loading, streaming feature build | BSD-3-Clause |
| SciPy | Ubuntu package | scikit-learn dependency | BSD-3-Clause |
| scikit-learn | Ubuntu package | HistGradientBoostingClassifier, metrics | BSD-3-Clause |
| matplotlib | 3.6.3 (Ubuntu) | Evaluation figures | Matplotlib License (PSF-based) |
| joblib | Ubuntu package | Model persistence | BSD-3-Clause |
| JupyterLab | 4.x | Notebook environment | BSD-3-Clause |
| IPython / ipykernel | pip | Notebook kernel | BSD-3-Clause |
| Kaggle API client | 2.2.4 | Dataset download | Apache-2.0 |

Fill in exact versions from the final run with:

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, joblib; print(numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, matplotlib.__version__, joblib.__version__)"
jupyter lab --version
```

XGBoost was considered but not used: no s390x wheel was available on PyPI for this VM.

## Datasets

| Dataset | Source | Use | License |
|---|---|---|---|
| IBM Transactions for Anti-Money Laundering (HI-Small) | Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | Main evaluation: network features, lead time, decision layer | TODO: copy the license shown on the Kaggle page |
| PaySim synthetic mobile-money transactions | Kaggle: `ealaxi/paysim1` | Per-transaction baseline | TODO: copy the license shown on the Kaggle page |

Raw data is not redistributed in this repository; the README shows how to download it.

Citations:

- E. Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets and Benchmarks.
- E. A. Lopez-Rojas, A. Elmir, S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection", EMSS 2016.

## Tools

| Tool | Use |
|---|---|
| Git and GitHub | Version control; commit history shows work done during the event |
| tmux | Keeps the Jupyter server running across SSH disconnects |
| OpenSSH | VM access and port tunnel to JupyterLab |
| Claude (Anthropic) | AI assistant used for guidance and code suggestions during the build |
