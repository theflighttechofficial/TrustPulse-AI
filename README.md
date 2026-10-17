# TrustPulse AI

Real-time, network-aware risk scoring for instant payments, built and run on IBM LinuxONE (s390x).

IBM Z Datathon 2026, Theme 1: Real-Time AI for Critical Decisions. Solo entry by S. Varun Vaibhav.

## The problem

Instant-payment scams and money mules are hard to stop one transaction at a time. Each transfer on its own can look ordinary. The giveaway is how accounts behave together: money arrives in an account and leaves again within hours, one account collects from many new senders, a chain of first-time counterparties forms. By the time the last transfer lands, the money is gone.

TrustPulse scores every payment as it arrives, using what is already known about both accounts and their connections. Each payment gets one of three decisions, with a reason a bank analyst can read:

- **APPROVE**: let it through.
- **VERIFY**: add friction, such as confirmation of payee or step-up authentication.
- **REVIEW**: hold for an analyst.

## What runs on LinuxONE

Everything in this repository was developed and executed on an IBM LinuxONE Community Cloud VM (Ubuntu 24.04, s390x, 3 vCPU, 8 GB RAM). The Python stack uses Ubuntu's native s390x builds of NumPy, pandas, SciPy, scikit-learn and matplotlib. Measured on that VM:

| Step | Time on LinuxONE |
|---|---|
| Streaming feature build, 5.08M transactions | 13.5 s |
| Model training (sampled train set) | 29.8 s |
| Scoring, one transaction | 2.78 ms (sklearn per-call overhead) |
| Scoring, batched | 5.7 µs per transaction |
| Explaining 21,944 alerts | 2.3 s (about 105 µs per alert) |

No claims are made about the on-chip AI accelerator; it was not verified or used on this VM.

## Approach

1. **Leakage-safe streaming features.** Every transaction becomes an outgoing event for the sender and an incoming event for the receiver. Running totals exclude the current event, so each row only sees strictly earlier activity: counts, amounts, distinct counterparties, pass-through ratio (money out vs money in), account age, time since last activity, and whether the pair has transacted before.
2. **Gradient-boosted model** (scikit-learn `HistGradientBoostingClassifier`) trained on a time split, with the alert threshold fixed on validation and then frozen for test.
3. **Decision layer.** Validation-set quantiles set REVIEW (top 0.5%) and VERIFY (top 2%) thresholds.
4. **Faithful reasons.** For each alert, each feature group is reset to its typical value, and the drop in score is measured. The top three groups become the reason text, so the explanation comes from the model itself, not from separate rules.

## Data

- **IBM Transactions for Anti-Money Laundering, HI-Small** (main evaluation): 5,078,345 transactions, 515,088 accounts, with laundering-pattern labels (fan-in, fan-out, cycle, scatter-gather, and others). Rows after 2022-09-10 are dropped: they are almost entirely pattern tails (over 60% laundering), not real traffic.
- **PaySim** (per-transaction baseline): 6.36M mobile-money transactions. Balance columns are excluded because PaySim cancels fraudulent transactions, so balances leak the label. PaySim senders are 99.85% unique, so it carries no network signal.

Split for IBM AML: train 1–6 Sep, validation 7–8 Sep, test 9–10 Sep 2022. Laundering rate is about 0.1% in all three.

## Results (IBM AML HI-Small, test set)

**Network features are the main driver.** Same model, same alert budget (about 0.6% of transactions flagged):

| Features | PR-AUC | ROC-AUC | Recall | Precision |
|---|---|---|---|---|
| Transaction only (amount, channel, currency, hour) | 0.064 | 0.910 | 0.354 | 0.073 |
| Transaction + network | **0.525** | **0.958** | **0.719** | **0.129** |
| Network, without channel and currency | 0.469 | – | 0.714 | 0.056 |

The last row checks for a dataset artifact: about 87% of laundering in HI-Small uses ACH. With channel and currency removed, recall holds and PR-AUC stays near 0.47. The network signal is real; channel mainly adds precision.

**Decision bands:**

| Decision | Transactions | Laundering | Rate |
|---|---|---|---|
| APPROVE | 840,848 | 198 | 0.02% |
| VERIFY | 16,609 | 71 | 0.43% |
| REVIEW | 5,335 | 687 | 12.9% |

REVIEW plus VERIFY capture 79% of laundering transactions while touching 2.5% of traffic.

**Lead time.** Of 79 laundering patterns that start inside the test window, 96% raise at least one alert, and 58% are flagged before their final transfer (median: one transaction, 4.7 hours early). Patterns that continue past 10 Sep are cut off, so these lead times are a lower bound.

**Hard negatives.** 4,037 high-volume receiving accounts with no laundering involvement in training (merchant and payroll analogues). Their false-alert rate is 0.60%, against 0.54% for all legitimate traffic, so busy legitimate accounts are not over-flagged.

**PaySim baseline** (per-transaction only): test PR-AUC 0.559, ROC-AUC 0.966. PaySim's fraud rate jumps from 0.08% in training to 1.09% in the late test steps as legitimate volume collapses, so its precision figures are inflated and are not used for claims.

### What did not work

Risk propagation was tested: a flagged transaction leaves decaying risk on both accounts, and later payments touching them inherit it. Fourteen settings (attenuation 0.2–0.95, half-life 6–48 h, two seeding thresholds) were compared on validation. None beat the plain network model, so propagation is switched off. The structural network features already carry the linked-account signal, and propagation mostly added false alerts.

## Repository layout

```
notebooks/main_project.ipynb   end-to-end notebook (PaySim baseline, IBM AML engine, evaluation)
figures/                       ablation, lead time, decision bands, example pattern
results/metrics.json           all headline numbers from the final run
resources.md                   libraries, datasets and tools, with licenses
```

Raw data and model artifacts are not committed. They are downloaded and regenerated on the VM.

## How to run on LinuxONE

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential tmux \
    python3-numpy python3-pandas python3-scipy python3-sklearn python3-matplotlib
python3 -m venv --system-site-packages ~/tp
source ~/tp/bin/activate
pip install jupyterlab kaggle

git clone https://github.com/theflighttechofficial/TrustPulse-AI.git ~/trustpulse
mkdir -p ~/trustpulse/data && cd ~/trustpulse/data
kaggle datasets download -d ealaxi/paysim1 --unzip
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Patterns.txt
unzip -o '*.zip'

cd ~ && jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

From your own machine, open a tunnel and browse to `http://localhost:8888`:

```bash
ssh -i <key> -L 8888:localhost:8888 linux1@<vm-ip>
```

Open `notebooks/main_project.ipynb` and run all cells. On an 8 GB VM, add 4 GB of swap first; the feature build peaks at about 3 GB.

Pip has no s390x wheels for SciPy or scikit-learn, and building them from source needs a Fortran compiler and takes a long time. Ubuntu's packages avoid that.

## Limitations

- Both datasets are synthetic. Results show the method works on realistic simulated behaviour, not on a bank's live traffic.
- Lead time is measured on a two-day test window, so long patterns are truncated.
- The model leans on payment channel for precision; the robustness check above quantifies this.

## References

- E. Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets and Benchmarks.
- E. A. Lopez-Rojas, A. Elmir, S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection", EMSS 2016.
