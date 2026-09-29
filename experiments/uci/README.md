# Density-estimation experiment

Compares **SSTM** against the normalizing-flow baselines **MAF**, **NSF** (nflows,
the NSF authors' library) and **GNF** by test NLL on the UCI datasets **POWER, GAS,
HEPMASS, MINIBOONE** (MAF-paper splits). NLL is in nats; lower is better.

## Data (download once — not committed)

Run from the repo root. The archive has a top-level `data/`, so extract it into
this package directory; only the four UCI sets are needed (mnist/cifar10/bsds300
in the archive are skipped):

```bash
curl -L -o data.tar.gz https://zenodo.org/record/1161203/files/data.tar.gz
tar -xzf data.tar.gz -C experiments/uci/ data/power data/gas data/hepmass data/miniboone
rm data.tar.gz
```

Resulting layout under `experiments/uci/data/`:

```
experiments/uci/data/power/data.npy
experiments/uci/data/gas/ethylene_CO.pickle
experiments/uci/data/hepmass/{1000_train.csv,1000_test.csv}
experiments/uci/data/miniboone/data.npy
```

`data/` is git-ignored. The loaders in `data.py` reproduce the francois-rozet /
MAF cleaning + splitting (returning un-normalized arrays); the experiment then
fits a single `StandardScaler` on TRAIN, shared across all models.

## Run

```bash
uv run python -m experiments.runner.launch --experiment uci --jobs 4   # full sweep
uv run python -m experiments.uci.make_tables                          # the paper tables
```

Edit `experiments/config.yaml` (`uci` section) to change datasets, models, seeds or
hyperparameters. Each cell writes its own row to
`results/uci/<dataset>_<model>_seed<seed>.json`, and `make_tables` reads those into
`results/tables/tab_density_results.tex` and `tab_density_hparams.tex`.


