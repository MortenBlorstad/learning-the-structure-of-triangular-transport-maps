"""UCI density-estimation dataset loaders (POWER, GAS, HEPMASS, MINIBOONE).

Adapted from https://github.com/francois-rozet/uci-datasets (a fork of the MAF
paper's `datasets/`, Papamakarios et al. 2017). Cleaning + splitting are
reproduced verbatim (same fixed RNG, shuffle, noise draws, column filters, split
rule), but the loaders' *internal z-score normalization is omitted* — a deliberate
deviation: the density experiment applies a single `StandardScaler` fit on TRAIN
only (the papers use train+val for POWER/MINIBOONE, all rows for GAS, train for
HEPMASS), shared across models, so standardization lives in one place. One
consequence: the HEPMASS discrete-column filter therefore runs on raw values
instead of z-scored ones; the retained set is unchanged, since distinct-value
counts are invariant under a per-column affine map. Arrays are returned in
float64; the caller casts after standardizing.

Raw data (download once into experiments/uci/data/, not committed — see
.gitignore). The archive has a top-level `data/`, so extract it into this
package directory; only the four UCI sets are needed:

    curl -L -o data.tar.gz https://zenodo.org/record/1161203/files/data.tar.gz
    tar -xzf data.tar.gz -C experiments/uci/ data/power data/gas data/hepmass data/miniboone
    rm data.tar.gz

Resulting layout under `experiments/uci/data/`:
    power/data.npy
    gas/ethylene_CO.pickle
    hepmass/{1000_train.csv,1000_test.csv}
    miniboone/data.npy               (already outlier/discrete-filtered upstream)

Source datasets:
    POWER     http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
    GAS       http://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures
    HEPMASS   http://archive.ics.uci.edu/ml/datasets/HEPMASS
    MINIBOONE http://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
"""

# The cleaning and splitting below are adapted from the UCI loaders of the MAF paper
# (https://github.com/gpapamak/maf, via https://github.com/francois-rozet/uci-datasets).
#
# Copyright (c) 2017, George Papamakarios
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent / "data"  # experiments/uci/data/


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the UCI archive first (see "
            f"experiments/uci/README.md):\n"
            f"  curl -L -o data.tar.gz https://zenodo.org/record/1161203/files/data.tar.gz\n"
            f"  tar -xzf data.tar.gz -C experiments/uci/ data/power data/gas data/hepmass data/miniboone\n"
            f"  rm data.tar.gz"
        )
    return path


# --------------------------------------------------------------------------- #
# POWER  (n_dims = 6)
# --------------------------------------------------------------------------- #
def _load_power():
    data = np.load(_require(DATA_ROOT / "power" / "data.npy"))
    rng = np.random.RandomState(42)  # fixed split/noise across experiment seeds
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    
    voltage_noise = 0.01 * rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    data = data + np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))

    n_test = int(0.1 * data.shape[0])
    test, rest = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * rest.shape[0])
    val, train = rest[-n_val:], rest[:-n_val]
    return train, val, test


# --------------------------------------------------------------------------- #
# GAS  (n_dims = 8)
# --------------------------------------------------------------------------- #
def _legacy_pandas_pickle(path):
    """Read a pandas pickle written by pre-1.0 pandas. The GAS pickle references
    the old `pandas.core.index` module and the removed `Int64Index`; install a
    minimal compat shim (idempotent) before reading. Purely enables loading on
    pandas 2.x — no data is altered. (francois-rozet's own pipeline requires
    pandas<2.0 to avoid exactly this.)"""
    import sys
    import types

    import pandas as pd

    if "pandas.core.index" not in sys.modules:
        from pandas.core.indexes.base import Index, _new_Index
        from pandas.core.indexes.multi import MultiIndex

        shim = types.ModuleType("pandas.core.index")
        shim.Index = Index
        shim.Int64Index = Index      # removed in pandas 2.0 -> map to Index
        shim.Float64Index = Index
        shim.MultiIndex = MultiIndex
        shim._new_Index = _new_Index
        sys.modules["pandas.core.index"] = shim
    return pd.read_pickle(path)


def _load_gas():
    df = _legacy_pandas_pickle(_require(DATA_ROOT / "gas" / "ethylene_CO.pickle"))
    for col in ("Meth", "Eth", "Time"):
        df.drop(col, axis=1, inplace=True)

    # iteratively drop the first column still correlated (>0.98) with another
    def over_correlated(d):
        return (d.corr() > 0.98).values.sum(axis=1)  

    B = over_correlated(df)
    while np.any(B > 1):
        col = df.columns[np.where(B > 1)[0][0]]
        df.drop(col, axis=1, inplace=True)
        B = over_correlated(df)

    data = df.values
    n_test = int(0.1 * data.shape[0])
    test, rest = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * rest.shape[0])
    val, train = rest[-n_val:], rest[:-n_val]
    return train, val, test


# --------------------------------------------------------------------------- #
# HEPMASS  (n_dims = 21)
# --------------------------------------------------------------------------- #
def _load_hepmass():
    import pandas as pd

    root = DATA_ROOT / "hepmass"
    train_df = pd.read_csv(_require(root / "1000_train.csv"), index_col=False)
    test_df = pd.read_csv(_require(root / "1000_test.csv"), index_col=False)

    # keep signal class only (label in first column == 1), then drop the label
    train_df = train_df[train_df[train_df.columns[0]] == 1]
    train_df = train_df.drop(train_df.columns[0], axis=1)
    test_df = test_df[test_df[test_df.columns[0]] == 1]
    test_df = test_df.drop(test_df.columns[0], axis=1)
    test_df = test_df.drop(test_df.columns[-1], axis=1)  # upstream: test has a trailing junk column

    train, test = train_df.values, test_df.values

    # remove features with too few distinct values
    remove = []
    for i, feature in enumerate(train.T):
        counts = [v for _, v in sorted(Counter(feature).items())]
        if counts[0] > 5:
            remove.append(i)
    keep = [i for i in range(train.shape[1]) if i not in remove]
    train, test = train[:, keep], test[:, keep]

    n_val = int(0.1 * train.shape[0])
    val, train = train[-n_val:], train[:-n_val]
    return train, val, test


# --------------------------------------------------------------------------- #
# MINIBOONE  (n_dims = 43)  — data.npy is already outlier/discrete-filtered upstream
# --------------------------------------------------------------------------- #
def _load_miniboone():
    data = np.load(_require(DATA_ROOT / "miniboone" / "data.npy"))
    n_test = int(0.1 * data.shape[0])
    test, rest = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * rest.shape[0])
    val, train = rest[-n_val:], rest[:-n_val]
    return train, val, test


_LOADERS = {
    "POWER": _load_power,
    "GAS": _load_gas,
    "HEPMASS": _load_hepmass,
    "MINIBOONE": _load_miniboone,
}


def load_uci(name: str):
    """Return (X_train, X_val, X_test) as un-normalized float64 arrays.

    Cleaning/splitting follows the MAF-paper (francois-rozet) convention; the
    caller standardizes (StandardScaler fit on TRAIN, statistics in float64) and
    casts to float32 afterwards.
    """
    if name not in _LOADERS:
        raise KeyError(f"unknown UCI dataset '{name}'. available: {sorted(_LOADERS)}")
    return _LOADERS[name]()
