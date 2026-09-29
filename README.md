# Learning the Structure of Triangular Transport Maps

Code for the paper. **SSTM** (Self-Structuring Transport Maps) is a monotone triangular
transport map that learns the variable ordering and the sparsity pattern jointly with the
map, in a single optimization. The model is `triangularmap/sstm.py`.

## Install

Python 3.13, managed with [uv](https://docs.astral.sh/uv/). `uv.lock` pins every dependency.

```bash
uv sync
```

## Data

The synthetic experiments generate their own data. The real-data experiments need one
download. UCI (POWER, GAS, HEPMASS, MINIBOONE) is described in
`experiments/uci/README.md`, Sachs in `experiments/sachs/README.md`.

## Run an experiment

All four experiments go through one launcher.

```bash
uv run python -m experiments.runner.launch --experiment synthetic --jobs 8
uv run python -m experiments.runner.launch --experiment method_ablation --jobs 8
uv run python -m experiments.runner.launch --experiment sachs --jobs 8
uv run python -m experiments.runner.launch --experiment uci --jobs 8
```

`--jobs N` runs N cells at a time on CPU or MPS. `--gpus 0,1 --per-gpu 4` pins cells to
GPUs. `--dry-run` prints the schedule without running anything. Each cell writes one json
to `results/<experiment>/`.

## Figures and tables
After running the experiments, you can plot the results and make tables:

```bash
uv run python -m experiments.synthetic.plot_results
uv run python -m experiments.synthetic.plot_dag            
uv run python -m experiments.method_ablation.plot_results  
uv run python -m experiments.sachs.plot_results            
uv run python -m experiments.uci.make_tables              
```


## Use SSTM in your own projects

SSTM installs on its own, without the experiments. It needs only numpy, scipy and torch.

```bash
uv add "sstm @ git+https://github.com/MortenBlorstad/learning-the-structure-of-triangular-transport-maps"
```

or 

```bash
pip install git+https://github.com/MortenBlorstad/learning-the-structure-of-triangular-transport-maps
```

Example usage:
```python
import numpy as np
from triangularmap import SSTM

X = ...                                        # [n, K] float32, standardized
model = SSTM(input_size=X.shape[1], hidden=[64, 64], output_dim=1, random_state=0)
model.fit(X, epochs=500, batch_size=64, l1=1e-3, sparsity_strength=1e-3)

model.log_pdf(X)                  # log density, [n]
Z = model.pushforward(X)          # X -> Z, [n, K]
model.pullback(Z)                 # Z -> X, [n, K]
model.sample(1000)                # draw from the fitted map
model.condition(1000, 0.5)        # fix the first variable in the learned order, sample the rest
model.get_adjacency_matrix()      # the sparsity pattern it learned, [K, K]
model.hard_order()                # the variable ordering it learned, rank -> feature
```



    
## Licence

MIT, with three exceptions. `experiments/baselines/gnf/` vendors code from
[Graphical Normalizing Flows](https://github.com/AWehenkel/Graphical-Normalizing-Flows) under
CC-BY-4.0; see `experiments/baselines/gnf/ATTRIBUTION.md`. `experiments/uci/data.py` adapts
the UCI loaders of Papamakarios et al. under BSD-2-Clause. `distributions/spiral.py` adapts a
sampler from the
[Triangular Transport Tutorial](https://github.com/MaxRamgraber/Triangular-Transport-Tutorial)
under Apache-2.0. Each notice is in the file it applies to.
