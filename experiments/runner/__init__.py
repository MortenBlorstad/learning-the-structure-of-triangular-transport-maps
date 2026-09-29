"""Generic parallel experiment runner.

A registry of experiments (name -> Experiment) plus an experiment-agnostic
launcher and single-cell runner. Each experiment provides three small functions
(cells, cell_id, run_one); the runner handles grid -> parallel dispatch, per-cell
result jsons, logging, and checkpoints — identically for every experiment.

    python -m experiments.runner.launch --experiment synthetic --jobs 8
"""
