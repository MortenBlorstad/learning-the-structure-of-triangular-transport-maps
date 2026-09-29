# Vendored code: Graphical Normalizing Flows

- Source: https://github.com/AWehenkel/Graphical-Normalizing-Flows
- Commit: 2cc6fba392897ec1884b4f01a695b83d3c04883a (master), fetched 2026-09-11
- Authors: Antoine Wehenkel, Gilles Louppe
- Paper: "Graphical Normalizing Flows", AISTATS 2021, arXiv:2006.02548
- License: Creative Commons Attribution 4.0 International (CC-BY-4.0), per the
  repository's `license` file

Vendored subset (from `models/`): `NormalizingFlow.py`, `NormalizingFlowFactories.py`,
`MLP.py`, `Conditionners/{Conditioner,DAGConditioner}.py`,
`Normalizers/{Normalizer,AffineNormalizer,MonotonicNormalizer}.py`.

Local modifications (kept minimal and auditable):
- `__init__.py` files trimmed to the vendored subset (the autoregressive/coupling
  conditioners and image models are not shipped).
- Any Python/torch/networkx version-compatibility fix is marked with a
  `# compat fix:` comment at the changed line.

`wrapper.py` (ours, not vendored) adapts the authors' UCIExperiments.py training
procedure to this repo's fit/log_pdf interface. The UMNN dependency is installed from
PyPI, not vendored.
