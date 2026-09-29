# Sachs 2005 data — provenance

The data files are not committed. Fetch them by running from the repo root:

```bash
mkdir -p experiments/sachs/data
curl -sL -o experiments/sachs/data/sachs.2005.continuous.txt \
  "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.continuous.txt"
curl -sL -o experiments/sachs/data/sachs.2005.ground.truth.graph.txt \
  "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/ground.truth/sachs.2005.ground.truth.graph.txt"
```

Downloaded 2026-09-09 from cmu-phil/example-causal-datasets (main branch). `load_sachs`
asserts the shape, column names and edge count on every load, and points here when the
files are missing.

The reference graph has exactly 20 directed edges, the consensus network NOTEARS reports
against. Variants circulate elsewhere: 17-edge textbook consensus (GraN-DAG), 18-edge
(cdt). SHD is comparable only within one variant; ours use the 20-edge version (disclosed
in the paper).

`subset="obs853"` needs no separate download. It slices the first 853 rows of the pooled
file, which are the observational cd3/cd28 condition, and asserts a sha256 of the block.
