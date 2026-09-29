# Sachs 2005 data — provenance

The data files are NOT committed (kept out of git history); only this doc is tracked.
After cloning, fetch them by running from this directory:

```bash
curl -sL -O "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.continuous.txt"
curl -sL -o sachs.2005.ground.truth.graph.txt "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/ground.truth/sachs.2005.ground.truth.graph.txt"
```

Downloaded 2026-09-09 from cmu-phil/example-causal-datasets (main branch):

- `sachs.2005.continuous.txt` —
  https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.continuous.txt
  Tab-separated, header row + 7,466 rows x 11 columns (raf mek plc pip2 pip3 erk akt pka pkc p38 jnk).
  The pooled continuous measurements of Sachs et al. (2005), all 14 experimental conditions —
  the same matrix NOTEARS (1803.01422) and NOTEARS-MLP (1909.13189) evaluate on.
- `sachs.2005.ground.truth.graph.txt` —
  https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/ground.truth/sachs.2005.ground.truth.graph.txt
  Tetrad format, exactly 20 directed edges (the consensus network NOTEARS reports against).
  Known variants elsewhere: 17-edge textbook consensus (GraN-DAG), 18-edge (cdt). SHD values
  are comparable only within one variant; ours use this 20-edge version (disclosed in the paper).

`experiments/sachs/experiment.py::load_sachs` asserts shape, column names, and edge count
on every load, and points here when the files are missing.

## Observational subset (obs853)

The pooled file is the 14 experimental conditions concatenated in order, with the
observational condition (cd3/cd28 stimulation, 853 rows) first. Verified 2026-09-09: the
first 853 rows are value-identical (max diff 3e-6, decimal formatting) to bnlearn's
independent observational file (https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz,
853 x 11). The loader's `subset="obs853"` slices `X[:853]` and asserts a sha256 checksum
of the block, so any upstream reordering fails loudly. No separate download needed.
