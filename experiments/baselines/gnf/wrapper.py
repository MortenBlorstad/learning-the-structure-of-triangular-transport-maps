"""GNFBaseline — Graphical Normalizing Flow (Wehenkel & Louppe, AISTATS 2021) wrapped to
the repo's fit/log_pdf interface (see FlowBase in experiments/baselines/flows.py).

Faithful to the authors' UCIExperiments.py training procedure (vendored code under this
package; provenance in ATTRIBUTION.md):

- monotonic (UMNN) normalizer, single flow step, DAG conditioner with hot encoding and
  Gumbel temperature 0.5 (their train() settings);
- per epoch: constrainA(0) on every conditioner, then shuffled minibatches minimizing
  model.loss = augmented-Lagrangian constraint + l1 - mean log-likelihood, then
  model.step(epoch, avg_loss) which runs their dual updates on their schedule;
- integration steps randomized per batch during training (nb_steps + U{0..9}); evaluation
  always at nb_steps + 20 (their validation/test setting);
- model selection: best validation NLL SUBJECT TO DAGness < 1e-20 (their rule); the best
  state is restored after training when one exists, as with the other flow baselines.

Structure read-out: get_adjacency_matrix() applies the authors' post_process() (threshold
raised until acyclic) to a COPY of the current weights, so the live density model is never
mutated and eval_both's best/last state swapping yields state-faithful structure metrics.

The train-loss NaN guard breaks the epoch loop instead of exiting the process (their
script calls exit()); the row's NLL then surfaces the failure.
"""
from __future__ import annotations

import contextlib
import copy
import io
import math

import numpy as np
import torch

from experiments.baselines.gnf import buildFCNormalizingFlow, DAGConditioner, MonotonicNormalizer


class GNFBaseline:
    def __init__(self, emb_net=(60, 60, 60, 30), int_net=(100, 100, 100), nb_steps=20,
                 solver="CC", l1=0.0, nb_steps_dual=30, gumble_T=0.5, nb_flow=1, device=None,
                 gamma=None):
        self.emb_net = list(emb_net)
        self.int_net = list(int_net)
        self.nb_steps = int(nb_steps)
        self.solver = solver
        self.l1 = float(l1)
        self.nb_steps_dual = int(nb_steps_dual)
        self.gumble_T = float(gumble_T)
        self.nb_flow = int(nb_flow)
        # progress threshold of their ratchet (c *= eta unless the constraint fell below
        # gamma x its previous value). None = their released 0.9; 0.25 is the value the
        # NOTEARS/DAG-GNN line uses and Charpentier et al. ran for DAG-NF.
        self.gamma = None if gamma is None else float(gamma)
        self.flow = None
        self.d_ = None
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def _build(self, d: int):
        cond_args = {"in_size": d, "hidden": self.emb_net[:-1], "out_size": self.emb_net[-1],
                     "l1": self.l1, "gumble_T": self.gumble_T,
                     "nb_epoch_update": self.nb_steps_dual, "hot_encoding": True}
        norm_args = {"integrand_net": self.int_net, "cond_size": self.emb_net[-1],
                     "nb_steps": self.nb_steps, "solver": self.solver}
        return buildFCNormalizingFlow(self.nb_flow, DAGConditioner, cond_args,
                                      MonotonicNormalizer, norm_args)

    def _cpu_state(self):
        return {k: v.detach().to("cpu", copy=True) for k, v in self.flow.state_dict().items()}

    @torch.no_grad()
    def _eval_nll(self, X_t: torch.Tensor, batch_size: int) -> np.ndarray:
        """Per-point NLL at the authors' evaluation setting (nb_steps + 20)."""
        for nrm in self.flow.getNormalizers():
            nrm.nb_steps = self.nb_steps + 20
        out = []
        for i in range(0, len(X_t), batch_size):
            z, jac = self.flow(X_t[i:i + batch_size])
            out.append(-(self.flow.z_log_density(z) + jac).cpu().numpy())
        return np.concatenate(out).ravel()

    def fit(self, X: np.ndarray, epochs: int = 200, batch_size: int = 100,
            lr: float = 1e-3, weight_decay: float = 1e-5, verbose: bool = False,
            X_val: np.ndarray = None, eval_every: int = 0):
        X = np.asarray(X, dtype=np.float32)
        self.d_ = X.shape[1]
        self.flow = self._build(self.d_).to(self.device)
        if self.gamma is not None:                  # their gamma is a buffer, not a ctor arg
            for cond in self.flow.getConditioners():
                cond.gamma.fill_(self.gamma)
        X_t = torch.tensor(X, device=self.device)
        val_t = None if X_val is None else torch.tensor(np.asarray(X_val, np.float32),
                                                        device=self.device)

        opt = torch.optim.Adam(self.flow.parameters(), lr=lr, weight_decay=weight_decay)
        self.history_ = {"epoch": [], "nll_train": [], "nll_val": [], "nll_val_median": [],
                         "nll_val_max": [], "dagness": [], "lambd": [], "c": [],
                         "no_update": [], "cond_loss": []}   # dual-update diagnostics
        self.is_fitted_ = True
        self.best_val_nll_ = float("inf")
        self.best_epoch_ = None
        self.best_state_ = None
        self.dagness_final_ = float("nan")
        self.nan_epoch_ = None      # epoch where a non-finite loss aborted training (None = ran
        self.step_log_ = []         # to the budget); the row's NLL is then untrained, not a result

        n = len(X_t)
        for epoch in range(epochs):
            with torch.no_grad():                       # their per-epoch A clean-up
                for cond in self.flow.getConditioners():
                    cond.constrainA(zero_threshold=0.)

            self.flow.train()
            perm = torch.randperm(n, device=self.device)
            loss_tot, nan_hit = 0.0, False
            n_batches = max(1, n // batch_size)
            for i in range(n_batches):
                idx = perm[i * batch_size:(i + 1) * batch_size]
                for nrm in self.flow.getNormalizers():  # their per-batch step randomization
                    nrm.nb_steps = self.nb_steps + int(torch.randint(0, 10, (1,)))
                z, jac = self.flow(X_t[idx])
                loss = self.flow.loss(z, jac)
                if math.isnan(loss.item()) or math.isinf(abs(loss.item())):
                    nan_hit = True                      # their script exits; we stop and surface
                    break
                loss_tot += loss.detach()
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
            if nan_hit:
                self.nan_epoch_ = epoch
                break
            loss_avg = loss_tot / n_batches
            buf = io.StringIO()                 # capture their step() prints (post-processing
            with contextlib.redirect_stdout(buf):   # narrative) instead of discarding them
                self.flow.step(epoch, loss_avg)
            msg = buf.getvalue()
            for line in msg.splitlines():           # keep only the event lines, tagged by epoch
                if "ost processing" in line or "Error" in line or "edges" in line or "cycle" in line:
                    self.step_log_.append(f"epoch {epoch}: {line.strip()}")

            if verbose and epoch % 100 == 0:
                c0 = self.flow.getConditioners()[0]
                print(f"epoch {epoch}  loss {float(loss_avg):.3f}  dagness {float(max(self.flow.DAGness())):.3g}"
                      f"  c {float(c0.c):.3g}  lambda {float(c0.lambd):.3g}", flush=True)
            self.flow.eval()
            with torch.no_grad():
                dagness = float(max(self.flow.DAGness()))
                self.dagness_final_ = dagness
                if val_t is not None:
                    # Best-val only among COMMITTED states (in-loop post_process succeeded ->
                    # dag_const == 0, binary graph, deterministic phase). Gating on raw
                    # DAGness < 1e-20 admits the pre-commit soft-adjacency epoch, whose
                    # thresholded re-eval explodes (+100s of nats on Sachs pooled).
                    committed = all(float(c.dag_const) == 0.0
                                    for c in self.flow.getConditioners())
                    log_due = bool(eval_every) and (epoch % eval_every == 0 or epoch == epochs - 1)
                    if not (committed or log_due):
                        continue    # pre-certification, no log due: the gate would discard this
                                    # eval anyway -- skipping it is the dominant epoch cost saved
                                    # (5k val points x 40 UMNN integration steps vs ~10 updates)
                    val_nll = self._eval_nll(val_t, max(batch_size, 4096))
                    v = float(val_nll.mean())
                    if committed and v < self.best_val_nll_:
                        self.best_val_nll_ = v
                        self.best_epoch_ = epoch
                        self.best_state_ = self._cpu_state()
                    if log_due:
                        self.history_["epoch"].append(epoch)
                        self.history_["nll_train"].append(float(loss_avg))
                        self.history_["nll_val"].append(v)
                        self.history_["nll_val_median"].append(float(np.median(val_nll)))
                        self.history_["nll_val_max"].append(float(val_nll.max()))
                        self.history_["dagness"].append(dagness)
                        cond = self.flow.getConditioners()[0]
                        self.history_["lambd"].append(float(cond.lambd))
                        self.history_["c"].append(float(cond.c))
                        self.history_["no_update"].append(int(cond.no_update))
                        self.history_["cond_loss"].append(float(cond.loss()))

        self.last_state_ = self._cpu_state()
        if self.best_state_ is not None:                # restore best, the flow baselines' convention
            self.flow.load_state_dict(self.best_state_)
        return self

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(np.asarray(X, np.float32), device=self.device)
        self.flow.eval()
        return -self._eval_nll(X_t, 8192)

    @torch.no_grad()
    def pushforward(self, X: np.ndarray) -> np.ndarray:
        """z = T(x) at the evaluation setting (nb_steps + 20); presence of this method makes
        eval_metrics emit the val_z_absmax / val_n_blowup20 tail diagnostics for GNF rows."""
        X_t = torch.tensor(np.asarray(X, np.float32), device=self.device)
        self.flow.eval()
        for nrm in self.flow.getNormalizers():
            nrm.nb_steps = self.nb_steps + 20
        out = []
        for i in range(0, len(X_t), 8192):
            z, _ = self.flow(X_t[i:i + 8192])
            out.append(z.cpu().numpy())
        return np.concatenate(out, axis=0)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Binary DAG via the authors' post_process(), applied to a copy of the current
        weights (state-faithful under eval_both's best/last swapping; model unmutated)."""
        adj = np.zeros((self.d_, self.d_), dtype=int)
        for cond in self.flow.getConditioners():
            # deepcopy refuses non-leaf tensors: their code stores graph-attached tensors in
            # A.grad (back-off branch) and in registered buffers (prev_trace at init). Detach
            # both on the source; harmless post-training.
            for prm in cond.parameters():
                prm.grad = None
            for name, buf in list(cond._buffers.items()):
                if buf is not None:
                    cond._buffers[name] = buf.detach().clone()
            c = copy.deepcopy(cond)
            c.A.requires_grad_(False)   # their in-place post_process trips torch>=2 on a grad leaf
            c.post_process()
            adj |= (c.A.detach().cpu().numpy() > 0).astype(int)
        # their A[i, j] masks input j into component i => edge j -> i; our B[i, j] = i -> j
        return adj.T
