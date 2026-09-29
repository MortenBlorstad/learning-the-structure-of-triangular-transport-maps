"""Shared matplotlib style for the paper figures — one palette, one mode.

Single source of the visual conventions: the method/baseline identity used by every
results plot, and the font stack that makes a figure read as the paper's own text.

Three tiers so a results plot reads as "ours inside the envelope, hugging the good
edge": our method is the only saturated element, the oracle and random references are
neutral broken-linestyle bounds (a frame, not a contestant), and the baselines are
muted competitors.

``use_paper_style()`` is opt-in (it mutates global rcParams), so importing this module
never changes anything on its own.

Usage
-----
    from experiments import paper_style as ps
    ps.use_paper_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color=ps.C_OURS)
    ps.save_paper(fig, "results/figures/headline_gap")   # .pdf + .png (preview)
"""
from __future__ import annotations

from matplotlib import rc, rcParams

LABELSIZE = 12         # axis labels, panel titles
C_GRAY = "#666666"     # neutral annotations, reference lines, bounds
C_OURS = "#FF5000"     # solid, with markers, the only saturated line
BOUND_STYLES = {
    "oracle": dict(color="k", ls="--", lw=1.0),     # best achievable (drawn as the y=0 line)
    "random": dict(color=C_GRAY, ls=":", lw=1.1),   # naive lower bound
}

# Muted qualitative set for baselines — avoids orange, which is reserved for C_OURS.
# Known baselines get a fixed colour; unregistered names fall back to a deterministic
# cycle over the same set.
_BASELINE_CYCLE = ["#4C72B0", "#55A868", "#8172B3", "#937860", "#64B5CD"]
BASELINE_COLORS = {
    "two_stage": "#8896AB",        # muted blue-gray — the real competitor, not a bound
    "MAF": _BASELINE_CYCLE[1],
    "NSF": _BASELINE_CYCLE[2],
    "GNF": _BASELINE_CYCLE[0],
}


def baseline_color(name: str) -> str:
    """Colour for a baseline method, keyed by name.

    Registered baselines are stable; an unknown name gets a deterministic colour
    from the muted cycle (hashed by name so the same baseline always maps the same).
    """
    if name in BASELINE_COLORS:
        return BASELINE_COLORS[name]
    return _BASELINE_CYCLE[hash(name) % len(_BASELINE_CYCLE)]


def use_paper_style() -> None:
    """Paper mode: serif Computer Modern, CM mathtext — no LaTeX toolchain.

    ``mathtext.fontset="cm"`` renders ``$…$`` in Computer Modern, and the serif
    stack matches CM for body text, so figures read as the paper's font without
    needing ``usetex``, which is not installed everywhere. The
    difference from true LaTeX rendering is invisible at print scale.
    """
    # Computer Modern first (the paper font); fall back to always-present serifs so a
    # machine without CM renders cleanly instead of warning + defaulting.
    rc("font", **{"family": "serif", "serif": ["Computer Modern", "CMU Serif", "DejaVu Serif"]})
    rcParams["mathtext.fontset"] = "cm"
    rc("text", usetex=False)


def save_paper(fig, stem: str, dpi: int = 600) -> None:
    """Paper mode save: ``<stem>.pdf`` (goes in the paper) + ``<stem>.png`` (preview).

    Both at ``dpi`` with ``bbox_inches="tight"``. ``stem`` carries no
    extension, e.g. ``"results/figures/fig2_wavy"``.
    """
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
