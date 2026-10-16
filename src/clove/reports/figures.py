"""clove.reports.figures — plots drawn ONLY from saved prediction-derived tables.

matplotlib is optional; without it, figures are skipped and the report says so.
"""
from __future__ import annotations

from pathlib import Path


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # noqa: BLE001
        return None


def trajectories(rows: list[dict], out: Path) -> str | None:
    """F3: old / new / unknown-rejection per episode, one panel each, line per method
    (mean over orders at the primary budget)."""
    plt = _plt()
    if plt is None or not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    keys = [("old_macro_acc", "old-class macro acc"), ("new_macro_acc", "new-class macro acc"),
            ("pending_unknown_rejection", "not-yet-admitted rejection")]
    methods = sorted({r["method"] for r in rows})
    for ax, (k, title) in zip(axes, keys):
        for m in methods:
            pts = sorted((r["episode"], r[k]) for r in rows if r["method"] == m and r.get(k) is not None)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def accuracy_vs_tokens(rows: list[dict], out: Path) -> str | None:
    """F6: open-world accuracy vs mean total tokens per image, one point per
    (method, budget)."""
    plt = _plt()
    if plt is None or not rows:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for r in rows:
        if r.get("acc") is None:
            continue
        ax.scatter(r["mean_total_tokens"], r["acc"])
        ax.annotate(f"{r['method']}/{r['budget_id']}", (r["mean_total_tokens"], r["acc"]), fontsize=7)
    ax.set_xlabel("mean total tokens / image (prompt incl. image + output)")
    ax.set_ylabel("open-world accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def memory_contrast(rows: list[dict], out: Path) -> str | None:
    """F4: F_memory(t) per persistent method, per order."""
    plt = _plt()
    if plt is None or not rows:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for key in sorted({(r["method"], r["order_id"]) for r in rows}):
        pts = sorted((r["episode"], r["F_memory"]) for r in rows
                     if (r["method"], r["order_id"]) == key and r["F_memory"] is not None)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{key[0]}·{key[1]}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("F_memory = A_fresh − A_persistent")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)
