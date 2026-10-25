"""Regenerate the README figures from the recorded standard-profile results (validation split).

Numbers are copied from the README tables (mean +/- sample std over 3 seeds), which were taken from
segformer_vs_unet_colab_v2.ipynb. Usage: python scripts/make_readme_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C_UNET, C_SEG = "#3a86c8", "#e4572e"
FRACS = ["100%", "25%", "10%"]

MIOU = {"U-Net (ResNet-50)": ([0.5037, 0.4731, 0.4728], [0.0069, 0.0035, 0.0090]),
        "SegFormer-B2": ([0.5125, 0.5055, 0.5110], [0.0032, 0.0100, 0.0080])}
GAP = ([0.0088, 0.0324, 0.0382], [0.0037, 0.0066, 0.0132])
DOMAIN = {  # (mean, std) per fraction
    "urban": {"U-Net (ResNet-50)": ([0.5365, 0.5276, 0.5081], [0.0170, 0.0050, 0.0057]),
              "SegFormer-B2": ([0.5753, 0.5424, 0.5378], [0.0008, 0.0055, 0.0106])},
    "rural": {"U-Net (ResNet-50)": ([0.4382, 0.3953, 0.4027], [0.0289, 0.0041, 0.0096]),
              "SegFormer-B2": ([0.4022, 0.4283, 0.4486], [0.0067, 0.0174, 0.0138])}}
EFF = {"Parameters (M)": (32.522, 27.352), "GFLOPs / 512² image": (85.513, 121.077),
       "Inference (ms / image)": (49.720, 57.477), "Training time (min)": (4440.952 / 60, 4765.857 / 60)}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)


def _bars(ax, data, ylabel, ylim=None):
    xs = np.arange(len(FRACS))
    for k, (name, col) in enumerate(zip(data, (C_UNET, C_SEG))):
        m, s = data[name]
        ax.bar(xs + (k - 0.5) * 0.38, np.array(m) * 100, 0.38, yerr=np.array(s) * 100,
               capsize=3, label=name, color=col)
    ax.set_xticks(xs); ax.set_xticklabels(FRACS); ax.set_xlabel("Training labels used")
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    _style(ax)


def fig_miou():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _bars(ax, MIOU, "Validation mIoU (%)", (40, 56)); ax.legend(frameon=False)
    ax.set_title("mIoU vs amount of labelled data (mean ± std, 3 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "miou_by_fraction.png", dpi=160); plt.close(fig)


def fig_gap():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(FRACS, np.array(GAP[0]) * 100, yerr=np.array(GAP[1]) * 100, capsize=4, color=C_SEG)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Training labels used"); ax.set_ylabel("Paired gap, SegFormer − U-Net (mIoU points)")
    ax.set_title("The gap widens as labels become scarce", fontsize=10); _style(ax)
    fig.tight_layout(); fig.savefig(OUT / "gap_by_fraction.png", dpi=160); plt.close(fig)


def fig_domain():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, dom in zip(axes, ("urban", "rural")):
        _bars(ax, DOMAIN[dom], "Validation mIoU (%)" if dom == "urban" else "", (30, 62))
        ax.set_title(dom.capitalize(), fontsize=10)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "urban_vs_rural.png", dpi=160); plt.close(fig)


def fig_efficiency():
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.2))
    for ax, (name, (u, s)) in zip(axes, EFF.items()):
        ax.bar(["U-Net", "SegFormer"], [u, s], color=[C_UNET, C_SEG])
        ax.set_title(name, fontsize=9); _style(ax)
        for i, v in enumerate((u, s)):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Cost on a Tesla T4 (100% labels, mean over seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "efficiency.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_miou, fig_gap, fig_domain, fig_efficiency):
        f()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
