"""Aggregate finished runs into tables (CSV + Markdown) and figures. Pure bookkeeping: it reports measured
numbers only and draws no conclusions.

    python -m dl_research.aggregate --exp-dir experiments --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl_research.loveda import CLASS_NAMES  # noqa: E402
from dl_research.utils import read_json  # noqa: E402

LABELS = {"unet_r50": "U-Net (ResNet-50)", "segformer_b2": "SegFormer-B2"}
COLORS = {"unet_r50": "tab:blue", "segformer_b2": "tab:orange"}


def load_runs(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(exp_dir.glob("*/metrics.json")):  # tuning runs live one level deeper and are excluded
        m = read_json(p)
        if not m.get("complete"):
            continue
        v, eff, tr = m["val"], m["efficiency"], m["training"]
        row = {
            "model": m["config"]["model"], "fraction": m["config"]["fraction"], "seed": m["config"]["seed"],
            "lr": m["config"]["lr"], "miou": v["overall"]["miou"], "pixel_acc": v["overall"]["pixel_acc"],
            "mf1": v["overall"]["mf1"], "urban_miou": v["urban"]["miou"], "rural_miou": v["rural"]["miou"],
            "final_epoch_miou": tr["final_epoch_val_miou"], "best_epoch": tr["best_epoch"],
            "params_m": m["model_stats"]["params_total"] / 1e6, "gflops": eff["gflops_per_image"],
            "train_time_s": tr["train_time_s"], "ms_per_image": eff["inference_ms_per_image"],
            "images_per_s": eff["inference_images_per_s"], "device": m["env"]["device_name"],
            "n_train_images": tr["n_train_images"],
        }
        row.update({f"iou_{c}": v["overall"]["iou"][c] for c in CLASS_NAMES})
        row.update({f"f1_{c}": v["overall"]["f1"][c] for c in CLASS_NAMES})
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["model", "fraction"])
    out = g[cols].agg(["mean", lambda s: s.std(ddof=1)])
    out.columns = [f"{c}_{'mean' if s == 'mean' else 'std'}" for c, s in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def pm(mean, std, n, digits=4) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if n > 1 and not pd.isna(std) else f"{mean:.{digits}f} (n=1)"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in df.itertuples(index=False))


def paired_gap(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """SegFormer-B2 minus U-Net per fraction, paired by seed (only seeds present for both models)."""
    rows = []
    for frac, d in df.groupby("fraction"):
        w = d.pivot(index="seed", columns="model", values=col).dropna()
        if {"unet_r50", "segformer_b2"} <= set(w.columns) and len(w):
            diff = w["segformer_b2"] - w["unet_r50"]
            rows.append({"fraction": frac, "n_pairs": len(diff), "gap_mean": diff.mean(),
                         "gap_std": diff.std(ddof=1) if len(diff) > 1 else np.nan})
    return pd.DataFrame(rows)


def make_tables(df: pd.DataFrame, out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    main_cols = ["miou", "pixel_acc", "mf1", "urban_miou", "rural_miou"]
    agg = mean_std(df, main_cols + ["final_epoch_miou"] + [f"iou_{c}" for c in CLASS_NAMES])
    agg.to_csv(out / "summary.csv", index=False)
    df.to_csv(out / "all_runs.csv", index=False)

    def table(cols):
        rows = []
        for r in agg.sort_values(["fraction", "model"], ascending=[False, True]).itertuples():
            rows.append({"fraction": f"{r.fraction:g}", "model": LABELS.get(r.model, r.model), "seeds": r.n_seeds,
                         **{c: pm(getattr(r, f"{c}_mean"), getattr(r, f"{c}_std"), r.n_seeds) for c in cols}})
        return pd.DataFrame(rows)

    gap = paired_gap(df, "miou")
    gap_dom = {d: paired_gap(df, f"{d}_miou") for d in ("urban", "rural")}
    eff = df[df.fraction == df.fraction.max()].groupby("model")[["params_m", "gflops", "train_time_s", "ms_per_image",
                                                                  "images_per_s"]].mean().reset_index()
    eff["model"] = eff["model"].map(lambda m: LABELS.get(m, m))
    eff = eff.round(3)

    def gap_md(g):
        if g.empty:
            return "_no seeds with both models yet_\n"
        g = g.assign(gap=[pm(m, s, n) for m, s, n in zip(g.gap_mean, g.gap_std, g.n_pairs)])
        return md_table(g[["fraction", "n_pairs", "gap"]])

    md = ["# Results (validation split)\n",
          f"Generated from {len(df)} completed runs. mean ± sample std over seeds (ddof=1). All numbers are measured on "
          "the LoveDA **validation** split, which was also used for LR tuning and checkpoint selection (see README).\n",
          "## Overall metrics\n", md_table(table(["miou", "pixel_acc", "mf1"])),
          "\n## Urban vs rural mIoU\n", md_table(table(["urban_miou", "rural_miou"])),
          "\n## Per-class IoU\n", md_table(table([f"iou_{c}" for c in CLASS_NAMES])),
          "\n## Paired gap, SegFormer-B2 minus U-Net (mIoU, paired by seed)\n", gap_md(gap)]
    for d, g in gap_dom.items():
        md += [f"\n### {d} domain\n", gap_md(g)]
    md += [f"\n## Efficiency (runs at fraction {df.fraction.max():g}, mean over seeds)\n", md_table(eff),
           "\nGFLOPs counted by torch FlopCounterMode for one 3x{0}x{0} image (1 MAC = 2 FLOPs); inference timing is "
           "fp32, batch size 1, same device for both models. Devices: ".format(int(512)) + ", ".join(sorted(df.device.unique())) + "\n"]
    text = "".join(md)
    (out / "summary.md").write_text(text)
    gap.to_csv(out / "gap_vs_fraction.csv", index=False)
    return text


def make_plots(df: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    agg = mean_std(df, ["miou", "urban_miou", "rural_miou"])
    fracs = sorted(df.fraction.unique())

    fig, ax = plt.subplots(figsize=(5, 4))
    for m, d in agg.groupby("model"):
        d = d.sort_values("fraction")
        ax.errorbar(d.fraction * 100, d.miou_mean, yerr=d.miou_std.fillna(0), marker="o", capsize=3,
                    label=LABELS.get(m, m), color=COLORS.get(m))
        for _, r in df[df.model == m].iterrows():
            ax.scatter(r.fraction * 100, r.miou, s=8, color=COLORS.get(m), alpha=0.3)
    ax.set_xscale("log"); ax.set_xticks([f * 100 for f in fracs]); ax.set_xticklabels([f"{f*100:g}%" for f in fracs])
    ax.set_xlabel("training labels"); ax.set_ylabel("val mIoU"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "miou_vs_fraction.png", dpi=150); plt.close(fig)

    gap = paired_gap(df, "miou")
    if not gap.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(gap.fraction * 100, gap.gap_mean, yerr=gap.gap_std.fillna(0), marker="o", capsize=3, color="k")
        ax.axhline(0, color="gray", lw=0.8); ax.set_xscale("log")
        ax.set_xticks([f * 100 for f in gap.fraction]); ax.set_xticklabels([f"{f*100:g}%" for f in gap.fraction])
        ax.set_xlabel("training labels"); ax.set_ylabel("mIoU gap (SegFormer-B2 − U-Net)"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "gap_vs_fraction.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(fracs), figsize=(3.2 * len(fracs), 4), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], fracs):
        d = agg[agg.fraction == f]
        for j, dom in enumerate(("urban", "rural")):
            for k, m in enumerate(sorted(d.model)):
                r = d[d.model == m].iloc[0]
                ax.bar(j + 0.2 * (k - 0.5), r[f"{dom}_miou_mean"], 0.2, yerr=0 if pd.isna(r[f"{dom}_miou_std"]) else r[f"{dom}_miou_std"],
                       color=COLORS.get(m), label=LABELS.get(m, m) if j == 0 else None, capsize=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["urban", "rural"]); ax.set_title(f"{f*100:g}% labels")
    axes[0][0].set_ylabel("val mIoU"); axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "domain_miou.png", dpi=150); plt.close(fig)

    top = df[df.fraction == df.fraction.max()].groupby("model")[["miou", "params_m", "gflops", "ms_per_image"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (col, lab) in zip(axes, [("params_m", "parameters (M)"), ("gflops", "GFLOPs / image"),
                                     ("ms_per_image", "inference ms / image")]):
        for m, r in top.iterrows():
            ax.scatter(r[col], r.miou, s=60, color=COLORS.get(m), label=LABELS.get(m, m))
        ax.set_xlabel(lab); ax.set_ylabel("val mIoU"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "efficiency.png", dpi=150); plt.close(fig)


def aggregate_runs(exp_dir="experiments", out="results") -> pd.DataFrame:
    df = load_runs(Path(exp_dir))
    if df.empty:
        raise SystemExit(f"no completed runs (metrics.json) under {exp_dir}")
    make_tables(df, Path(out))
    make_plots(df, Path(out))
    print(f"wrote tables and figures for {len(df)} runs to {out}/")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default="experiments")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    aggregate_runs(args.exp_dir, args.out)


if __name__ == "__main__":
    main()
