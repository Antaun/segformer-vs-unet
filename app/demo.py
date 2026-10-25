"""Land-Cover Detective: a Gradio demo that compares the trained U-Net and SegFormer-B2 on one satellite image.

    pip install -r requirements.txt && pip install -r app/requirements.txt && pip install -e . --no-deps
    python app/demo.py --unet-ckpt experiments/unet_r50_f1.0_s0/checkpoints/best.pt \
                       --segformer-ckpt experiments/segformer_b2_f1.0_s0/checkpoints/best.pt \
                       --examples path/to/some/LoveDA/Val/images_png --share
    python app/demo.py --smoke      # tiny random models, only checks that the UI runs (predictions are meaningless)

Checkpoints are the `best.pt` files written by `dl_research.train` (a dict with a "model" key).
"""
from __future__ import annotations

import argparse
import glob
import io
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dl_research.config import load_config
from dl_research.data import IMAGENET_MEAN, IMAGENET_STD
from dl_research.models import build_model

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
FRIENDLY = {"background": "🟫 Other stuff", "building": "🏠 Buildings", "road": "🛣️ Roads",
            "water": "💧 Water", "barren": "🏜️ Bare land", "forest": "🌳 Forest",
            "agriculture": "🌾 Farmland"}
PLAIN = {c: c.capitalize() for c in CLASSES}            # emoji-free labels for matplotlib
HEX = ["#d9d4c7", "#e4572e", "#f2c14e", "#3a86c8", "#a78a7f", "#2d6a4f", "#b7e27a"]
PALETTE = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in HEX], dtype=np.uint8)
C_UNET, C_SEG = "#3a86c8", "#e4572e"
INK, GOLD, CLAY = "#14213d", "#e9a820", "#c8553d"
CARD_BG, CARD_INK, CARD_GRID = "#f6f4ec", "#14213d", "#d8d4c4"

# Validation mIoU (%), mean over 3 seeds: the README's standard-profile results.
SCORE_LABELS = ["100% labels", "25% labels", "10% labels"]
SCORE_UNET, SCORE_SEG = [50.37, 47.31, 47.28], [51.25, 50.55, 51.10]


# ---------- build ----------
def build_trained(model_name: str, ckpt: str | None, device: str, smoke: bool = False):
    """Build a model with the training code's own factory and load its best checkpoint."""
    overrides = ["pretrained=false", f"smoke={'true' if smoke else 'false'}"]
    cfg = load_config([str(ROOT / "configs/base.yaml"), str(ROOT / f"configs/{model_name}.yaml")], overrides)
    model = build_model(cfg, num_classes=7)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


# ---------- compute ----------
def preprocess(img: np.ndarray, device: str) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)[None].to(device)


@torch.no_grad()
def predict(model, x: torch.Tensor) -> np.ndarray:
    """Both wrappers return logits at input resolution; U-Net needs H, W divisible by 32, so pad."""
    h, w = x.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    out = model(F.pad(x, (0, pw, 0, ph), mode="reflect"))[..., :h, :w]
    return out.argmax(1)[0].cpu().numpy()


def class_share(mask): return np.bincount(mask.ravel(), minlength=7) / mask.size * 100
def disagreement(a, b): return a != b


def highlight_classes(img, mask, keep):
    out = np.array(Image.fromarray(img).convert("L").convert("RGB"), dtype=np.float32) * 0.6 + 60
    for c in keep:
        out[mask == c] = PALETTE[c]
    return out.astype(np.uint8)


def render_view(img, mask, view, alpha):
    if view == "Original photo":
        return img
    if view == "Colour map only":
        return PALETTE[mask]
    return (img * (1 - alpha) + PALETTE[mask] * alpha).astype(np.uint8)


def to_label_ids(gt):
    """LoveDA mask: 0 = no-data (ignored -> 255), 1..7 -> classes 0..6."""
    gt = (gt if gt.ndim == 2 else gt[..., 0]).astype(np.int64)
    return np.where(gt == 0, 255, gt - 1)


def per_class_iou(pred, gt):
    valid, ious = gt != 255, []
    for c in range(7):
        inter = ((pred == c) & (gt == c) & valid).sum()
        union = (((pred == c) | (gt == c)) & valid).sum()
        ious.append(inter / union * 100 if union else np.nan)
    return np.array(ious)


def describe_scene(share_a, share_b, agree, kid):
    top = int(np.argmax((share_a + share_b) / 2))
    name, pct = FRIENDLY[CLASSES[top]], (share_a[top] + share_b[top]) / 2
    if kid:
        verdict = ("They almost always agree!" if agree > 85 else
                   "They mostly agree, but argue about some spots." if agree > 70 else
                   "This one is tricky: the two robots disagree a lot!")
        return (f"### 🤖 What the robots see\nMost of this picture is **{name}** (about {pct:.0f}%). {verdict}\n\n"
                "The **clay-red spots** on the third picture are places where the robots gave different answers.")
    return f"### Summary\nDominant class: **{name}** (~{pct:.0f}% of pixels). Pixel agreement between models: **{agree:.1f}%**."


# ---------- plot ----------
def style_chart(fig, ax, ylabel=None):
    """Light card + dark text, so labels stay readable in both Gradio light and dark mode."""
    fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=CARD_INK, labelsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=CARD_INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(CARD_INK)
    ax.grid(axis="y", color=CARD_GRID, linewidth=0.8); ax.set_axisbelow(True)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor(CARD_BG); leg.get_frame().set_edgecolor(CARD_GRID)
        for t in leg.get_texts(): t.set_color(CARD_INK)
    fig.tight_layout()
    return fig


def _grouped_bars(labels, a, b, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, a, 0.4, label="U-Net (CNN)", color=C_UNET)
    ax.bar(xs + 0.2, b, 0.4, label="SegFormer-B2 (Transformer)", color=C_SEG)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30 if len(labels) > 3 else 0,
                                          ha="right" if len(labels) > 3 else "center")
    if ylim: ax.set_ylim(*ylim)
    ax.legend(frameon=True)
    return style_chart(fig, ax, ylabel)


def plot_class_share(a, b): return _grouped_bars([PLAIN[c] for c in CLASSES], a, b, "% of the picture")
def plot_iou(a, b): return _grouped_bars([PLAIN[c] for c in CLASSES], np.nan_to_num(a), np.nan_to_num(b),
                                         "IoU score (higher = better)")
def plot_scoreboard(): return _grouped_bars(SCORE_LABELS, SCORE_UNET, SCORE_SEG, "mIoU (%)", ylim=(40, 56))


def fig_to_image(fig):
    """Opaque RGB array, immune to dark-mode transparency."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig); buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ---------- look & feel ----------
THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(*["#f4f8f1", "#e3ecde", "#cfdccb", "#aec4a8", "#86a37f", "#5f8259",
                                  "#476843", "#36503a", "#273a2b", "#1b291f", "#111a14"]),
    neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(body_background_fill="#eef2ec", block_background_fill="#fbfcf9", block_border_width="1px",
      block_border_color="#d5ddd0", block_radius="10px", block_shadow="none",
      button_primary_background_fill=INK, button_primary_background_fill_hover="#26365f",
      button_primary_text_color="#ffffff")

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;1,9..144,400&display=swap');
:root {{ --ink:{INK}; --soft:#46566f; --cap:#5b6b82; --clay:{CLAY}; --gold:{GOLD}; --rule:{INK}; }}
.dark {{ --ink:#f1efe6; --soft:#aab6cc; --cap:#8fa0ba; --clay:#e8806a; --rule:#f1efe6; }}
.gradio-container {{max-width: 1180px !important; margin: auto;}}
h1,h2,h3,.serif {{font-family:'Fraunces',Georgia,serif !important; color:var(--ink) !important;}}
.hero {{padding:34px 4px 18px; border-bottom:2px solid var(--rule); margin-bottom:22px;}}
.eyebrow {{font-size:.78rem; letter-spacing:.16em; text-transform:uppercase; color:var(--clay); font-weight:600;}}
.hero h1 {{font-size:2.6rem; line-height:1.1; margin:.2em 0;}}
.hero .sub {{font-family:'Fraunces',serif; font-style:italic; font-size:1.15rem; color:var(--soft);}}
.step {{display:flex; align-items:center; gap:16px; margin:28px 0 4px;}}
.step .num {{font-family:'Fraunces',serif; font-size:2rem; line-height:1; color:var(--gold); font-weight:700; min-width:2.4rem;}}
.step h2 {{margin:0 !important; padding:0 !important; font-size:1.7rem; line-height:1;}}
.analogy {{font-family:'Fraunces',serif; font-style:italic; color:var(--soft); margin:0 0 12px 3.4rem;}}
.caption {{font-family:'Fraunces',serif; font-style:italic; font-size:.9rem; color:var(--cap); text-align:center;}}
.chip {{display:inline-block; padding:3px 11px; margin:3px; border-radius:6px; font-size:.82rem;
       color:#14213d; border:1px solid rgba(20,33,61,.25);}}
.note {{border-left:4px solid var(--gold); background:rgba(233,168,32,.12); padding:10px 14px; border-radius:0 8px 8px 0;}}
footer {{display:none !important;}}
"""

HERO = """<div class="hero">
<div class="eyebrow">Satellite image segmentation · CNN vs Transformer</div>
<h1>Land-Cover Detective</h1>
<div class="sub">Two AIs look at the same satellite photo. One reads it like a map with a magnifying glass,
the other glances at the whole picture at once. Who sees the land better?</div></div>"""

HOW_IT_WORKS = """
### How does it work? 🧠
1. **The photo** is a view of Earth from above. Every tiny square (pixel) is about 30 cm wide.
2. **The robot** is a neural network that has seen thousands of photos where humans already painted the answer:
   "this is water, this is a road…".
3. **The guess:** for each pixel it picks one of 7 labels, which gives the colourful map.

### The two contestants
| | 🧱 U-Net (CNN) | 🔭 SegFormer-B2 (Transformer) |
|---|---|---|
| Idea | Looks at small neighbourhoods, like reading with a magnifying glass | Looks at the whole picture at once and links far-apart parts |
| Size | 32.5 M parameters | 27.4 M parameters |
| Cost | about 86 GFLOPs per 512² image | about 121 GFLOPs (≈42% more) |
"""

SCORE_MD = """
### What we found 🏆
- With **all** training data, SegFormer-B2 was slightly better (51.25% vs 50.37% mIoU).
- With only **10–25%** of the labels, SegFormer-B2 stayed strong while U-Net dropped about 3 points.
- In **rural** scenes with all the data U-Net was ahead; in **urban** scenes SegFormer-B2 was.
- One dataset (LoveDA), 3 random seeds, validation split (also used for tuning): treat the numbers as indicative.
"""


def step_html(num, title, analogy):
    return (f'<div class="step"><span class="num">{num}</span><h2>{title}</h2></div>'
            f'<p class="analogy">{analogy}</p>')


def legend_html():
    return "".join(f'<span class="chip" style="background:{h}">{FRIENDLY[c]}</span>' for c, h in zip(CLASSES, HEX))


# ---------- UI ----------
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6


def _blocks_style():
    """Gradio <6 takes theme/css in Blocks(); Gradio 6 takes them in launch()."""
    return {} if GRADIO_6 else {"theme": THEME, "css": CSS}


def _launch_style():
    return {"theme": THEME, "css": CSS} if GRADIO_6 else {}


def make_app(unet, segformer, device, examples=None):
    def run(img, view, alpha, keep_names, kid, gt_img, progress=gr.Progress()):
        if img is None:
            gr.Warning("Pick a satellite photo or click an example first!")
            return [None] * 7
        img = np.asarray(Image.fromarray(img).convert("RGB"))
        progress(0.1, desc="Waking up the robots…")
        x = preprocess(img, device)
        progress(0.35, desc="🧱 U-Net is squinting at the pixels…")
        ma = predict(unet, x)
        progress(0.7, desc="🔭 SegFormer is taking in the big picture…")
        mb = predict(segformer, x)
        progress(0.9, desc="Comparing notes…")
        diff = disagreement(ma, mb); agree = 100 - diff.mean() * 100
        if keep_names:
            keep = [i for i, c in enumerate(CLASSES) if FRIENDLY[c] in keep_names]
            va, vb = highlight_classes(img, ma, keep), highlight_classes(img, mb, keep)
        else:
            va, vb = render_view(img, ma, view, alpha), render_view(img, mb, view, alpha)
        dimg = (img * 0.45).astype(np.uint8); dimg[diff] = [200, 85, 61]
        iou_plot = None
        if gt_img is not None:
            gt = to_label_ids(gt_img)
            if gt.shape == ma.shape:
                iou_plot = fig_to_image(plot_iou(per_class_iou(ma, gt), per_class_iou(mb, gt)))
            else:
                gr.Warning("The answer key must have the same size as the photo; skipping the score.")
        return (va, vb, dimg, describe_scene(class_share(ma), class_share(mb), agree, kid),
                fig_to_image(plot_class_share(class_share(ma), class_share(mb))), iou_plot, f"{agree:.1f}%")

    with gr.Blocks(title="Land-Cover Detective", **_blocks_style()) as app:
        gr.HTML(HERO)
        gr.HTML(step_html("01", "Choose a photo", "Like handing two detectives the same aerial snapshot."))
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Image(type="numpy", label="Satellite photo", height=340)
                if examples:
                    gr.Examples(examples=examples, inputs=inp, cache_examples=False, label="Or start with an example")
            with gr.Column(scale=2):
                kid = gr.Checkbox(True, label="🧒 Explain it like I'm 10")
                view = gr.Radio(["Overlay on photo", "Colour map only", "Original photo"],
                                value="Overlay on photo", label="How to show the answer")
                alpha = gr.Slider(0, 1, 0.55, label="Colour strength")
                keep = gr.CheckboxGroup([FRIENDLY[c] for c in CLASSES],
                                        label="🔦 Spotlight only… (empty = show everything)")
                gt = gr.Image(type="numpy", image_mode="L", height=110,
                              label="Optional: answer key (LoveDA mask) to score the models")
                go = gr.Button("Run the comparison →", variant="primary", size="lg")

        gr.HTML(step_html("02", "Read the maps", "Each colour is a guess: water, forest, road, roof… for every pixel."))
        with gr.Row():
            o1 = gr.Image(label="🧱 U-Net (CNN)")
            o2 = gr.Image(label="🔭 SegFormer-B2 (Transformer)")
            o3 = gr.Image(label="⚡ Where they disagree (clay red)")
        gr.HTML(f'<div style="text-align:center">{legend_html()}</div>'
                '<p class="caption">Fig. 1 · Same photo, two opinions. Red patches mark disagreement.</p>')

        gr.HTML(step_html("03", "Who's right?", "A scoreboard for the argument, and the answer key if you uploaded one."))
        with gr.Row():
            with gr.Column(scale=3):
                txt = gr.Markdown(elem_classes="note")
            with gr.Column(scale=1):
                agree_box = gr.Textbox(label="Pixels where both models agree", interactive=False)
        with gr.Row():
            p1 = gr.Image(label="What's in the picture?")
            p2 = gr.Image(label="Score vs answer key (needs an answer key)")
        go.click(run, [inp, view, alpha, keep, kid, gt], [o1, o2, o3, txt, p1, p2, agree_box])

        gr.HTML(step_html("04", "The big experiment",
                          "We repeated this three times with less and less training data."))
        with gr.Row():
            gr.Image(value=fig_to_image(plot_scoreboard()), label="Accuracy by amount of labelled data",
                     interactive=False)
            gr.Markdown(SCORE_MD)
        gr.HTML('<p class="caption">Fig. 2 · Validation mIoU, mean of 3 seeds (LoveDA).</p>')
        with gr.Accordion("📚 How do these robots actually work?", open=False):
            gr.Markdown(HOW_IT_WORKS)
        gr.HTML('<p class="caption" style="margin-top:30px">MSc Applied AI project · data: LoveDA</p>')
    return app


def find_examples(folder: str | None, n: int = 6) -> list[str]:
    if not folder:
        return []
    files = sorted(glob.glob(str(Path(folder) / "**" / "*.png"), recursive=True)
                   + glob.glob(str(Path(folder) / "**" / "*.jpg"), recursive=True))
    files = [f for f in files if "mask" not in f.lower()]
    return files[:: max(1, len(files) // n)][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unet-ckpt"); ap.add_argument("--segformer-ckpt")
    ap.add_argument("--examples", help="folder of example images (LoveDA images_png folders work)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny random models, UI check only")
    ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if not a.smoke and not (a.unet_ckpt and a.segformer_ckpt):
        ap.error("pass --unet-ckpt and --segformer-ckpt (or --smoke)")
    unet = build_trained("unet_r50", a.unet_ckpt, a.device, a.smoke)
    seg = build_trained("segformer_b2", a.segformer_ckpt, a.device, a.smoke)
    make_app(unet, seg, a.device, find_examples(a.examples) or None).launch(share=a.share, server_port=a.port, **_launch_style())


if __name__ == "__main__":
    main()
