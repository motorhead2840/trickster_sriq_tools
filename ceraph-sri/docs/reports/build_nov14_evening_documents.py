"""Two 14 November evening documents.

How-it-works is for counsel. The technical record is for the company file and
for an early note after the provisional draft is approved.

The pages use the warm manuscript palette of the letter to Swami Tyageeswara
and Swami Sandeepananda Giri: parchment, maroon, gold, and forest ink. They do
not use the cool navy of the note for home.

Histograms are redrawn from the evening telemetry. Each figure keeps its legend
above the axes and its tick labels outside the bars, and each picture is placed
at its true aspect ratio so a caption cannot sit on the drawing.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark-results" / "rtx6000-evening"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures" / "nov14-evening"

PARCHMENT = "#f4ead7"
INK = "#3a2416"
MAROON = "#7c2d2d"
GOLD = "#a67c42"
SAFFRON = "#c56a1a"
FOREST = "#2c4a3e"
CARD = "#fbf6ec"
RULE = "#e2d3bc"
BLUE = "#2f4f6f"

P = colors.HexColor(PARCHMENT)
I = colors.HexColor(INK)
M = colors.HexColor(MAROON)
G = colors.HexColor(GOLD)
F = colors.HexColor(FOREST)


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def style_3d(ax) -> None:
    ax.set_facecolor(PARCHMENT)
    ax.tick_params(colors=INK, labelsize=10, pad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(INK)
        axis.label.set_fontsize(11)
        axis.pane.set_facecolor(CARD)
        axis.pane.set_edgecolor(RULE)
        axis.pane.set_alpha(0.95)
        axis._axinfo["grid"]["color"] = RULE
        axis._axinfo["grid"]["linewidth"] = 0.6


def save_fig(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)
    return path


def legend_above(fig, handles) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=min(4, len(handles)),
               frameon=False, fontsize=11, labelcolor=INK)


def bar3(ax, xs, y, heights, color, depth=0.55, width=0.62) -> None:
    ax.bar3d(xs, np.full(len(xs), y), np.zeros(len(xs)), width, depth, heights, color=color, shade=True)


def draw_figures() -> dict[str, Path]:
    one_v = load("one_validation.json")
    one_w = load("one_workstation.json")
    one_s = load("one_soak.json")
    four_h = load("four_heavy.json")
    four_s = load("four_soak.json")
    paths: dict[str, Path] = {}

    fig = plt.figure(figsize=(11.2, 7.2), facecolor=PARCHMENT)
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    labels = ["identity", "average", "raw", "flux"]
    scores = one_v["quality"]
    for row, (score, color) in enumerate(zip(scores, [BLUE, SAFFRON])):
        vals = [score["identity_mse"], score["moving_average_mse"], score["raw_mse"], score["flux_mse"]]
        bar3(ax, np.arange(4), row, vals, color, depth=0.45, width=0.55)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["offset 0", "offset 5"], color=INK)
    ax.set_zlabel("mean squared error")
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.5, 0.9, 0.85))
    legend_above(fig, [Patch(facecolor=BLUE, label="Offset 0"), Patch(facecolor=SAFFRON, label="Offset 5")])
    fig.suptitle("Error left behind  ·  lower is better", color=INK, fontsize=15, y=0.995)
    fig.subplots_adjust(top=0.78, bottom=0.08)
    paths["quality"] = save_fig(fig, "quality_mse_3d.png")

    rows = one_w["rows"]
    fig = plt.figure(figsize=(11.2, 7.2), facecolor=PARCHMENT)
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    xs = np.arange(len(rows))
    bar3(ax, xs, 0, [r["triton_ms"] for r in rows], FOREST, depth=0.4, width=0.45)
    bar3(ax, xs, 1, [r["avg_pool_ms"] for r in rows], SAFFRON, depth=0.4, width=0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['length']:,}" for r in rows], color=INK, rotation=18, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Triton", "moving average"], color=INK)
    ax.set_zlabel("milliseconds")
    ax.view_init(elev=24, azim=-58)
    legend_above(fig, [Patch(facecolor=FOREST, label="Triton kernel"), Patch(facecolor=SAFFRON, label="Moving-average pool")])
    fig.suptitle("One board  ·  time against length", color=INK, fontsize=15, y=0.995)
    fig.subplots_adjust(top=0.78, bottom=0.12)
    paths["latency"] = save_fig(fig, "latency_3d.png")

    fig = plt.figure(figsize=(11.2, 7.0), facecolor=PARCHMENT)
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    heights = [four_h["queued_on_one_board_ms"], four_h["before_ms"]]
    ax.bar3d([0.0], [0], [0], 0.7, 0.5, [heights[0]], color=MAROON, shade=True)
    ax.bar3d([1.6], [0], [0], 0.7, 0.5, [heights[1]], color=FOREST, shade=True)
    ax.set_xticks([0.35, 1.95])
    ax.set_xticklabels(["four jobs\nqueued", "four jobs\nat once"], color=INK)
    ax.set_yticks([])
    ax.set_zlabel("milliseconds")
    ax.view_init(elev=22, azim=-62)
    legend_above(fig, [Patch(facecolor=MAROON, label="Queued on one board, 29.53 ms"), Patch(facecolor=FOREST, label="One job per board, 7.39 ms")])
    fig.suptitle("Four heavy jobs  ·  wall time", color=INK, fontsize=15, y=0.995)
    fig.subplots_adjust(top=0.78)
    paths["jobs"] = save_fig(fig, "jobs_3d.png")

    samples = one_s["samples"][::6]
    times = [s["t_s"] for s in samples]
    xs = np.arange(len(samples))

    def series_chart(key: str, title: str, zlabel: str, color: str, filename: str) -> Path:
        fig = plt.figure(figsize=(11.2, 6.6), facecolor=PARCHMENT)
        ax = fig.add_subplot(111, projection="3d")
        style_3d(ax)
        bar3(ax, xs, 0, [s[key] for s in samples], color, depth=0.35, width=0.55)
        ax.set_xlabel("minutes into the soak")
        ax.set_xticks(xs[::2])
        ax.set_xticklabels([f"{times[i] / 60:.1f}" for i in range(0, len(times), 2)], color=INK)
        ax.set_yticks([])
        ax.set_zlabel(zlabel)
        ax.view_init(elev=22, azim=-68)
        fig.suptitle(title, color=INK, fontsize=15)
        fig.subplots_adjust(top=0.88, bottom=0.08)
        return save_fig(fig, filename)

    paths["soak_power"] = series_chart("power_w", "One board  ·  five minutes of power", "watts", MAROON, "soak_one_power_3d.png")
    paths["soak_temp"] = series_chart("temperature_c", "One board  ·  five minutes of temperature", "degrees C", GOLD, "soak_one_temp_3d.png")
    paths["soak_util"] = series_chart("util_pct", "One board  ·  five minutes of how busy the chip was", "percent busy", FOREST, "soak_one_util_3d.png")

    palette = [BLUE, FOREST, GOLD, MAROON]
    boards = four_s["boards"]
    soak_samples = four_s["samples"][::8]

    def four_chart(metric: str, title: str, zlabel: str, filename: str) -> Path:
        fig = plt.figure(figsize=(11.4, 7.2), facecolor=PARCHMENT)
        ax = fig.add_subplot(111, projection="3d")
        style_3d(ax)
        handles = []
        for i, board in enumerate(boards):
            vals = [s["gpus"][i][metric] for s in soak_samples]
            bar3(ax, np.arange(len(vals)), i, vals, palette[i], depth=0.45, width=0.55)
            handles.append(Patch(facecolor=palette[i], label=f"Board {i}"))
        ax.set_xlabel("minutes into the soak")
        step = max(1, len(soak_samples) // 6)
        ax.set_xticks(np.arange(len(soak_samples))[::step])
        ax.set_xticklabels([f"{soak_samples[i]['t_s'] / 60:.1f}" for i in range(0, len(soak_samples), step)], color=INK)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["0", "1", "2", "3"], color=INK)
        ax.set_ylabel("board")
        ax.set_zlabel(zlabel)
        ax.view_init(elev=22, azim=-62)
        legend_above(fig, handles)
        fig.suptitle(title, color=INK, fontsize=15, y=0.995)
        fig.subplots_adjust(top=0.78, bottom=0.08)
        return save_fig(fig, filename)

    paths["four_power"] = four_chart("power_w", "Four boards  ·  five minutes of power", "watts", "soak_four_power_3d.png")
    paths["four_temp"] = four_chart("temperature_c", "Four boards  ·  five minutes of temperature", "degrees C", "soak_four_temp_3d.png")

    heavy = four_h["samples"]
    fig = plt.figure(figsize=(11.2, 7.0), facecolor=PARCHMENT)
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    handles = []
    for i in range(4):
        vals = [s["gpus"][i]["power_w"] for s in heavy]
        bar3(ax, np.arange(len(vals)), i, vals, palette[i], depth=0.4, width=0.5)
        handles.append(Patch(facecolor=palette[i], label=f"Board {i}"))
    ax.set_xlabel("sample, about every 2 seconds")
    ax.set_xticks(np.arange(len(heavy)))
    ax.set_ylabel("board")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"], color=INK)
    ax.set_zlabel("watts")
    ax.view_init(elev=22, azim=-62)
    legend_above(fig, handles)
    fig.suptitle("Four heavy jobs together  ·  a short burn", color=INK, fontsize=15, y=0.995)
    fig.subplots_adjust(top=0.78)
    paths["heavy_power"] = save_fig(fig, "heavy_four_power_3d.png")

    # Rows are laboratories. Columns are lengths. A missing cell is simply not drawn.
    platform_rows = [
        ("Colab", SAFFRON, {0: 0.0082, 1: 0.0062}),
        ("DGX morning", FOREST, {0: 0.0109, 1: 0.0108, 2: 0.0107}),
        ("DGX evening", MAROON, {1: 0.0119, 2: 0.0119}),
    ]
    fig = plt.figure(figsize=(11.6, 7.4), facecolor=PARCHMENT)
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    handles = []
    for row, (label, color, values) in enumerate(platform_rows):
        xs = np.array(sorted(values))
        bar3(ax, xs, row, [values[i] for i in xs], color, depth=0.45, width=0.55)
        handles.append(Patch(facecolor=color, label=label))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["8,192", "10,000", "1,048,576"], color=INK)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Colab", "morning", "evening"], color=INK)
    ax.yaxis.label.set_text("")
    ax.tick_params(axis="y", pad=14)
    ax.set_zlabel("milliseconds")
    ax.view_init(elev=24, azim=-58)
    legend_above(fig, handles)
    fig.suptitle("Blackwell kernel time  ·  Colab and DGX", color=INK, fontsize=15, y=0.995)
    fig.subplots_adjust(top=0.76, bottom=0.08)
    paths["platforms"] = save_fig(fig, "platforms_latency_3d.png")

    fig = plt.figure(figsize=(11.2, 7.0), facecolor=PARCHMENT)
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    names = ["MMSE", "Wiener", "average", "median"]
    errs = [0.6137, 0.6944, 1.0881, 1.2352]
    err_colors = [FOREST, BLUE, SAFFRON, MAROON]
    for i, (height, color) in enumerate(zip(errs, err_colors)):
        ax.bar3d([i], [0], [0], 0.62, 0.45, [height], color=color, shade=True)
    ax.set_xticks(np.arange(len(names)) + 0.3)
    ax.set_xticklabels(names, color=INK)
    ax.set_yticks([])
    ax.set_zlabel("mean squared error")
    ax.view_init(elev=22, azim=-62)
    fig.suptitle("AWS Braket campaign  ·  11 November  ·  lower is better", color=INK, fontsize=15)
    fig.subplots_adjust(top=0.86, bottom=0.08)
    paths["braket"] = save_fig(fig, "braket_mse_3d.png")
    return paths


def equation(name: str, line: str, caption: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 2.8), facecolor=PARCHMENT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.62, line, ha="center", va="center", fontsize=30, color=INK)
    ax.text(0.5, 0.22, caption, ha="center", va="center", fontsize=12, color=FOREST, fontstyle="italic")
    path = FIG / name
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return path


def draw_equations() -> dict[str, Path]:
    return {
        "omega": equation("eq_omega.png", r"$\Omega=\sum \sigma\,\beta\,(v+\xi)\,\Delta\omega$", "A discrete stand-in for the continuous omega-dit."),
        "raw": equation("eq_raw.png", r"$A(\theta)=\theta\,e^{-\lambda\xi}$", "Raw damping. The whole series is shrunk."),
        "flux": equation("eq_flux.png", r"$A(y)=\mu+(y-\mu)\,e^{-\lambda\xi}$", "Flux damping. The center stays. Only the wiggle shrinks."),
        "closed": equation("eq_closed.png", r"$A_{40}=\mu+(y-\mu)\,e^{-40\xi}$", "Forty passes with a fixed center and a fixed instability."),
        "jvp": equation("eq_jvp.png", r"$dA=e^{-\lambda\xi}\,d\theta-\lambda\theta e^{-\lambda\xi}\,d\xi$", "The derivative JAX checks against the kernel."),
        "sphota": equation("eq_sphota.png", r"$128,\;256,\;512,\;1024$", "The @sphota ladder, chosen from the mean absolute instability."),
        "sign": equation("eq_sign.png", r"$\mathrm{bits}\oplus 0\mathrm{x}80000000$", "Sign-bit negation. Exact in the float32 encoding."),
    }


def flow_chain(name: str, title: str, nodes: list[tuple[str, str]]) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    n = len(nodes)
    fig, ax = plt.subplots(figsize=(2.55 * n, 3.6), facecolor=PARCHMENT)
    ax.set_xlim(0, n * 2.55)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    fills = [MAROON, FOREST, SAFFRON, BLUE, "#6b3a22"]
    rights = []
    for i, (head, body) in enumerate(nodes):
        x = 0.18 + i * 2.55
        ax.add_patch(FancyBboxPatch((x, 0.45), 2.15, 2.35, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor=fills[i % len(fills)], edgecolor="none"))
        ax.text(x + 1.07, 2.35, head, ha="center", va="center", color=CARD, fontsize=11, fontweight="bold")
        wrapped = "\n".join(textwrap.wrap(body, 24))
        ax.text(x + 1.07, 1.35, wrapped, ha="center", va="center", color=CARD, fontsize=9, linespacing=1.25)
        rights.append(x + 2.15)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((rights[i] + 0.02, 1.55), (rights[i] + 0.36, 1.55),
                                     arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.3))
    fig.suptitle(title, color=INK, fontsize=13)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def flow_bounds(name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 4.2), facecolor=PARCHMENT)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 0.35), 5.4, 3.3, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=FOREST, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((6.3, 0.35), 5.4, 3.3, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=MAROON, edgecolor="none"))
    ax.text(3.0, 3.2, "Enabled tonight", ha="center", color=CARD, fontsize=13, fontweight="bold")
    ax.text(9.0, 3.2, "Still ahead", ha="center", color=CARD, fontsize=13, fontweight="bold")
    left = "Flux map on the chip.\nJulia, JAX, and Triton agree.\nOne heavy job in 7.38 ms.\nFour heavy jobs in 7.39 ms.\nFive minutes, drift zero."
    right = "A silence gate that earns\nits keep. A calorimeter.\nA lithium-niobate core.\nA material quasicrystal.\nThese are not this filing's proof."
    ax.text(3.0, 1.7, left, ha="center", va="center", color=CARD, fontsize=11, linespacing=1.45)
    ax.text(9.0, 1.7, right, ha="center", va="center", color=CARD, fontsize=11, linespacing=1.45)
    fig.suptitle("What the record can carry, and what it leaves for later", color=INK, fontsize=13)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cover", fontName="Times-Bold", fontSize=20, leading=24, textColor=M, spaceAfter=4))
    base.add(ParagraphStyle("Sub", fontName="Times-Italic", fontSize=11, leading=14, textColor=F, spaceAfter=10))
    base.add(ParagraphStyle("H", fontName="Times-Bold", fontSize=13, leading=16, textColor=M, spaceBefore=12, spaceAfter=6))
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=11, leading=15, textColor=I, alignment=TA_JUSTIFY, spaceAfter=8))
    base.add(ParagraphStyle("Cap", fontName="Times-Italic", fontSize=10, leading=13, textColor=F, spaceBefore=4, spaceAfter=12))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8.5, leading=11, textColor=I))
    base.add(ParagraphStyle("Head", fontName="Times-Bold", fontSize=8.5, leading=11, textColor=colors.HexColor(CARD)))
    return base


def paint(canvas, doc, footer: str) -> None:
    canvas.saveState()
    canvas.setFillColor(P)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.setFillColor(M)
    canvas.rect(0, letter[1] - 0.34 * inch, letter[0], 0.34 * inch, fill=1, stroke=0)
    canvas.setFillColor(G)
    canvas.rect(0, letter[1] - 0.40 * inch, letter[0], 0.06 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(CARD))
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(0.72 * inch, letter[1] - 0.22 * inch, "Ceraph Corporation")
    canvas.setFillColor(F)
    canvas.setFont("Times-Italic", 8)
    canvas.drawString(0.72 * inch, 0.38 * inch, footer)
    canvas.drawRightString(letter[0] - 0.72 * inch, 0.38 * inch, str(doc.page))
    canvas.restoreState()


def pic(path: Path, max_w: float = 7.15 * inch, max_h: float = 5.6 * inch) -> Image:
    image = PILImage.open(path)
    width_px, height_px = image.size
    width = max_w
    height = width * (height_px / width_px)
    if height > max_h:
        height = max_h
        width = height * (width_px / height_px)
    return Image(str(path), width=width, height=height)


def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    s = styles()
    data = [[Paragraph(h, s["Head"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(cell), s["Cell"]) for cell in row])
    built = Table(data, colWidths=widths, repeatRows=1)
    built.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), M),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(CARD), colors.HexColor("#efe4d0")]),
        ("GRID", (0, 0), (-1, -1), 0.3, G),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return built


def figure_block(title: str, path: Path, caption: str) -> KeepTogether:
    s = styles()
    return KeepTogether([
        CondPageBreak(5.8 * inch),
        Paragraph(title, s["H"]),
        pic(path),
        Paragraph(caption, s["Cap"]),
    ])


def doc_template(path: Path, footer: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.68 * inch, rightMargin=0.68 * inch,
        topMargin=0.72 * inch, bottomMargin=0.62 * inch,
        title=path.stem, author="Ceraph Corporation",
    )


def how_story(figures: dict[str, Path], flows: dict[str, Path]) -> list:
    s = styles()
    wide = [7.15 * inch]
    return [
        Paragraph("How the technology works", s["Cover"]),
        Paragraph("For counsel. Ceraph Corporation. 14 November 2026. The pictures are redrawn from the evening telemetry so the words sit beside the bars, not on them.", s["Sub"]),
        Paragraph("A computer usually finishes a messy answer and only afterwards tries to clean it. This invention cleans while the answer is being made. It keeps the steady part and quiets the disturbance. It does not need a secret copy of the true signal in order to do that.", s["Body"]),
        Paragraph("The path", s["H"]),
        pic(flows["path"], max_h=3.4 * inch),
        Paragraph("Read left to right. A noisy series arrives. A short neighborhood yields a local center and a measure of how wild that neighborhood is. Only the departure from the center is shrunk. The series that leaves is steadier, and the center is still there.", s["Cap"]),
        Paragraph("Three languages, one formula", s["H"]),
        pic(flows["languages"], max_h=3.4 * inch),
        Paragraph("Julia reads the density of the disturbance and chooses a block width. JAX, on the graphics chip, checks that the map and its derivative match the formula. Triton runs that block. The host writes down the time, the watts, and the temperature. On this night all three agreed.", s["Cap"]),
        Paragraph("One bench, and four", s["H"]),
        pic(flows["benches"], max_h=3.4 * inch),
        Paragraph("A heavy job on one board took 7.38 milliseconds. Four such jobs, launched together on four boards, took 7.39 milliseconds. The same four jobs standing in a line on one board took 29.53 milliseconds. The wall did not grow when the boards did.", s["Cap"]),
        Paragraph("The formula, written large", s["H"]),
        pic(figures["flux"]),
        Paragraph("Mu is the local center. Xi is how wild the neighborhood is. Lambda was 1 in every test. The first term keeps the center. The second term shrinks only the departure from it. Counsel does not need the surrounding algebra to see that split.", s["Cap"]),
        Paragraph("Three laboratories", s["H"]),
        pic(flows["labs"], max_h=3.4 * inch),
        Paragraph("Colab held the first Blackwell timings and the dissertation notebooks. AWS Braket, on 11 November, checked an earlier Gaussian estimator against ordinary filters and touched a live simulator. DGX Cloud, on 14 November, is where Julia, JAX, and Triton ran the flux map for five minutes. The formula is one. The laboratories asked different questions.", s["Cap"]),
        figure_block(
            "The same kernel, from Colab to DGX",
            figures["platforms"],
            "Gold is the March Colab record: 0.0082 ms at 8,192 points and 0.0062 ms at 10,000. Forest is the DGX morning on GPU 0: about 0.011 ms from 8,192 through 1,048,576. Maroon is the evening Triton kernel at one million points, 0.0119 ms. All of them sit under the 0.2379 ms figure carried in the notebooks. That figure was a printed budget, not a timed competitor.",
        ),
        Paragraph("Layman explainer", s["H"]),
        Paragraph("Think of a song playing in a busy kitchen. The song is the steady thing. The clatter of plates is the disturbance. A moving average turns both down, so the song goes soft along with the noise. This method keeps the song at its own loudness and turns down only the clatter. That is why, on the test, the error stayed at 1.472 when the whole signal was shifted by five, while a method that multiplies the entire signal by a shrinking factor jumped to 18.17.", s["Body"]),
        Paragraph("Think of a photograph taken on a shaky day. The person in the frame should stay. The shake should go. The invention is that kind of steadying, done to a long row of numbers, inside the same pass that reads them.", s["Body"]),
        Paragraph("Think of a workshop. A tiny screw is not finished faster by hiring three more people, because picking the screw up takes longer than turning it. A heavy timber is. On these chips, a short job of one million points was the screw: four boards and one board finished in the same 0.065 milliseconds. A job of 16.8 million points, repeated forty times, was the timber: four boards finished four of them in the time one board finished one.", s["Body"]),
        Paragraph("A sentence counsel can use. The invention is an in-process way to keep the local center of a series and to damp only the fluctuation, scheduled by a density ladder, checked by a differentiable program, and executed as a fused graphics kernel. On 14 November 2026 it ran for five minutes on NVIDIA Blackwell boards without the result drifting.", s["Body"]),
        Paragraph("Comparison with current technology", s["H"]),
        Paragraph("The comparison is on the same kind of board, not a graphics chip against a laptop processor. Three ordinary tools are the yardstick: leaving the series alone, a moving average, and an eager loop written in ordinary PyTorch.", s["Body"]),
        table(
            ["Yardstick", "What it did", "What the evening showed"],
            [
                ["Untouched series", "Keep every sample", "Error 6.910"],
                ["Moving average", "Blur a short window", "Error 1.675. At one million points the pool took 0.0145 ms."],
                ["Raw damping", "Shrink the whole series", "Error 0.774 near zero, and 18.17 when the series sits at five. Useful only near zero."],
                ["Flux map, this invention", "Keep the center, shrink the wiggle", "Error 1.472 at both offsets. Triton matched it within about 5e-7."],
                ["Eager heavy loop, earlier the same evening", "Forty passes in ordinary PyTorch", "About 17.5 ms, near 195 W and 53 percent busy."],
                ["Triton heavy job", "The same forty passes as a fused kernel", "7.38 ms, drift 0, about 331 W and 91 percent busy on one board."],
                ["Four short jobs", "One million points each", "Queued 0.065 ms, together 0.065 ms. Too small for a second board to help."],
                ["Four heavy jobs", "16,777,216 points, forty passes", "Together 7.39 ms. Queued 29.53 ms. About four times the work in the same wait."],
            ],
            [1.45 * inch, 2.15 * inch, 3.55 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The moving-average pool is a few microseconds faster at short lengths, where both are waiting on the launch rather than on the arithmetic. At one million points the Triton kernel was the faster of the two, 0.0119 ms against 0.0145 ms. The quality win is the clearer one: less error than the average, and the same error when the signal is shifted. Speed without that quality would be a smaller claim. Quality with a heavy job that also got faster is the claim this record supports.", s["Body"]),
        figure_block(
            "The error, drawn large",
            figures["quality"],
            "Blue is the series sitting near zero. Gold is the same series shifted by five. Identity is the noisy series itself. Average is the moving average. Raw shrinks everything and grows tall once the signal leaves zero. Flux stays low in both rows: 1.472 against the average at 1.675.",
        ),
        figure_block(
            "Time across lengths, drawn large",
            figures["latency"],
            "Forest is the Triton kernel. Gold is a moving-average pool on the same board. From 256 points to 1,048,576 points the Triton time stays near 0.012 milliseconds. The slope on a log-log plot is about minus 0.0017. Every length is under the 0.2379 ms budget carried from the March notebooks. Julia chose block 256 at every length, because the density stayed between 0.69 and 0.81.",
        ),
        figure_block(
            "Four heavy jobs, drawn large",
            figures["jobs"],
            "Maroon is four heavy jobs one after another on a single board: 29.53 milliseconds. Forest is the same four jobs launched together, one per board: 7.39 milliseconds. A second timing the same evening agreed, 29.51 ms queued and 7.40 ms together. One board alone had needed 7.38 ms for one of those jobs.",
        ),
        figure_block(
            "Five minutes on one board",
            figures["soak_power"],
            "Each bar is a sample about twelve seconds apart, taken from a reading every two seconds, so the words on the axis have room. The run lasted 300.3 seconds and completed 40,377 jobs. Mean power was 307 watts. The peak was 369 watts. The board is rated at 600 watts. Drift between the opening result and the closing result was zero.",
        ),
        figure_block(
            "Five minutes on one board, the heat",
            figures["soak_temp"],
            "Temperature is drawn on its own scale so it is not hidden under the watts. The board began at 33 degrees and ended at 46. The clock stayed between 2377 and 2422 MHz. The chip was busy about 98 percent of the time. The opening sample, before the loop had registered, was the idle edge at 87 watts. The last sample was the sustained state: 306 watts, 46 degrees, 99 percent busy.",
        ),
        figure_block(
            "Five minutes on four boards",
            figures["four_temp"],
            "Each color is one board. The hottest any board became was 49 degrees. Mean power sat near 300 to 307 watts, with peaks from 321 to 333 watts. Every board stayed finite, drift stayed at zero, and every clock still reached 2422 MHz. The four boards completed 158,436 heavy jobs, about 3.92 times the one-board rate over the same five minutes.",
        ),
        Paragraph("What the record can carry", s["H"]),
        pic(flows["bounds"], max_h=3.6 * inch),
        Paragraph("Counsel can describe the flux map, the density ladder, the agreement of Julia, JAX, and Triton, and the measured behavior on one board and on four. A silence gate based on window variance reached an F1 of 0.291 in the earlier session, under a bar of 0.5, and was not retried tonight. A calorimeter, a lithium-niobate core, and a material quasicrystal were not in the room. They remain future embodiments.", s["Cap"]),
    ]


def tech_story(figures: dict[str, Path]) -> list:
    s = styles()
    one_v = load("one_validation.json")
    one_h = load("one_heavy.json")
    one_s = load("one_soak.json")
    four_h = load("four_heavy.json")
    four_s = load("four_soak.json")
    raw = one_v["raw_stack"]
    q0, q5 = one_v["quality"]
    return [
        Paragraph("Technical record and early investor note", s["Cover"]),
        Paragraph("Ceraph Corporation. 14 November 2026. For the company file, and for a note that can go out after the provisional draft is approved. Planning figures for a raise are estimates for a conversation. They are not a forecast of revenue.", s["Sub"]),
        Paragraph("Abstract", s["H"]),
        Paragraph("This note records a measured computing architecture, called here the Resonant Awareness Architecture, on the evening it first spent five uninterrupted minutes under load on NVIDIA Blackwell. The enabled invention is a flux form of Abhaya damping. A local center is preserved. Only the deviation from that center is multiplied by a shrinking exponential of the local instability. A density ladder, shared by a Julia macro named @sphota and by a Python twin, chooses the launch block: 128, 256, 512, or 1024. A JAX primitive on the graphics processor carries the same map and an analytical directional derivative. An OpenAI Triton kernel executes the map at the chosen block. A sign-bit exclusive-or negates a float32 value inside the same class of kernel.", s["Body"]),
        Paragraph(f"On seed 11, density {raw['density']:.3f} selected block {raw['block']} and grid {raw['grid']}. JAX matched the raw map and its derivative with error {raw['jax_max_abs_error']:.1f}. Triton stayed within {raw['triton_max_abs_error']:.2e}. Sign-bit negation error was {raw['sign_xor_error']:.1f}. The spiky quality series has density {q0['density']:.2f}, so the ladder correctly moved that kernel to block {q0['block']}. Flux mean squared error was {q0['flux_mse']:.3f} at offset 0 and {q5['flux_mse']:.3f} at offset 5, against a moving average of {q0['moving_average_mse']:.3f} and an untouched series of {q0['identity_mse']:.3f}. Raw damping was {q0['raw_mse']:.3f} near zero and {q5['raw_mse']:.2f} at offset 5. Flux retained {q0['flux_variance_remaining']:.1%} of the variance.", s["Body"]),
        Paragraph(f"A heavy job of {one_h['length']:,} float32 points and {one_h['passes']} flux passes took {one_h['before_ms']:.2f} ms before a burn and {one_h['after_ms']:.2f} ms after, with repeat drift {one_h['repeat_drift']:.1f} and a closed-form discrepancy of {one_h['closed_form_max_abs_error']:.2e}. Four such jobs together took {four_h['before_ms']:.2f} ms. The same four queued on one board took {four_h['queued_on_one_board_ms']:.2f} ms. A five-minute soak on one board completed {one_s['loops']:,} jobs in {one_s['elapsed_s']:.1f} s at drift {one_s['repeat_drift']:.1f}, mean power {one_s['mean_power_w']:.0f} W, peak temperature {one_s['max_c']:.0f} C, and a clock still inside {one_s['min_clock_mhz']:.0f} to {one_s['max_clock_mhz']:.0f} MHz. Four boards completed {four_s['loops'] * 4:,} jobs in {four_s['elapsed_s']:.1f} s, drift zero on every board, none warmer than 49 C. The boards are RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W limit. Software was Python 3.12, PyTorch 2.14.1 with CUDA 13, Triton 3.8.0, JAX 0.12 with the CUDA 13 plugin, and Julia 1.13.4.", s["Body"]),
        Paragraph("Mathematical framework", s["H"]),
        Paragraph("The continuous picture used in the notebooks is an omega-dit. An existence density, a resonant phase, and a sum of manifest value plus unmodeled flux are multiplied and summed. The evening session did not need the full integral in order to run the damping. It needed the split between what is steady and what is flux. The picture below is the discrete stand-in.", s["Body"]),
        pic(figures["omega"]),
        Paragraph("Sigma is existence density, beta is the resonant weight, v is the manifest value, and xi is the flux. Delta omega is the step of the sum. This is the notebook's continuous-variable picture brought down to a finite series.", s["Cap"]),
        Paragraph("Raw Abhaya damping multiplies the whole series by a shrinking exponential. Lambda was 1 in every test below. The map is right when the carrier itself sits near zero. When a large offset is part of the signal, the same exponential shrinks the signal along with the noise. That is why raw error rose from 0.774 to 18.17 when the latent series was shifted by five.", s["Body"]),
        pic(figures["raw"]),
        Paragraph("Theta is the series. Xi is the instability. The exponential can be smaller than one or, if xi is negative, larger than one. Nothing in this formula protects a large steady offset.", s["Cap"]),
        Paragraph("Flux damping is the deployable form. Mu is a local center, estimated by a window mean. Xi is a local instability, estimated by a window variance and clamped at zero. The center is an additive fixed point: if the series already equals the center, the map returns the center.", s["Body"]),
        pic(figures["flux"]),
        Paragraph("Read the right-hand side in two beats. Mu is kept. The parenthesis is the wiggle, and only the wiggle is multiplied by the exponential.", s["Cap"]),
        Paragraph("Forty iterations with a fixed center and a fixed xi collapse to one exponential. On the heavy job the Triton iteration matched that closed form within 4.2e-7 on one board and within 7.9e-7 on the four-board run.", s["Body"]),
        pic(figures["closed"]),
        Paragraph("The 40 is the pass count used in the heavy job and in both five-minute soaks. A drift of zero means a second run of those forty passes matched the first.", s["Cap"]),
        Paragraph("The directional derivative of the raw map is what a training step would need. JAX binds it as a custom Jacobian-vector product. On the seed-11 draw the derivative check returned 0 on every board, so the primitive and the hand derivative agreed to the precision of the comparison.", s["Body"]),
        pic(figures["jvp"]),
        Paragraph("d theta and d xi are the directions of the probe. The first term carries the state derivative. The second term carries the instability derivative, with a minus from the exponential.", s["Cap"]),
        Paragraph("The block ladder reads the mean absolute instability. Below 0.5 the block is 128. Below 1 it is 256. Below 2 it is 512. Otherwise it is 1024. The grid is the length divided by the block, rounded up. Julia exposes the choice as the macro @sphota. The Python function sphota_schedule is the same ladder. On ordinary normal noise the density was about 0.80 and both chose 256. On the spiky quality series the density was 7.06 and both chose 1024.", s["Body"]),
        pic(figures["sphota"]),
        Paragraph("These four widths are the only legal blocks. The ladder picks one. It does not claim that 256 is best for every tensor. On this night, for ordinary noise, 256 was also the fastest of the four.", s["Cap"]),
        Paragraph("Sign-bit negation flips the float32 sign by an unsigned exclusive-or with 0x80000000, formed as an unsigned full vector so the constant is not parsed as a signed 32-bit immediate. The residual against arithmetic negation was 0. This is the software image of a paired-state cancellation: half of (v + xi) plus (v - xi) returns v.", s["Body"]),
        pic(figures["sign"]),
        Paragraph("The constant is the IEEE sign bit of a float32, written so the compiler keeps it unsigned. The measured residual was 0 on every board.", s["Cap"]),
        Paragraph("Kinship with modern science", s["H"]),
        Paragraph("The exponential shrink is kin to classical damping and to an exponential moving weight, with one structural difference: the weight multiplies the deviation, not the whole observation. That difference is what keeps a shifted signal intact. A moving average is the linear cousin. It reduces variance by mixing neighbors, and it pays for that mix with bias. The flux map can leave less squared error than that mix, which is what the seed-11 series showed, without being given the latent series as an input.", s["Body"]),
        Paragraph("The JAX primitive is ordinary automatic differentiation, pointed at a map simple enough to write by hand. The value of the primitive is that a later training loop can sit on the same formula the kernel runs. The Triton kernel is a fused elementwise program: load a tile, apply the exponential, store the tile. Its time from a few hundred points to a million points is nearly flat because the launch, not the arithmetic, dominates. That is a known fact of short graphics kernels. It becomes useful here because the scheduled block stays inside that flat region, and because the heavy iterated job, where the arithmetic does show, still finished in 7.38 ms.", s["Body"]),
        Paragraph("A current parallel is the split, now ordinary in graphics software, between a fused kernel and a chain of eager operations. Triton, CUDA tiles, and the compiler stacks around them exist so a short map can stay in one launch. The evening heavy job is that idea measured: the eager forty-pass loop took about 17.5 ms, and the fused kernel took 7.38 ms, on the same class of board.", s["Body"]),
        Paragraph("Another parallel is dual-mode estimation against a single-mode Wiener filter. The AWS campaign on 11 November measured that cousin: a two-mode Gaussian estimator at mean squared error 0.614, a single-mode Wiener filter at 0.694, a moving average at 1.088, and a median filter at 1.235. The flux map is not that Gaussian estimator. Both lines of work share the habit of using a second view of the disturbance instead of blurring the observation.", s["Body"]),
        Paragraph("A third parallel is how a machine room should be used. Sharding one small vector across four boards made the 14 November morning slower, 0.088 ms against 0.047 ms on one board, because the launch and the join cost more than the arithmetic. Independent heavy jobs are the other pattern. Four of them finished together in 7.39 ms. Current multi-GPU practice already knows this distinction. The soak is the measurement for this map.", s["Body"]),
        Paragraph("Landauer's principle, a physical calorimeter, a periodically poled lithium-niobate optical core, and a material quasicrystal with many-body localization are related ideas in the longer notebooks. They are not results of this session. Naming them keeps the kinship visible. It does not move them into the enabled claim.", s["Body"]),
        Paragraph("Colab, AWS, and DGX", s["H"]),
        Paragraph("The testing phase closes across three laboratories. Google Colab held the dissertation notebooks and the first saved Blackwell kernel times. AWS Braket held a classical accuracy campaign and a live simulator contact. DGX Cloud held the four-board Blackwell machines, first for the morning claim card and then, the same day, for the heavy jobs and the five-minute soaks. Each laboratory is allowed to answer only the question it was given.", s["Body"]),
        table(
            ["Laboratory", "When", "What was actually run", "What it is allowed to prove"],
            [
                ["Google Colab", "Notebooks through March, saved outputs", "Triton medians 0.0082 ms at 8,192 and 0.0062 ms at 10,000. A JAX run at 16,384 median 0.0888 ms.", "The kernel was already in the sub-millisecond band on Blackwell. The 0.2379 ms line is a printed budget, not a timed opponent."],
                ["AWS Braket", "11 November 2026", "Campaign S1 through S5 passed. Live SV1 task completed. A Hybrid Job on ml.m5.large returned training success. SV1 cost 0.00375 dollars.", "A Gaussian estimator beat a moving average, a median, and single-mode Wiener on that campaign. It is not the flux kernel, and it is not a quantum-processor proof."],
                ["L40S, earlier GPU host", "Before the Blackwell evening", "A synthetic map at one million elements finished about 41 times sooner than the CPU beside it, near 85 W and 36 C, on a 350 W board.", "A same-host speed record for that synthetic map. Not a same-GPU contest against a named filter."],
                ["DGX, morning", "14 November, four RTX PRO 6000 boards", "Julia, JAX, and Triton agreed. Flux MSE 1.472. Triton near 0.010 to 0.012 ms. Silence-gate F1 0.291. A four-way shard was slower than one board.", "The stack and the flux quality. Also the misses: the gate, the shard, and raw damping at offset 5."],
                ["DGX, evening", "14 November, one board and four", "Heavy Triton job 7.38 ms. Four jobs together 7.39 ms, queued 29.53 ms. Five minutes, drift 0, peaks 46 C and 49 C.", "Endurance and the enterprise pattern: heavy independent jobs scale, short jobs do not."],
            ],
            [1.15 * inch, 1.25 * inch, 2.35 * inch, 2.4 * inch],
        ),
        Spacer(1, 8),
        figure_block(
            "Colab and DGX, one time axis",
            figures["platforms"],
            "Every bar is a Blackwell kernel median in milliseconds. Gold is Colab. Forest is the DGX morning on GPU 0. Maroon is the evening Triton kernel at one million points. The bars are the same height for practical purposes. The later machines did not leave the band the notebooks had already entered.",
        ),
        figure_block(
            "AWS, the filter contest that campaign could run",
            figures["braket"],
            "Lower is better. Dual-mode Gaussian MMSE at 0.614, single-mode Wiener at 0.694, moving average at 1.088, median at 1.235. These are campaign numbers from 11 November, on classical arrays plus a live SV1 smoke and a Hybrid Job. They are not a timing of the Triton flux kernel, and they are not a claim about a quantum processing unit.",
        ),
        Paragraph("Sphota, the hybrid Om-lang", s["H"]),
        Paragraph("Om-lang is the project's name for one formula living in three tiers. The control tier is Julia. The differentiable tier is JAX on CUDA. The kernel tier is Triton, launched from PyTorch tensors. Python and NumPy hold the reference. A run is accepted when the Julia block matches the Python ladder, the JAX values and the derivative match the formula, and Triton matches the same formula on the device that will do the work.", s["Body"]),
        Paragraph("On the one-board machine JAX reported cuda:0. On the four-board machine it reported cuda:0 through cuda:3. The stack was certified on every visible device before any soak began. Triton flux error against the NumPy flux values was about 2e-7 to 5e-7. JAX flux error on an 8,192-point slice of the heavy tensors was 2.4e-7.", s["Body"]),
        Paragraph("Proven execution", s["H"]),
        Paragraph("Hardware. NVIDIA RTX PRO 6000 Blackwell Server Edition. One machine with one board. One machine with four. Each board reports 97,887 MiB and a 600 W limit. The numbers below are taken from the telemetry files saved with the histograms.", s["Body"]),
        table(
            ["Check", "One board", "Four boards"],
            [
                ["Density, seed 11", f"{raw['density']:.3f}", "0.793 on each board"],
                ["Block and grid", f"{raw['block']} / {raw['grid']}", "256 / 64 on each board"],
                ["JAX value and derivative", "0 and 0", "0 and 0 on each board"],
                ["Triton versus formula", f"{raw['triton_max_abs_error']:.2e}", "7.63e-6 on each board"],
                ["Sign-bit negation", "0", "0 on each board"],
                ["Flux MSE, both offsets", f"{q0['flux_mse']:.3f}", "same series, same 1.472"],
                ["Moving average MSE", f"{q0['moving_average_mse']:.3f}", "same"],
                ["Raw MSE at offset 5", f"{q5['raw_mse']:.2f}", "same, and not the deployable form"],
                ["Heavy job", f"{one_h['before_ms']:.2f} ms, drift {one_h['repeat_drift']:.0f}", f"{four_h['before_ms']:.2f} ms together, {four_h['queued_on_one_board_ms']:.2f} ms queued"],
                ["Five-minute jobs", f"{one_s['loops']:,} in {one_s['elapsed_s']:.1f} s", f"{four_s['loops'] * 4:,} in {four_s['elapsed_s']:.1f} s"],
                ["Five-minute power", f"mean {one_s['mean_power_w']:.0f} W, peak {one_s['max_power_w']:.0f} W", "mean about 300 to 307 W, peaks 321 to 333 W"],
                ["Five-minute heat", f"{one_s['start_c']:.0f} C to {one_s['max_c']:.0f} C", "start 34 to 35 C, peak 48 to 49 C"],
                ["Clock", f"{one_s['min_clock_mhz']:.0f} to {one_s['max_clock_mhz']:.0f} MHz", "mins 2280 to 2347 MHz, every max 2422 MHz"],
            ],
            [1.7 * inch, 2.7 * inch, 2.75 * inch],
        ),
        Spacer(1, 8),
        Paragraph("A second four-board heavy timing, taken earlier the same evening while the boards were already warm, agreed within a few hundredths of a millisecond: 7.40 ms together and 29.51 ms queued. The file behind the drawings is the cooler-start run, 7.395 ms and 29.525 ms. Both are the same result.", s["Body"]),
        Paragraph("At short lengths the Triton kernel and the pool are both near a hundredth of a millisecond. The pool is slightly ahead until the series reaches a million points, where Triton is ahead. The log-log slope of Triton time against length is -0.0017. Flatness here is a launch-bound kernel, which is a good fact for a map this small, and it is not a claim that every future program will be flat.", s["Body"]),
        figure_block("Error histogram", figures["quality"], "Lower bars are better. Flux is the short bar in both rows. Raw is short only in the front row and very tall in the back row. That picture is the reason flux, and not raw, is the map we take forward."),
        figure_block("Latency histogram", figures["latency"], "The forest row is nearly level. That is the sub-millisecond, near-flat launch of the raw kernel. The gold row is the moving-average pool. Numbers are in the table above. The drawing is here so a reader can see the level, not so the reader has to squint a label out of a bar."),
        figure_block("Enterprise histogram", figures["jobs"], "The tall bar is the queue. The short bar is four boards working at once. Same jobs, same formula, same block size. The wait shrinks by about a factor of four."),
        figure_block("One-board endurance, power", figures["soak_power"], "Five minutes. The first thin region is the loop getting hold of the chip. After that the draw sits near 300 watts, with a peak of 369, against a 600 watt limit."),
        figure_block("One-board endurance, temperature", figures["soak_temp"], "Its own scale, so 46 degrees is visible. The rise is about thirteen degrees across five minutes at roughly 98 percent busy. The clock did not fall out of the boost band."),
        figure_block("One-board endurance, occupancy", figures["soak_util"], "Percent of the chip reported busy. After the opening sample the bars sit near the top of the scale. Mean occupancy was 97.7 percent."),
        figure_block("Four-board endurance, power", figures["four_power"], "Four colors, four boards, one soak. The boards stay in a band together, near 300 watts, rather than one board running away from the others."),
        figure_block("Four-board endurance, temperature", figures["four_temp"], "Peaks of 48 and 49 degrees. No board left the boost clocks. Drift on each board was zero."),
        figure_block("The short heavy burn on four boards", figures["heavy_power"], "About twenty seconds, sampled every two seconds, while four heavy jobs ran together. This is the burn between the before and after timings. Mean power on the four boards was 299, 297, 298, and 299 watts."),
        Paragraph("Why the work can be called a beginning", s["H"]),
        Paragraph("A beginning is not a finished market. It is the moment a long idea survives contact with the machine it was written for. This one did. Three implementations of one formula agreed. The form that keeps the center beat a moving average on the series that had embarrassed the raw form. A heavy job got faster when it was fused, and four heavy jobs occupied four boards without stretching the wait. Then the same job was left alone for five minutes, forty thousand times on one board and a hundred and fifty-eight thousand times across four, and the bits that came out matched the bits that went in.", s["Body"]),
        Paragraph("The potential, said plainly, is an in-process cleaning step that a later product can drop into a pipeline which already lives on graphics chips: a sensor stream, a studio signal, a scientific series, or the inner state of a model. The step is small enough to schedule, differentiable enough to train, and fused enough to repeat. The five-minute record says the step does not wander while it repeats. That is the ground a company can stand on while it chooses one wedge and teaches that wedge to earn a fee.", s["Body"]),
        Paragraph("What the raise is for, and what it is not for, should stay visible. The raise continues the enabled work: a cleaner silence decision than window variance, a small library a design partner can call, a longer measurement program, and the people to do those things. It does not buy a claim that a calorimeter has been built, or that a new physical law has been shown on this night.", s["Body"]),
        Paragraph("Initial funding", s["H"]),
        Paragraph("Two bands are enough for a first conversation after the provisional draft is approved. Both are planning estimates for eighteen or twelve months of continuation. Salaries are loaded roughly, cloud is an average rather than a peak, and counsel covers the path from this provisional toward a utility filing. Geography and investor type can move either band by about a quarter.", s["Body"]),
        Paragraph("Working ask: 1.25 million US dollars for 18 months. This is the band that hires, keeps a research year, and still has a reserve.", s["Body"]),
        table(
            ["Use", "Basis", "Amount"],
            [
                ["Founder", "18 months at 10,000 a month", "$180,000"],
                ["Kernel engineer", "18 months, loaded near 16,000 a month", "$288,000"],
                ["Research scientist", "12 months, loaded near 14,000 a month", "$168,000"],
                ["Finance and operations", "Fractional, 2,000 a month", "$36,000"],
                ["Counsel and utility path", "Beyond the provisional already in motion", "$120,000"],
                ["Entity, insurance, accounting", "18 months", "$40,000"],
                ["Blackwell and other cloud", "Planned average 8,000 a month", "$144,000"],
                ["Bench held for later embodiments", "Not spent as if the devices already exist", "$30,000"],
                ["Papers, site, two gatherings, travel", "A small public face", "$70,000"],
                ["Design-partner program", "Our cost to show up. Their fee is income, below.", "$40,000"],
                ["Recruiting", "Two hires", "$20,000"],
                ["Reserve", "About 10 percent", "$114,000"],
                ["Working ask", "18 months", "$1,250,000"],
            ],
            [2.3 * inch, 3.15 * inch, 1.7 * inch],
        ),
        Spacer(1, 8),
        Paragraph("Lean ask: 700,000 US dollars for 12 months, if the round is smaller and the scientist starts later. Founder 120,000. Engineer for 12 months 192,000. Scientist for 6 months 84,000. Counsel 80,000. Cloud at 6,000 a month 72,000. The small public face and partner program 50,000. Operations 36,000. Reserve 66,000.", s["Body"]),
        Paragraph("The lean ask keeps the measurement program and one engineer. The working ask is the one that matches the phrase successful continuation: a second researcher, a full utility path, and enough cloud to repeat what tonight did on new series rather than on a single seed.", s["Body"]),
        Paragraph("Projected income", s["H"]),
        Paragraph("Income is projected from one assumption, and the assumption should be said before the numbers. A design partner who already runs a graphics pipeline pays on the order of 60,000 US dollars a year to put this map inside one of their streams, once a small library exists and the provisional is on file. Pilots in the first year can be smaller. None of this income exists today. The table is a planning band for investors, not a booking.", s["Body"]),
        table(
            ["Year after the raise", "Quiet", "Working", "One wedge sticks"],
            [
                ["Year 1", "No paid partner. The year is the library and the filing.", "Two paid pilots, about $120,000.", "Same as the working column. A wedge is not a product yet."],
                ["Year 2", "One retainer, about $60,000.", "Four retainers, about $400,000.", "Six to eight, about $0.8 million to $1.2 million."],
                ["Year 3", "Two retainers, about $150,000.", "Eight retainers, about $1.2 million.", "A product in one vertical, planned band $2.5 million to $4 million."],
            ],
            [1.35 * inch, 1.9 * inch, 1.9 * inch, 2.0 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The quiet column is a company that keeps the science alive and finds one friend. The working column is a company that sells a careful service to teams who already have the hardware. The right-hand column is what happens if one of those teams becomes a market, for example studio signals or a sensor line, and the license becomes a product. A consumer application, a multi-tenant cloud, and a claim of thermodynamic advantage are outside all three columns.", s["Body"]),
        Paragraph("Conclusion", s["H"]),
        Paragraph("On 14 November 2026 the hybrid language was real on Blackwell, the flux map held its error when the raw map did not, one heavy job held 7.38 milliseconds, four heavy jobs held that same wait, and five minutes of repetition left a drift of zero. The heat stayed under 50 degrees on a 600 watt board. The clock stayed in boost. That is the record. The raise described above buys the people and the months to turn the record into one paid wedge, and to file the utility that follows this provisional. It does not buy a finished market, and it does not need to.", s["Body"]),
        Paragraph("Absolute remains untouched, only the error is taken away.", s["Cover"]),
    ]


def shade_cell(cell, hex_color: str) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    shd = tc_pr.makeelement(qn("w:shd"), {qn("w:fill"): hex_color, qn("w:val"): "clear"})
    tc_pr.append(shd)


def add_docx(path: Path, title: str, blocks: list) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0x3A, 0x24, 0x16)
    for kind, payload in blocks:
        if kind == "h":
            paragraph = document.add_heading(payload, level=1)
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(0x7C, 0x2D, 0x2D)
                run.font.name = "Times New Roman"
        elif kind == "p":
            paragraph = document.add_paragraph(payload)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            paragraph.paragraph_format.space_after = Pt(8)
        elif kind == "img":
            document.add_picture(str(payload), width=Inches(6.5))
        elif kind == "table":
            grid = document.add_table(rows=len(payload), cols=len(payload[0]))
            grid.style = "Table Grid"
            for r, row in enumerate(payload):
                for c, text in enumerate(row):
                    grid.rows[r].cells[c].text = text
                    for paragraph in grid.rows[r].cells[c].paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(9)
                            run.font.name = "Times New Roman"
                            if r == 0:
                                run.bold = True
                                run.font.color.rgb = RGBColor(0xFB, 0xF6, 0xEC)
                    if r == 0:
                        shade_cell(grid.rows[r].cells[c], "7C2D2D")
                    elif r % 2 == 0:
                        shade_cell(grid.rows[r].cells[c], "EFE4D0")
            document.add_paragraph("")
        elif kind == "cap":
            paragraph = document.add_paragraph(payload)
            for run in paragraph.runs:
                run.italic = True
                run.font.color.rgb = RGBColor(0x2C, 0x4A, 0x3E)
                run.font.size = Pt(10)
    document.save(path)


def docx_from_story(path: Path, story_title: str, pdf_items: list) -> None:
    blocks: list[tuple[str, object]] = []
    for item in pdf_items:
        if isinstance(item, Paragraph):
            text = item.getPlainText()
            style = item.style.name if item.style is not None else ""
            if style in {"Cover", "H"}:
                blocks.append(("h", text))
            elif style == "Cap":
                blocks.append(("cap", text))
            elif style == "Sub":
                blocks.append(("cap", text))
            else:
                blocks.append(("p", text))
        elif isinstance(item, Image):
            blocks.append(("img", item.filename))
        elif isinstance(item, Table):
            rows = []
            for row in item._cellvalues:
                rows.append([cell.getPlainText() if hasattr(cell, "getPlainText") else str(cell) for cell in row])
            blocks.append(("table", rows))
        elif isinstance(item, KeepTogether):
            for inner in item._content:
                if isinstance(inner, Paragraph):
                    style = inner.style.name if inner.style is not None else ""
                    text = inner.getPlainText()
                    blocks.append(("h" if style == "H" else "cap" if style == "Cap" else "p", text))
                elif isinstance(inner, Image):
                    blocks.append(("img", inner.filename))
    add_docx(path, story_title, blocks)


def build() -> None:
    figures = draw_figures()
    flows = {
        "path": flow_chain("flow_path.png", "Keep the center, quiet the wiggle", [
            ("Series", "A noisy row of numbers arrives."),
            ("Neighborhood", "A short window gives a center and a wildness."),
            ("Flux map", "Only the departure from the center is shrunk."),
            ("Steadier row", "The center remains. The clatter is quieter."),
        ]),
        "languages": flow_chain("flow_languages.png", "Julia, JAX, and Triton", [
            ("Density", "Mean absolute instability of the series."),
            ("Julia", "@sphota chooses 128, 256, 512, or 1024."),
            ("JAX", "The map and its derivative, on the graphics chip."),
            ("Triton", "The fused kernel runs at that block."),
        ]),
        "benches": flow_chain("flow_benches.png", "One heavy job, and four", [
            ("One board", "One heavy job, 7.38 milliseconds."),
            ("Four boards", "Four heavy jobs together, 7.39 milliseconds."),
            ("One queue", "Those four jobs in a line, 29.53 milliseconds."),
        ]),
        "bounds": flow_bounds("flow_bounds.png"),
        "labs": flow_chain("flow_labs.png", "Colab, AWS, and DGX", [
            ("Colab", "First Blackwell timings and the dissertation notebooks."),
            ("AWS", "A filter contest, plus a live simulator contact."),
            ("DGX", "Julia, JAX, and Triton, then five minutes under load."),
        ]),
    }
    figures.update(draw_equations())
    how_items = how_story(figures, flows)
    tech_items = tech_story(figures)
    how_pdf = OUT / "nov14-how-it-works.pdf"
    tech_pdf = OUT / "nov14-technical-record.pdf"
    how_doc = doc_template(how_pdf, "How it works  ·  for counsel  ·  14 November 2026")
    tech_doc = doc_template(tech_pdf, "Technical record  ·  Ceraph Corporation  ·  14 November 2026")
    how_footer = "How it works  ·  for counsel  ·  14 November 2026"
    how_doc.build(how_items, onFirstPage=lambda c, d: paint(c, d, how_footer), onLaterPages=lambda c, d: paint(c, d, how_footer))
    tech_doc.build(tech_items, onFirstPage=lambda c, d: paint(c, d, "Technical record  ·  14 November 2026"), onLaterPages=lambda c, d: paint(c, d, "Technical record  ·  14 November 2026"))
    docx_from_story(OUT / "nov14-how-it-works.docx", "How it works", how_items)
    docx_from_story(OUT / "nov14-technical-record.docx", "Technical record", tech_items)
    print({"pdf": [str(how_pdf), str(tech_pdf)], "figures": len(figures) + len(flows)})


if __name__ == "__main__":
    build()
