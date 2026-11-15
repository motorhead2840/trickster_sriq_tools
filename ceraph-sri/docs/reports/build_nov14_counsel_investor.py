"""Counsel brief and investor record for the 14 November dual-suite night.

The page content is white so a printed copy does not flood the sheet with
parchment. The maroon header band, the gold rule, and the footer ink stay.
Bar colors, words, and numbers are unchanged.

Numbers come from the one-board notebook, the four-board notebook, and the
telemetry archives saved from those notebooks. Histograms are redrawn from
those files. A caption sits under each picture.
"""
from __future__ import annotations

import json
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
DATA = ROOT / "benchmark-results" / "rtx6000-dual-suite"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures" / "nov14-dual-suite"

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

FOOTER_COUNSEL = "For counsel  ·  Ceraph Corporation  ·  Kerala, India  ·  chief@ceraphco.com  ·  www.ceraphco.com"
FOOTER_INVESTOR = "Investor record  ·  Ceraph Corporation  ·  Kerala, India  ·  chief@ceraphco.com  ·  www.ceraphco.com"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def style_3d(ax) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors=INK, labelsize=9, pad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(INK)
        axis.label.set_fontsize(11)
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#d9d9d9")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = "#d9d9d9"
        axis._axinfo["grid"]["linewidth"] = 0.6


def save_fig(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.42)
    plt.close(fig)
    return path


def legend_above(fig, handles) -> None:
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
        ncol=min(4, len(handles)), frameon=False, fontsize=11, labelcolor=INK,
    )


def bar3(ax, xs, y, heights, color, depth=0.45, width=0.55) -> None:
    ax.bar3d(xs, np.full(len(xs), y), np.zeros(len(xs)), width, depth, heights, color=color, shade=True, edgecolor="#222222", linewidth=0.25)


def pick(samples: list, limit: int = 18) -> list:
    if len(samples) <= limit:
        return samples
    idx = np.linspace(0, len(samples) - 1, limit).astype(int)
    return [samples[i] for i in idx]


def draw_figures() -> dict[str, Path]:
    one = load("one_soak_10min.json")
    four = load("four_full_suite.json")
    paths: dict[str, Path] = {}
    palette = [BLUE, FOREST, GOLD, MAROON]

    fig = plt.figure(figsize=(11.2, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    labels = ["untouched", "average", "raw", "flux"]
    rows = [
        [6.910, 1.675, 0.774, 1.472],
        [6.910, 1.675, 18.171, 1.472],
    ]
    for row, (vals, color) in enumerate(zip(rows, [BLUE, SAFFRON])):
        bar3(ax, np.arange(4), row, vals, color, depth=0.42, width=0.52)
    ax.set_xticks(np.arange(4) + 0.2)
    ax.set_xticklabels(labels, color=INK)
    ax.set_yticks([0.2, 1.2])
    ax.set_yticklabels(["offset 0", "offset 5"], color=INK)
    ax.set_zlabel("mean squared error", labelpad=12)
    ax.view_init(elev=24, azim=-58)
    legend_above(fig, [Patch(facecolor=BLUE, label="Offset 0"), Patch(facecolor=SAFFRON, label="Offset 5")])
    fig.suptitle("Flux keeps its error when the signal is shifted", color=INK, fontsize=15, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.96, top=0.78, bottom=0.06)
    paths["quality"] = save_fig(fig, "quality_mse_3d.png")

    samples = pick(one["samples"], 18)
    xs = np.arange(len(samples))
    ticks = np.linspace(0, len(samples) - 1, 6).astype(int)

    def one_series(key: str, title: str, zlabel: str, color: str, filename: str) -> Path:
        fig = plt.figure(figsize=(11.2, 7.2), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        style_3d(ax)
        bar3(ax, xs, 0, [s[key] for s in samples], color, depth=0.38, width=0.55)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{samples[i]['elapsed_min']:.1f}" for i in ticks], rotation=18, ha="right")
        ax.set_xlabel("minutes", labelpad=14)
        ax.set_yticks([])
        ax.set_zlabel(zlabel, labelpad=12)
        ax.view_init(elev=25, azim=-52)
        fig.suptitle(title, color=INK, fontsize=15)
        fig.subplots_adjust(left=0.04, right=0.92, top=0.88, bottom=0.08)
        return save_fig(fig, filename)

    paths["one_power"] = one_series("power_w", "One board  ·  ten minutes of power", "watts", MAROON, "one_power_3d.png")
    paths["one_temp"] = one_series("temp_c", "One board  ·  ten minutes of temperature", "degrees C", GOLD, "one_temp_3d.png")
    paths["one_clock"] = one_series("clock_mhz", "One board  ·  ten minutes of SM clock", "megahertz", FOREST, "one_clock_3d.png")

    soak = pick(four["samples"], 18)
    xs4 = np.arange(len(soak))
    ticks4 = np.linspace(0, len(soak) - 1, 6).astype(int)

    def four_series(metric: str, title: str, zlabel: str, filename: str) -> Path:
        fig = plt.figure(figsize=(11.4, 7.6), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        style_3d(ax)
        handles = []
        for i in range(4):
            vals = [s["boards"][i][metric] for s in soak]
            bar3(ax, xs4, i, vals, palette[i], depth=0.42, width=0.5)
            handles.append(Patch(facecolor=palette[i], label=f"Board {i}"))
        ax.set_xticks(ticks4)
        ax.set_xticklabels([f"{soak[i]['elapsed_min']:.1f}" for i in ticks4], rotation=18, ha="right")
        ax.set_xlabel("minutes", labelpad=16)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["0", "1", "2", "3"], color=INK)
        ax.set_ylabel("board", labelpad=14)
        ax.set_zlabel(zlabel, labelpad=12)
        ax.view_init(elev=24, azim=-58)
        legend_above(fig, handles)
        fig.suptitle(title, color=INK, fontsize=15, y=1.0)
        fig.subplots_adjust(left=0.02, right=0.94, top=0.78, bottom=0.08)
        return save_fig(fig, filename)

    paths["four_power"] = four_series("power_w", "Four boards  ·  ten minutes of power", "watts", "four_power_3d.png")
    paths["four_temp"] = four_series("temp_c", "Four boards  ·  ten minutes of temperature", "degrees C", "four_temp_3d.png")

    fig = plt.figure(figsize=(11.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    rates = [(0, 136.1, BLUE), (1, 137.0, FOREST), (2, 542.9, GOLD), (3, 548.8, MAROON)]
    names = ["Gemini\none board", "Independent\none board", "Gemini\nfour boards", "Independent\nfour boards"]
    for x, height, color in rates:
        ax.bar3d([x], [0], [0], 0.62, 0.45, [height], color=color, shade=True, edgecolor="#222222", linewidth=0.3)
    ax.set_xticks(np.arange(4) + 0.3)
    ax.set_xticklabels(names, color=INK)
    ax.set_yticks([])
    ax.set_zlabel("heavy jobs per second", labelpad=12)
    ax.view_init(elev=22, azim=-62)
    fig.suptitle("Ten-minute job rate  ·  two suites", color=INK, fontsize=15)
    fig.subplots_adjust(left=0.04, right=0.94, top=0.88, bottom=0.08)
    paths["rates"] = save_fig(fig, "job_rates_3d.png")

    fig = plt.figure(figsize=(11.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    roof = [
        ("1.0M", 998, 1285),
        ("10.5M", 1017, 1264),
        ("50M\nGemini", 1467, 0),
        ("50M\nindependent", 0, 1465),
    ]
    for i, (label, gemini, indie) in enumerate(roof):
        if gemini:
            bar3(ax, [i], 0, [gemini], GOLD, depth=0.35, width=0.42)
        if indie:
            bar3(ax, [i], 1, [indie], FOREST, depth=0.35, width=0.42)
    ax.set_xticks(np.arange(4) + 0.2)
    ax.set_xticklabels([r[0] for r in roof], color=INK)
    ax.set_yticks([0.15, 1.15])
    ax.set_yticklabels(["Gemini", "independent"], color=INK)
    ax.set_zlabel("GB/s", labelpad=12)
    ax.view_init(elev=24, azim=-58)
    legend_above(fig, [Patch(facecolor=GOLD, label="Gemini suite"), Patch(facecolor=FOREST, label="Independent suite, 16 bytes per node")])
    fig.suptitle("Memory traffic  ·  toward the 1,597 GB/s peak", color=INK, fontsize=15, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.94, top=0.78, bottom=0.08)
    paths["roof"] = save_fig(fig, "roofline_3d.png")

    fig = plt.figure(figsize=(11.2, 7.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    uses = [
        ("People", 672),
        ("Counsel", 160),
        ("Cloud", 144),
        ("Partners", 110),
        ("Bench", 50),
        ("Reserve", 114),
    ]
    use_colors = [MAROON, FOREST, BLUE, SAFFRON, GOLD, "#6b3a22"]
    for i, ((label, amount), color) in enumerate(zip(uses, use_colors)):
        ax.bar3d([i], [0], [0], 0.62, 0.45, [amount], color=color, shade=True, edgecolor="#222222", linewidth=0.25)
    ax.set_xticks(np.arange(len(uses)) + 0.3)
    ax.set_xticklabels([u[0] for u in uses], color=INK)
    ax.set_yticks([])
    ax.set_zlabel("thousand USD", labelpad=12)
    ax.view_init(elev=22, azim=-62)
    fig.suptitle("Working ask  ·  1.25 million USD over 18 months", color=INK, fontsize=15)
    fig.subplots_adjust(left=0.04, right=0.94, top=0.88, bottom=0.06)
    paths["funds"] = save_fig(fig, "funds_3d.png")
    return paths


def equation(name: str, line: str, caption: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 2.6), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.62, line, ha="center", va="center", fontsize=28, color=INK)
    ax.text(0.5, 0.22, caption, ha="center", va="center", fontsize=12, color=FOREST, fontstyle="italic")
    path = FIG / name
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return path


def draw_equations() -> dict[str, Path]:
    return {
        "omega": equation("eq_omega.png", r"$|\Omega\rangle=\int \sigma\cdot\beta\cdot[v+\xi]\,d\omega$", "Existence, resonance, the manifest value, and the flux."),
        "raw": equation("eq_raw.png", r"$A(\theta)=\theta\cdot e^{-\lambda\xi}$", "Raw damping shrinks the whole series."),
        "flux": equation("eq_flux.png", r"$A(y)=\mu+(y-\mu)\cdot e^{-\lambda\xi}$", "Flux damping keeps the center and shrinks the wiggle."),
        "closed": equation("eq_closed.png", r"$A_{40}(y)=\mu+(y-\mu)\cdot e^{-40\xi}$", "Forty passes with a fixed center."),
        "jvp": equation("eq_jvp.png", r"$dA=e^{-\lambda\xi}\,d\theta-\lambda\,\theta\,e^{-\lambda\xi}\,d\xi$", "The derivative JAX checks."),
        "sign": equation("eq_sign.png", r"$\mathrm{bits}\oplus \mathtt{0x80000000}$", "Sign-bit negation inside the float32 encoding."),
    }


def flow_chain(name: str, title: str, nodes: list[tuple[str, str]]) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    n = len(nodes)
    fig, ax = plt.subplots(figsize=(2.7 * n, 3.8), facecolor="white")
    ax.set_xlim(0, n * 2.7)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    fills = [MAROON, FOREST, SAFFRON, BLUE, "#6b3a22"]
    rights = []
    for i, (head, body) in enumerate(nodes):
        x = 0.22 + i * 2.7
        ax.add_patch(FancyBboxPatch((x, 0.4), 2.2, 2.45, boxstyle="round,pad=0.03,rounding_size=0.1", facecolor=fills[i % len(fills)], edgecolor="none"))
        ax.text(x + 1.1, 2.4, head, ha="center", va="center", color=CARD, fontsize=12, fontweight="bold")
        wrapped = "\n".join(__import__("textwrap").wrap(body, 26))
        ax.text(x + 1.1, 1.3, wrapped, ha="center", va="center", color=CARD, fontsize=9.5, linespacing=1.3)
        rights.append(x + 2.2)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((rights[i] + 0.04, 1.55), (rights[i] + 0.42, 1.55), arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.4))
    fig.suptitle(title, color=INK, fontsize=14)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return path


def flow_bounds(name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 4.4), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.25, 0.3), 5.5, 3.5, boxstyle="round,pad=0.04,rounding_size=0.1", facecolor=FOREST, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((6.25, 0.3), 5.5, 3.5, boxstyle="round,pad=0.04,rounding_size=0.1", facecolor=MAROON, edgecolor="none"))
    ax.text(3.0, 3.35, "Shown on 14 November", ha="center", color=CARD, fontsize=13, fontweight="bold")
    ax.text(9.0, 3.35, "Left for later", ha="center", color=CARD, fontsize=13, fontweight="bold")
    left = "Flux map on the chip.\nJulia, JAX, and Triton agree.\nTen minutes, one board and four.\nAbout four times the jobs.\nAbout 1,465 GB/s at 50 million nodes."
    right = "A product library.\nA paid design partner.\nA silence gate that earns its keep.\nA calorimeter, an optical core,\na material quasicrystal."
    ax.text(3.0, 1.7, left, ha="center", va="center", color=CARD, fontsize=11, linespacing=1.45)
    ax.text(9.0, 1.7, right, ha="center", va="center", color=CARD, fontsize=11, linespacing=1.45)
    fig.suptitle("What this record can carry", color=INK, fontsize=14)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cover", fontName="Times-Bold", fontSize=20, leading=24, textColor=M, spaceAfter=4))
    base.add(ParagraphStyle("Sub", fontName="Times-Italic", fontSize=11, leading=14, textColor=F, spaceAfter=8))
    base.add(ParagraphStyle("H", fontName="Times-Bold", fontSize=13, leading=16, textColor=M, spaceBefore=11, spaceAfter=6))
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=11, leading=15, textColor=I, alignment=TA_JUSTIFY, spaceAfter=8))
    base.add(ParagraphStyle("Cap", fontName="Times-Italic", fontSize=10, leading=13, textColor=F, spaceBefore=3, spaceAfter=10))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8, leading=10.5, textColor=I))
    base.add(ParagraphStyle("Head", fontName="Times-Bold", fontSize=8, leading=10.5, textColor=colors.HexColor(CARD)))
    return base


def paint(canvas, doc, footer: str) -> None:
    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.setFillColor(M)
    canvas.rect(0, letter[1] - 0.38 * inch, letter[0], 0.38 * inch, fill=1, stroke=0)
    canvas.setFillColor(G)
    canvas.rect(0, letter[1] - 0.44 * inch, letter[0], 0.06 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(CARD))
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(0.62 * inch, letter[1] - 0.24 * inch, "Ceraph Corporation   ·   Kerala, India   ·   www.ceraphco.com")
    canvas.setFillColor(F)
    canvas.setFont("Times-Italic", 8)
    canvas.drawString(0.62 * inch, 0.36 * inch, footer)
    canvas.drawRightString(letter[0] - 0.62 * inch, 0.36 * inch, str(doc.page))
    canvas.restoreState()


def pic(path: Path, max_w: float = 7.2 * inch, max_h: float = 5.35 * inch) -> Image:
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
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.white]),
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
        CondPageBreak(5.6 * inch),
        Paragraph(title, s["H"]),
        pic(path),
        Paragraph(caption, s["Cap"]),
    ])


def how_story(figures: dict[str, Path], flows: dict[str, Path]) -> list:
    s = styles()
    return [
        Paragraph("How the technology works", s["Cover"]),
        Paragraph("A plain-language brief for patent counsel. Ceraph Corporation, Kerala, India. 14 November 2026. chief@ceraphco.com · www.ceraphco.com", s["Sub"]),
        Paragraph("This brief explains an in-process way to keep the steady part of a numerical series and to quiet only the disturbance. The explanation uses ordinary examples first, then the formula, then the measurements from two notebooks run the same evening on NVIDIA RTX PRO 6000 Blackwell Server Edition boards. One notebook drove a single board. The other drove four boards. Each notebook contains two independent suites, one written by a first assistant and one written separately and saved into its own output folder. The pictures are redrawn from those saved measurements so the words sit under the bars.", s["Body"]),
        Paragraph("The idea, in one pass", s["H"]),
        pic(flows["path"], max_h=3.5 * inch),
        Paragraph("Read left to right. A noisy row arrives. A short neighborhood supplies a local center and a measure of how wild that neighborhood is. The map keeps the center and shrinks only the departure from it. The row that leaves is steadier, and the level of the signal is still there.", s["Cap"]),
        Paragraph("Three languages carry one formula", s["H"]),
        pic(flows["languages"], max_h=3.5 * inch),
        Paragraph("Julia reads how dense the disturbance is and chooses a block width of 128, 256, 512, or 1024. JAX, on the graphics chip, checks that the map and its derivative match the written formula. Triton runs the map at that block, inside the chip's on-chip memory. On this night the three agreed, on one board and on each of four boards.", s["Cap"]),
        Paragraph("One board, and four", s["H"]),
        pic(flows["benches"], max_h=3.5 * inch),
        Paragraph("A heavy job is 16,777,216 numbers, passed through the map forty times. On the four-board notebook the first suite measured four such jobs in a queue at 29.51 milliseconds, and the same four jobs at once at 7.47 milliseconds. Over ten minutes the one-board rate was about 137 jobs a second and the four-board rate was about 543 to 549 jobs a second, four times the work.", s["Cap"]),
        Paragraph("The formula, written large", s["H"]),
        pic(figures["flux"]),
        Paragraph("Mu is the local center. The parenthesis is the wiggle around that center. Xi is how wild the neighborhood is. Lambda was 1 in every test below. The first term keeps the center. The exponential multiplies only the wiggle.", s["Cap"]),
        pic(figures["raw"]),
        Paragraph("Raw damping multiplies the entire series, center included. Near zero it can look better than flux. When the series sits at an offset of five, the same shrink pulls the signal down with the noise, and the error rises from under 1 to about 18.", s["Cap"]),
        Paragraph("Layman explainer", s["H"]),
        Paragraph("A song in a busy kitchen. The song is the steady thing. The clatter of plates is the disturbance. A moving average turns both down, so the song goes soft along with the noise. This method keeps the song at its own loudness and turns down only the clatter. On the seed used for the four-board quality check, flux error stayed at 1.472 when the whole signal was shifted by five. A moving average sat at 1.675. Raw damping, which turns the whole recording down, rose from 0.774 to 18.17.", s["Body"]),
        Paragraph("A photograph on a shaky day. The person in the frame should stay put. The shake should go. The invention is that kind of steadying, applied to a long row of numbers, in the same pass that reads them. Counsel can describe it as keeping a local center and damping only the fluctuation.", s["Body"]),
        Paragraph("A workshop. Hiring three more people does not finish a tiny screw faster, because picking the screw up takes longer than turning it. A heavy timber is different. On these chips a short job is the screw: launch cost dominates, and a second board does not help. A heavy job is the timber. Four boards finished about four times as many heavy jobs as one board, in the same ten minutes.", s["Body"]),
        Paragraph("Four musicians and a small folded note. A common habit in multi-chip computing is to ship the entire score to every player and then vote. The measured alternative, in the four-board notebook, was to let each board damp its own part and to pass a 16-byte state across the link instead of a few hundred megabytes of the whole tensor. The byte count is the architectural point. The large memory-bandwidth numbers later in this brief are a different measurement: how fast one board can read and write the series itself.", s["Body"]),
        Paragraph("A quiet room, and a gate. Julia also computes a stability index. On the ordinary test series that index was about 0.000036, under a horizon of 0.20, so the gate stayed closed. A spiky series, density about 7, selected the widest block, 1024. The gate is a decision about whether to launch. Tonight's record shows the decision being computed and matched. It does not yet show a silence detector that has earned a quality score of its own.", s["Body"]),
        Paragraph("Where a person would meet it", s["H"]),
        Paragraph("A microphone in a workshop, or a hydrophone, or a vehicle sensor, produces a stream whose level matters and whose jitter does not. A studio console, a scientific logger, and the inner state of a model are the same shape of problem: there is a center worth keeping and a fluctuation worth reducing, and the reduction has to happen while the stream is moving, on the chip that already holds the stream. The ten-minute runs are the evidence that this particular reduction can sit there, at boost clock, near half the board's power limit, without the values becoming non-finite.", s["Body"]),
        Paragraph("Comparison with current practice", s["H"]),
        Paragraph("The yardsticks below were run in the same notebooks, on the same class of board. They are not a graphics chip against a laptop processor.", s["Body"]),
        table(
            ["Yardstick", "What it does", "What 14 November showed"],
            [
                ["Leave the series alone", "Keep every sample", "Error about 6.3 to 6.9 on the seed-11 series."],
                ["Moving average", "Blur a short window", "Error about 1.64 to 1.68. At one million points the pool took about 0.015 ms."],
                ["Raw damping", "Shrink the whole series", "Error about 0.76 near zero, and about 18.2 when the series sits at five."],
                ["Flux map, this work", "Keep the center, shrink the wiggle", "Error 1.43 to 1.47 at both offsets. Triton matched the formula within 5×10^-7."],
                ["Eager heavy loop, earlier the same day", "Forty passes in ordinary PyTorch", "About 17.5 ms on a comparable board."],
                ["Triton heavy job", "The same forty passes, fused", "About 6.6 to 7.5 ms for one job. Four at once, 7.47 ms in the first suite."],
                ["Four boards against one", "The same heavy job, ten minutes", "137 jobs/s on one board. 543 to 549 jobs/s on four. About 4×."],
            ],
            [1.45 * inch, 2.05 * inch, 3.65 * inch],
        ),
        Spacer(1, 8),
        figure_block(
            "The error, drawn as a 3D histogram",
            figures["quality"],
            "Front row: the series near zero. Back row: the same kind of series shifted by five. The untouched series and the moving average stay put. Raw damping is short in front and very tall behind, at 18.17. Flux is the short bar in both rows, at 1.472. Lower is better. These bars are the four-board quality check, seed 11.",
        ),
        Paragraph("The measurements, side by side", s["H"]),
        Paragraph("Hardware in both notebooks: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, 97,887 MiB of GDDR7 with ECC, a 600 W board limit, and a stated peak memory bandwidth of 1,597 GB/s. Software in the independent suite: PyTorch 2.15.0 with CUDA 13.2, Triton 3.9.0, JAX 0.12.0.1 on cuda, and Julia 1.13.4. The first suite used the same generation of that stack, installed with its own installer, and reported Triton 3.9.0 and CUDA 13.2 as well.", s["Body"]),
        table(
            ["Ten-minute soak", "First suite, one board", "Independent suite, one board"],
            [
                ["Jobs", "81,634 in 600.0 s", "82,221 in 600.2 s"],
                ["Jobs per second", "136.1", "137.0"],
                ["Mean power", "305.1 W", "292.7 W"],
                ["Peak power", "337.7 W", "321.4 W, in the first minute"],
                ["Headroom under 600 W", "49%", "51%"],
                ["Temperature", "34 °C to 45 °C", "32 °C to a 45 °C plateau"],
                ["SM clock", "2317 to 2422 MHz", "2317 to 2422 MHz"],
                ["Occupancy", "99%", "100% after the opening sample"],
                ["Energy per heavy job", "2,242 mJ", "2,137 mJ"],
                ["Values stayed finite", "yes, same-input drift 0.0", "yes"],
            ],
            [1.7 * inch, 2.7 * inch, 2.75 * inch],
        ),
        Spacer(1, 8),
        table(
            ["Ten-minute soak", "First suite, four boards", "Independent suite, four boards"],
            [
                ["Jobs", "325,732 in 600.0 s", "329,404 in 600.2 s"],
                ["Jobs per second", "542.9", "548.8"],
                ["Against one board", "about 4.0×", "4.01×"],
                ["Cluster mean power", "1,192 W", "1,109 W"],
                ["Per-board mean", "296 to 301 W", "273 to 282 W"],
                ["Peak temperature", "48 to 49 °C", "47 to 48 °C"],
                ["Clock ceiling", "2422 MHz on each board", "2422 MHz on each board"],
                ["Energy per heavy job", "2,195 mJ", "2,020 mJ"],
                ["Four jobs at once", "7.47 ms, queue 29.51 ms", "8.34 ms for the concurrent wave"],
                ["Values stayed finite", "yes, same-input drift 0.0", "yes"],
            ],
            [1.7 * inch, 2.7 * inch, 2.75 * inch],
        ),
        Spacer(1, 8),
        Paragraph("A note counsel should keep with the drift column. The first suite repeated the same input and compared the outputs. That residual was 0.0. The independent soak fed each output back in as the next input, so its drift field records the series walking toward its center over eighty thousand applications. Every value in that run stayed finite. The numerical check for the independent suite is the certification: sign-bit negation residual 0, JAX derivative residual 0, Triton within 4.8×10^-7 of the flux formula, on each of the four boards.", s["Body"]),
        Paragraph("A second note, on a variance that prints as zero. The first suite's roofline kernel reports the variance of its synthetic series falling to 0.000000. That kernel's denominator saturates. The flux map measured in the quality check keeps about 12% of the variance, and a single flux pass on uniform noise keeps about 43%. The deployable claim is the second behavior: the center remains, and a fraction of the fluctuation remains with it.", s["Body"]),
        figure_block(
            "One board, ten minutes of power",
            figures["one_power"],
            "Independent suite. Eighteen bars sampled across 600.2 seconds, from a reading every five seconds. The first bar is the idle instant, 82 W. The draw then settles near 293 W. The peak in the file is 321 W, during the first minute, against a 600 W limit.",
        ),
        figure_block(
            "One board, ten minutes of temperature",
            figures["one_temp"],
            "The same run, on its own scale so 45 degrees is visible. The board started at 32 °C and reached a plateau of 45 °C before the second minute. It stayed there. The clock, in the next picture, rose from 2317 MHz into a held 2422 MHz.",
        ),
        figure_block(
            "One board, the clock",
            figures["one_clock"],
            "SM clock in megahertz. After the opening minute the bars sit on 2422. The chip was busy at 100% on every sample after the first. Mean energy was 2,137 millijoules per heavy job.",
        ),
        figure_block(
            "Four boards, ten minutes of power",
            figures["four_power"],
            "Independent suite. Each color is one RTX PRO 6000. After the opening sample the four boards sit together between about 274 and 291 watts. Means across the whole soak were 277, 282, 277, and 273 watts. Peaks were 291 to 302 watts.",
        ),
        figure_block(
            "Four boards, ten minutes of temperature",
            figures["four_temp"],
            "Peaks of 47 and 48 °C. No board left the boost clocks. The ceiling on every board was 2422 MHz. The cluster drew 1,109 watts on average and finished 329,404 heavy jobs.",
        ),
        figure_block(
            "Two suites, one job rate",
            figures["rates"],
            "Gold and blue are the first suite. Forest and maroon are the independent suite. One board is the short pair, near 137 jobs a second. Four boards are the tall pair, near 543 and 549. The two authors land on the same step.",
        ),
        figure_block(
            "How full the memory pipe was",
            figures["roof"],
            "The board's stated peak is 1,597 GB/s. The first suite measured 1,467 GB/s at 50 million nodes, in 0.409 ms. The independent suite measured 0.546 ms at the same length. Counting the three values the kernel loads and the one value it stores, that timing is 1,465 GB/s, 91.7% of peak. At one million nodes the independent figure is 1,285 GB/s. The saved JSON for that suite counted twelve bytes per node and therefore prints a lower percentage. The picture uses sixteen.",
        ),
        Paragraph("Certification, the part that is not a soak", s["H"]),
        Paragraph("Before either ten-minute run, each suite checked the formula. On the independent four-board run the density of the seed-11 draw was 0.793. Julia and the Python ladder both chose block 256, grid 64. The spiky case chose 1024. Sign-bit exclusive-or against arithmetic negation returned 0 on every board. The JAX directional derivative returned 0 on every board. Triton matched the flux reference within 4.8×10^-7. The first suite, on all four boards, reported sign-bit error 0, derivative error 0, Triton error 2.38×10^-7, and a forty-pass closed form within 2.46×10^-7.", s["Body"]),
        Paragraph("On the one-board certify file, flux error was 1.427 at offset 0 and 1.448 at offset 5, against a moving average of 1.640 and 1.645. Raw damping was 0.759 and then 18.18. Flux kept about 12% of the variance. The four-board quality check, reseeded so the noise draw repeats, held flux at exactly 1.472 at both offsets and raw damping at 0.774 and then 18.17.", s["Body"]),
        Paragraph("What this brief can support", s["H"]),
        pic(flows["bounds"], max_h=3.7 * inch),
        Paragraph("Counsel can describe the flux map, the density ladder, the agreement of Julia, JAX, and Triton, the sign-bit negation, the behavior when a series is shifted, the ten-minute power and temperature record, the four-board job rate, and the memory traffic at large length. A product, a paid customer, a silence detector with a passing quality score, and any physical device beyond the graphics boards were not in the room.", s["Cap"]),
        Paragraph("A sentence for the specification. The invention is an in-process map that preserves a local center and damps only the deviation from that center, the block width chosen by a density ladder, the map and its derivative checked by a differentiable program on the graphics processor, and the map executed as a fused kernel. On 14 November 2026 two independent implementations of that map ran for ten minutes on one NVIDIA RTX PRO 6000 Blackwell Server Edition and on four of them, stayed finite, held boost clocks, and drew about half of each board's 600 watt limit, while four boards completed about four times the heavy jobs of one board.", s["Body"]),
        Paragraph("Sources. The one-board notebook and the four-board notebook executed on DGX Cloud on 14 November 2026. The independent telemetry archives are spec_session_out_one_independent and spec_session_out_four_independent. The first suite wrote its scorecards into the company repository under benchmarks/rtx_pro_6000_blackwell, including the ten-minute scorecards committed the same evening. This brief uses both.", s["Body"]),
        Paragraph("Absolute remains untouched, only the error is taken away.", s["Cover"]),
        Paragraph("Ceraph Corporation · Kerala, India · chief@ceraphco.com · www.ceraphco.com", s["Sub"]),
    ]


def tech_story(figures: dict[str, Path], flows: dict[str, Path]) -> list:
    s = styles()
    return [
        Paragraph("Technical record and investor memorandum", s["Cover"]),
        Paragraph("Ceraph Corporation, Kerala, India. 14 November 2026. chief@ceraphco.com · www.ceraphco.com. For the company file and for a first conversation with investors. Planning figures are estimates for that conversation. They are not a forecast of revenue and they are not an offer of securities.", s["Sub"]),
        Paragraph("Abstract", s["H"]),
        Paragraph("This memorandum records a measured computing architecture on the evening two independent implementations spent ten uninterrupted minutes under load on NVIDIA Blackwell graphics boards. The enabled invention is a flux form of Abhaya damping. A local center is preserved. Only the deviation from that center is multiplied by a shrinking exponential of the local instability. A density ladder, shared by a Julia macro named @sphota and by a Python twin, chooses the launch block: 128, 256, 512, or 1024. A JAX primitive on the graphics processor carries the same map and an analytical directional derivative. An OpenAI Triton kernel executes the map. A sign-bit exclusive-or negates a float32 value in the same family of kernel.", s["Body"]),
        Paragraph("The first suite and the independent suite were written separately, installed separately, and saved into separate folders. On one RTX PRO 6000 Blackwell Server Edition they completed 81,634 and 82,221 heavy jobs in 600 seconds, at 136 and 137 jobs per second, with mean draws of 305 W and 293 W and a temperature plateau of 45 °C. On four such boards they completed 325,732 and 329,404 heavy jobs, at 543 and 549 jobs per second, about four times the one-board rate, with cluster draws of 1,192 W and 1,109 W and peak temperatures of 47 to 49 °C. Every reported value stayed finite. Sign-bit negation and the JAX derivative were exact. Triton matched the flux formula within 5×10^-7. At 50 million nodes both suites reached about 1,465 GB/s, against a stated peak of 1,597 GB/s.", s["Body"]),
        Paragraph("Flux mean squared error on the seed-11 series stayed between 1.43 and 1.47 when the latent series was shifted by five. A moving average sat near 1.64 to 1.68. Raw damping, which multiplies the whole series, rose from about 0.76 to about 18.2. That split is the invention. The raise described later buys the people and the months to turn the record into one callable library and one paid wedge. It does not buy a finished market.", s["Body"]),
        Paragraph("The company", s["H"]),
        Paragraph("Ceraph Corporation is based in Kerala, India. Correspondence for this memorandum goes to chief@ceraphco.com. The public site is www.ceraphco.com. The work is a long research program, carried in notebooks across Colab, AWS, and now DGX Cloud, whose enabled core is the map measured here. The company is at the stage of a provisional filing and a first institutional conversation. It does not have design-partner revenue as of this date.", s["Body"]),
        Paragraph("Mathematical framework", s["H"]),
        Paragraph("The continuous picture in the notebooks is an omega-dit. An existence density, a resonant weight, and a sum of manifest value plus unmodeled flux are integrated. The session did not need the full integral on the chip. It needed the split between what is steady and what is flux.", s["Body"]),
        pic(figures["omega"]),
        Paragraph("Sigma is existence density, at least zero. Beta is the resonant weight, greater than zero. v is the manifest value. Xi is the flux. The integral is the notebook picture. The kernel implements the split that picture names.", s["Cap"]),
        Paragraph("Raw Abhaya damping multiplies the whole series by a shrinking exponential. It is a fair map when the carrier sits near zero. A large offset is part of the signal, and the same exponential shrinks the signal along with the noise.", s["Body"]),
        pic(figures["raw"]),
        Paragraph("Lambda was 1 in every test in this memorandum. Nothing in the raw formula protects a steady offset.", s["Cap"]),
        Paragraph("Flux damping is the deployable form. Mu is a local center. Xi is a local instability, clamped so the exponential stays a shrink. The center is an additive fixed point: a series that already equals the center is returned unchanged.", s["Body"]),
        pic(figures["flux"]),
        Paragraph("Forty iterations with a fixed center and a fixed xi collapse to one exponential. The first suite matched that closed form within 2.5×10^-7.", s["Body"]),
        pic(figures["closed"]),
        Paragraph("The directional derivative of the raw map is what a later training step needs. JAX binds it as a custom Jacobian-vector product. On this night the derivative check returned 0 on every board that was asked.", s["Body"]),
        pic(figures["jvp"]),
        Paragraph("Sign-bit negation flips the float32 sign by an unsigned exclusive-or with 0x80000000, formed as an unsigned vector so the constant is not parsed as a signed immediate. The residual against arithmetic negation was 0. It is the software image of a paired cancellation: half of (v+xi) plus half of (v-xi) returns v, without a second buffer for the antithesis.", s["Body"]),
        pic(figures["sign"]),
        Paragraph("The hybrid language", s["H"]),
        pic(flows["languages"], max_h=3.5 * inch),
        Paragraph("Om-lang is the project name for one formula living in three tiers. Julia is the control tier. The macro @sphota reads a density and returns a block. Below 0.5 the block is 128. Below 1 it is 256. Below 2 it is 512. Otherwise it is 1024. JAX on CUDA is the differentiable tier. Triton, launched from PyTorch tensors so the CUDA context already exists, is the kernel tier. Python and NumPy hold the reference. A run is accepted when the Julia block matches the Python ladder, the JAX value and the derivative match the formula, and Triton matches the formula on the device that will do the work.", s["Cap"]),
        Paragraph("On ordinary noise the density was 0.793 to 0.798 and both ladders chose 256. On the spiky series the density was about 7 and both chose 1024. A stability index of about 3.6×10^-5 stayed under the 0.20 horizon, so the gate remained closed. That is a computed decision, recorded in the certify file, not yet a detector with its own accuracy score.", s["Body"]),
        Paragraph("Kinship with current science", s["H"]),
        Paragraph("The exponential shrink is kin to classical damping and to an exponential moving weight, with one structural difference: the weight multiplies the deviation, not the observation. A moving average is the linear cousin. It reduces variance by mixing neighbors, and it pays for that mix with bias. Flux can leave less squared error than that mix, which is what the seed-11 series showed, without being given the latent series as an input.", s["Body"]),
        Paragraph("The Triton kernel is a fused elementwise program: load a tile, apply the exponential, store the tile. From a few hundred points to a million points its time stays near 0.013 milliseconds, because the launch dominates the arithmetic. At 50 million nodes the arithmetic shows, and the pipe fills. A current parallel is the split, now ordinary in graphics software, between a fused kernel and a chain of eager operations. An eager forty-pass loop measured earlier the same day took about 17.5 ms. The fused kernel took about 7 ms.", s["Body"]),
        Paragraph("Another parallel is dual-mode estimation. An AWS campaign on 11 November measured a two-mode Gaussian estimator at mean squared error 0.614, a single-mode Wiener filter at 0.694, a moving average at 1.088, and a median filter at 1.235. Flux is not that estimator. Both lines share the habit of using a second view of the disturbance instead of only blurring the observation.", s["Body"]),
        Paragraph("A third parallel is how a machine room should be used. Sharding one small vector across four boards made an earlier run slower, because the launch and the join cost more than the arithmetic. Independent heavy jobs are the other pattern. Four of them, in the first suite, finished together in 7.47 ms against 29.51 ms in a queue. The independent suite's ten-minute job rate scaled by 4.01. The 16-byte state passed between boards in the first suite's semantic-bridge cell is the same idea at the level of the message: ship the disturbance, not the tensor. That cell reported 16 bytes transferred against 319,999,984 bytes for a full reduction of its buffers. The bandwidth figure in that same cell, 54.6 GB/s, belongs to that kernel's runtime. It is not the 1,465 GB/s roofline, and the two should not be folded together.", s["Body"]),
        Paragraph("Landauer's principle, a physical calorimeter, a periodically poled lithium-niobate core, and a material quasicrystal remain in the longer notebooks. They were not in the room on 14 November. Naming them keeps the kinship. It does not move them into the enabled claim or into the use of funds as if the devices had been built.", s["Body"]),
        Paragraph("Where the tests were run", s["H"]),
        table(
            ["Laboratory", "When", "What was run", "What it is allowed to support"],
            [
                ["Google Colab", "Notebooks through the spring, saved outputs", "Early Blackwell kernel medians, including 0.0082 ms at 8,192 points and 0.0062 ms at 10,000.", "The kernel was already in the sub-millisecond band. A 0.2379 ms line in those notebooks is a printed budget, not a timed opponent."],
                ["AWS Braket", "11 November 2026", "A classical filter campaign, a live simulator task, and a small hybrid job.", "An earlier estimator beat a moving average, a median, and single-mode Wiener on that campaign. It is not this flux kernel."],
                ["DGX Cloud, earlier the same day", "14 November, morning and early evening", "Agreement of the three languages, flux quality, and a finding that sharding a short vector across four boards was slower than one board.", "The stack works. Short jobs do not scale by being split. A silence gate based on window variance missed its quality bar and was not the subject of the night runs."],
                ["DGX Cloud, first suite", "14 November, one-board and four-board notebooks", "Ten-minute soaks, heavy-job scaling, and a roofline to 1,467 GB/s. Same-input drift 0.0.", "Endurance and scaling for that implementation. Its synthetic variance of zero is a saturated formula, not the flux map."],
                ["DGX Cloud, independent suite", "14 November, the same two machines", "Certification, quality, latency, roofline, and ten-minute soaks, saved apart from the first suite.", "A second implementation reproduces the power, the clock, the job rate, the flux error, and the bandwidth."],
            ],
            [1.15 * inch, 1.15 * inch, 2.4 * inch, 2.45 * inch],
        ),
        Spacer(1, 8),
        Paragraph("Proven execution", s["H"]),
        Paragraph("The tables are taken from the notebook outputs and from the telemetry files one_soak_10min.json, one_certify_roofline.json, and four_full_suite.json. Heavy job means 16,777,216 float32 values and 40 flux passes.", s["Body"]),
        table(
            ["Check", "One board", "Four boards"],
            [
                ["Machine", "RTX PRO 6000 Blackwell, one device", "Four of the same, cuda:0 through cuda:3"],
                ["Density, ordinary series", "0.793, block 256", "0.793, block 256 on each board"],
                ["Spiky block", "1024", "1024"],
                ["Sign-bit residual", "0", "0 on each board"],
                ["JAX derivative residual", "0", "0 on each board"],
                ["Triton versus flux formula", "within 5×10^-7", "4.8×10^-7 on each board"],
                ["Forty-pass closed form, first suite", "2.46×10^-7", "2.46×10^-7 on each board"],
                ["Flux MSE, offset 0 and offset 5", "1.427 and 1.448", "1.472 and 1.472"],
                ["Moving-average MSE", "1.640 and 1.645", "1.675 and 1.675"],
                ["Raw MSE at offset 5", "18.18", "18.17"],
                ["Flux variance remaining", "about 12%", "11.7%"],
                ["One million points, Triton", "0.0129 ms", "0.0129 ms on the timed board"],
                ["50 million nodes", "0.546 ms, 1,465 GB/s corrected", "same timing, run on board 0 of the four-board machine"],
                ["First suite, 50 million nodes", "0.409 ms, 1,467 GB/s", "reported from the one-board roofline cell"],
                ["Ten-minute jobs", "82,221 independent, 81,634 first suite", "329,404 independent, 325,732 first suite"],
                ["Ten-minute power", "293 W mean, 321 W peak", "1,109 W cluster mean, peaks 291 to 302 W per board"],
                ["Ten-minute heat", "32 °C to 45 °C", "peaks 47 to 48 °C, first suite to 49 °C"],
                ["Clock", "held at 2422 MHz", "held at 2422 MHz on each board"],
            ],
            [1.85 * inch, 2.65 * inch, 2.65 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The independent queued timing of 44.3 ms includes copies of each board's buffers onto one device inside the timer. It is not the clean queue. The clean queue is the first suite's 29.51 ms against 7.47 ms concurrent, a 3.95× reduction in wait. The clean scaling number for the independent suite is the ten-minute rate, 548.8 against 137.0, a factor of 4.01.", s["Body"]),
        figure_block("Error histogram", figures["quality"], "Lower is better. Flux is short in both rows. Raw is short only when the carrier is near zero. This is the reason flux, and not raw, is the map a product would call."),
        figure_block("One-board endurance, power", figures["one_power"], "Independent suite, 600.2 seconds, 82,221 jobs. Sustained draw about 293 W. The 600 W limit is the board's nameplate, not a target."),
        figure_block("One-board endurance, temperature", figures["one_temp"], "A rise of 13 degrees, then a plateau. Boost clocks continued. Mean occupancy after the opening sample was 100 percent."),
        figure_block("One-board endurance, clock", figures["one_clock"], "2317 MHz while the chip came up, then 2422 MHz held. The first suite recorded the same ceiling and the same floor."),
        figure_block("Four-board endurance, power", figures["four_power"], "Four boards, one soak, independent suite. They stay in a band. The first suite's per-board means were a little higher, 296 to 301 W, in the same band of the nameplate."),
        figure_block("Four-board endurance, temperature", figures["four_temp"], "Peaks 47 and 48 °C in this suite, 48 and 49 °C in the first suite. Ten minutes did not walk the temperature out of that band."),
        figure_block("Job rate, both authors", figures["rates"], "The step from one board to four is the enterprise result. About four times the heavy jobs, at a cluster power near 1.1 to 1.2 kilowatts, each board still near half of 600 W."),
        figure_block("Roofline", figures["roof"], "1,465 and 1,467 GB/s at 50 million nodes, two suites. 91.7 percent of the 1,597 GB/s peak on the corrected independent count. Shorter lengths are launch-bound and sit lower, which is the expected shape."),
        Paragraph("Why the work can be described as a beginning", s["H"]),
        Paragraph("A beginning is the moment a long idea survives contact with the machine it was written for, and then survives a second implementation. This one did. Three tiers of one formula agreed. The form that keeps the center held its error when the raw form did not. A heavy job stayed in a small number of milliseconds. Four boards did four times the jobs. Ten minutes of repetition left finite values, boost clocks, and a temperature that plateaued under 50 °C on a 600 W board. The potential, said plainly, is an in-process cleaning step a later product can drop into a pipeline that already lives on graphics chips: a sensor stream, a studio signal, a scientific series, or the inner state of a model. The step is small enough to schedule, differentiable enough to train, and fused enough to repeat.", s["Body"]),
        Paragraph("What a raise is for", s["H"]),
        Paragraph("Two bands are enough for a first conversation. Both are planning estimates. Salaries are loaded roughly and stated in US dollars. Cloud is an average, not a peak invoice. Counsel covers the path from the provisional now in draft toward a utility filing. Geography and the kind of investor can move either band by about a quarter. The company is in Kerala. The figures are not a local payroll quotation. They are a budget a conversation can edit.", s["Body"]),
        Paragraph("Working ask: 1.25 million US dollars for 18 months. This is the band that hires, keeps a research year, and still has a reserve.", s["Body"]),
        table(
            ["Use", "Basis", "Amount"],
            [
                ["Founder", "18 months at 10,000 a month", "$180,000"],
                ["Kernel engineer", "18 months, loaded near 16,000 a month", "$288,000"],
                ["Research scientist", "12 months, loaded near 14,000 a month", "$168,000"],
                ["Finance and operations, fractional", "2,000 a month", "$36,000"],
                ["Counsel and the utility path", "Beyond the provisional already in motion", "$120,000"],
                ["Entity, insurance, accounting", "18 months", "$40,000"],
                ["Blackwell and other cloud", "Planned average 8,000 a month", "$144,000"],
                ["Bench held for later embodiments", "Not spent as if those devices exist", "$30,000"],
                ["Papers, site, two gatherings, travel", "A small public face", "$70,000"],
                ["Design-partner program", "Our cost to show up. A partner's fee is income, below.", "$40,000"],
                ["Recruiting", "Two hires", "$20,000"],
                ["Reserve", "About 9 percent", "$114,000"],
                ["Working ask", "18 months", "$1,250,000"],
            ],
            [2.35 * inch, 3.15 * inch, 1.65 * inch],
        ),
        Spacer(1, 8),
        figure_block(
            "Where the working ask goes",
            figures["funds"],
            "People are the tall bar: founder, engineer, scientist, and fractional operations, 672 thousand dollars. Counsel and the entity are 160. Cloud is 144. The public face and the partner program are 110. A small bench and recruiting are 50. The reserve is 114. The sum is 1.25 million.",
        ),
        Paragraph("Lean ask: 700,000 US dollars for 12 months, if the round is smaller and the scientist starts later. Founder 120,000. Engineer for 12 months 192,000. Scientist for 6 months 84,000. Counsel 80,000. Cloud at 6,000 a month 72,000. The public face and partner program 50,000. Operations 36,000. Reserve 66,000.", s["Body"]),
        Paragraph("The lean ask keeps the measurement program and one engineer. The working ask matches a continuation that includes a second researcher, a full utility path, and enough cloud to repeat this night on new series rather than on a single seed.", s["Body"]),
        Paragraph("Projected income over three years", s["H"]),
        Paragraph("Income below rests on one assumption, and the assumption is stated before the numbers. A design partner who already runs a graphics pipeline pays on the order of 60,000 US dollars a year to put this map inside one of their streams, once a small library exists and the provisional is on file. Pilots in the first year can be smaller, on the order of 60,000 for a paid pilot rather than a retainer. None of this income exists today. The table is a planning band. It is not a booking, a forecast, or a promise.", s["Body"]),
        table(
            ["Year after the raise closes", "Quiet case", "Working case", "One wedge takes"],
            [
                ["Year 1", "No paid partner. The year builds the library, repeats the measurement, and finishes the filing. Income about 0.", "Two paid pilots, about $120,000.", "The same as the working case. A wedge is not yet a product."],
                ["Year 2", "One retainer, about $60,000.", "Four retainers, about $400,000.", "Six to eight retainers, about $0.8 million to $1.2 million."],
                ["Year 3", "Two retainers, about $150,000.", "Eight retainers, about $1.2 million.", "A product in one vertical. Planning band $2.5 million to $4 million."],
            ],
            [1.35 * inch, 1.9 * inch, 1.9 * inch, 2.0 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The quiet case is a company that keeps the science alive and finds one friend. The working case is a company that sells a careful service to teams who already own the hardware. The right-hand case is what happens if one of those teams becomes a market, for example a studio signal path or a sensor line, and the license becomes a product with a support burden. A consumer application, a multi-tenant cloud, and a claim of thermodynamic advantage sit outside all three columns.", s["Body"]),
        Paragraph("Costs those income figures have to carry, once they exist, include the people in the ask, cloud for each partner's pilot, and counsel while claims are examined. On the working income path, year 2 at about 400,000 does not yet cover a full 1.25 million burn, and it is not supposed to. The raise is the bridge. Year 3 in the working column, about 1.2 million, is the first year a services practice of this shape could approach covering an ongoing team. The right-hand column is the year a product, if one vertical has actually formed, could exceed that cost. Investors should read the quiet column as the plan that still leaves the company alive, and the right-hand column as an upside that has to be earned one partner at a time.", s["Body"]),
        Paragraph("Risks to say in the same conversation", s["H"]),
        Paragraph("The matched quality win is on a seeded series, not yet inside a customer's model or sensor. An incumbent can ship a blur faster than a new company can ship a library. The architectural advantage is the placement, in the same pass, and the preservation of the center. Deep-tech schedules slip. The silence gate has not earned a quality score. The physical embodiments in the longer notebooks are not built. Two suites agreeing on one night is strong evidence and still one night, on one class of board, on one family of series. The next money is for making that night repeatable on a partner's data.", s["Body"]),
        Paragraph("Conclusion", s["H"]),
        Paragraph("On 14 November 2026, in Kerala's company and on Blackwell boards reached through DGX Cloud, the hybrid language was real, the flux map held its error when the raw map did not, one board and four boards stayed near half their power limit for ten minutes, and a second implementation reproduced the result. Clocks held at 2422 MHz. Temperatures plateaued under 50 °C. Four boards did about four times the heavy jobs. At 50 million nodes the memory pipe reached about 92 percent of its stated peak. The raise of 1.25 million dollars, or 700,000 in the lean band, buys the people and the months to turn that record into one library and one paid wedge, and to carry the provisional through to a utility filing. It does not buy a finished market, and it does not need to.", s["Body"]),
        Paragraph("Absolute remains untouched, only the error is taken away.", s["Cover"]),
        Paragraph("Ceraph Corporation · Kerala, India · chief@ceraphco.com · www.ceraphco.com", s["Sub"]),
    ]


def shade_cell(cell, hex_color: str) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    shd = tc_pr.makeelement(qn("w:shd"), {qn("w:fill"): hex_color, qn("w:val"): "clear"})
    tc_pr.append(shd)


def add_docx(path: Path, blocks: list) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    header = section.header.paragraphs[0]
    run = header.add_run("Ceraph Corporation  ·  Kerala, India  ·  chief@ceraphco.com  ·  www.ceraphco.com")
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x7C, 0x2D, 0x2D)
    run.italic = True
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
            document.add_picture(str(payload), width=Inches(6.4))
        elif kind == "table":
            grid = document.add_table(rows=len(payload), cols=len(payload[0]))
            grid.style = "Table Grid"
            for r, row in enumerate(payload):
                for c, text in enumerate(row):
                    grid.rows[r].cells[c].text = text
                    for paragraph in grid.rows[r].cells[c].paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(8)
                            run.font.name = "Times New Roman"
                            if r == 0:
                                run.bold = True
                                run.font.color.rgb = RGBColor(0xFB, 0xF6, 0xEC)
                    if r == 0:
                        shade_cell(grid.rows[r].cells[c], "7C2D2D")
                    else:
                        shade_cell(grid.rows[r].cells[c], "FFFFFF")
            document.add_paragraph("")
        elif kind == "cap":
            paragraph = document.add_paragraph(payload)
            for run in paragraph.runs:
                run.italic = True
                run.font.color.rgb = RGBColor(0x2C, 0x4A, 0x3E)
                run.font.size = Pt(10)
    footer = section.footer.paragraphs[0]
    fr = footer.add_run("Ceraph Corporation  ·  Kerala, India  ·  14 November 2026")
    fr.font.name = "Times New Roman"
    fr.font.size = Pt(9)
    fr.italic = True
    fr.font.color.rgb = RGBColor(0x2C, 0x4A, 0x3E)
    document.save(path)


def docx_from_story(path: Path, pdf_items: list) -> None:
    blocks: list[tuple[str, object]] = []
    for item in pdf_items:
        if isinstance(item, Paragraph):
            text = item.getPlainText()
            style = item.style.name if item.style is not None else ""
            if style in {"Cover", "H"}:
                blocks.append(("h", text))
            elif style in {"Cap", "Sub"}:
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
        elif isinstance(item, Spacer):
            continue
    add_docx(path, blocks)


def build() -> None:
    figures = draw_figures()
    figures.update(draw_equations())
    flows = {
        "path": flow_chain("flow_path.png", "Keep the center, quiet the wiggle", [
            ("Series", "A noisy row of numbers arrives."),
            ("Neighborhood", "A short window gives a center and a wildness."),
            ("Flux map", "Only the departure from the center is shrunk."),
            ("Steadier row", "The center remains. The clatter is quieter."),
        ]),
        "languages": flow_chain("flow_languages.png", "Julia, JAX, and Triton", [
            ("Density", "Mean absolute size of the disturbance."),
            ("Julia", "@sphota chooses 128, 256, 512, or 1024."),
            ("JAX", "The map and its derivative, on the graphics chip."),
            ("Triton", "The fused kernel runs at that block."),
        ]),
        "benches": flow_chain("flow_benches.png", "One board, then four", [
            ("One board", "About 137 heavy jobs each second, for ten minutes."),
            ("Four boards", "About 543 to 549 heavy jobs each second."),
            ("Same wait", "Four jobs at once finished in 7.47 milliseconds."),
        ]),
        "bounds": flow_bounds("flow_bounds.png"),
    }
    how_pdf = OUT / "nov14-counsel-how-it-works.pdf"
    tech_pdf = OUT / "nov14-investor-technical-record.pdf"
    how_doc = SimpleDocTemplate(
        str(how_pdf), pagesize=letter,
        leftMargin=0.68 * inch, rightMargin=0.68 * inch,
        topMargin=0.72 * inch, bottomMargin=0.62 * inch,
        title="How the technology works", author="Ceraph Corporation",
    )
    tech_doc = SimpleDocTemplate(
        str(tech_pdf), pagesize=letter,
        leftMargin=0.68 * inch, rightMargin=0.68 * inch,
        topMargin=0.72 * inch, bottomMargin=0.62 * inch,
        title="Technical record and investor memorandum", author="Ceraph Corporation",
    )
    how_doc.build(how_story(figures, flows), onFirstPage=lambda c, d: paint(c, d, FOOTER_COUNSEL), onLaterPages=lambda c, d: paint(c, d, FOOTER_COUNSEL))
    tech_doc.build(tech_story(figures, flows), onFirstPage=lambda c, d: paint(c, d, FOOTER_INVESTOR), onLaterPages=lambda c, d: paint(c, d, FOOTER_INVESTOR))
    docx_from_story(OUT / "nov14-counsel-how-it-works.docx", how_story(figures, flows))
    docx_from_story(OUT / "nov14-investor-technical-record.docx", tech_story(figures, flows))
    print({"pdf": [str(how_pdf), str(tech_pdf)]})


if __name__ == "__main__":
    build()
