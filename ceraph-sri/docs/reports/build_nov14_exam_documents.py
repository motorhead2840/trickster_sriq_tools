"""Counsel brief and investor memorandum for the labelled exam on 14 November.

Page content is white. The maroon header, gold rule, and footer ink stay.
Numbers are taken from the one-board and four-board notebooks and from the
exam JSON saved with those runs.
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
DATA = ROOT / "benchmark-results" / "rtx6000-labelled-exam"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures" / "nov14-exam"

INK = "#3a2416"
MAROON = "#7c2d2d"
GOLD = "#a67c42"
SAFFRON = "#c56a1a"
FOREST = "#2c4a3e"
CARD = "#fbf6ec"
BLUE = "#2f4f6f"
M = colors.HexColor(MAROON)
G = colors.HexColor(GOLD)
F = colors.HexColor(FOREST)
I = colors.HexColor(INK)

FOOTER_C = "For counsel  ·  Ceraph Corporation  ·  Kerala, India  ·  chief@ceraphco.com  ·  www.ceraphco.com"
FOOTER_I = "Investor record  ·  Ceraph Corporation  ·  Kerala, India  ·  chief@ceraphco.com  ·  www.ceraphco.com"


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
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return path


def legend_above(fig, handles) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=min(4, len(handles)), frameon=False, fontsize=11, labelcolor=INK)


def bar3(ax, xs, y, heights, color, depth=0.45, width=0.5) -> None:
    ax.bar3d(xs, np.full(len(xs), y), np.zeros(len(xs)), width, depth, heights, color=color, shade=True, edgecolor="#222222", linewidth=0.25)


def draw_figures() -> dict[str, Path]:
    one = load("one_exam.json")
    four = load("four_exam.json")
    paths = {}
    palette = [BLUE, FOREST, GOLD, MAROON]

    fig = plt.figure(figsize=(11.2, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    labels = ["untouched", "average", "raw", "flux"]
    seed = one["melbourne"][0]
    for row, key, color in ((0, "offset_0", BLUE), (1, "offset_5", SAFFRON)):
        vals = [seed[key][name] for name in ("identity", "moving_average", "raw", "flux")]
        bar3(ax, np.arange(4), row, vals, color)
    ax.set_xticks(np.arange(4) + 0.2)
    ax.set_xticklabels(labels, color=INK)
    ax.set_yticks([0.2, 1.2])
    ax.set_yticklabels(["recorded", "+5 C"], color=INK)
    ax.set_zlabel("mean squared error", labelpad=12)
    ax.view_init(elev=24, azim=-58)
    legend_above(fig, [Patch(facecolor=BLUE, label="Recorded temperatures"), Patch(facecolor=SAFFRON, label="Same series, plus 5 C")])
    fig.suptitle("Melbourne daily minimums  ·  seed 11  ·  lower is better", color=INK, fontsize=15, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.96, top=0.78, bottom=0.06)
    paths["melbourne"] = save_fig(fig, "melbourne_mse_3d.png")

    fig = plt.figure(figsize=(11.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    syn = one["synthetic_seed_11"]
    for row, key, color in ((0, "offset_0", BLUE), (1, "offset_5", SAFFRON)):
        vals = [syn[key][name] for name in ("identity", "moving_average", "raw", "flux")]
        bar3(ax, np.arange(4), row, vals, color)
    ax.set_xticks(np.arange(4) + 0.2)
    ax.set_xticklabels(labels, color=INK)
    ax.set_yticks([0.2, 1.2])
    ax.set_yticklabels(["offset 0", "offset 5"], color=INK)
    ax.set_zlabel("mean squared error", labelpad=12)
    ax.view_init(elev=24, azim=-58)
    legend_above(fig, [Patch(facecolor=BLUE, label="Offset 0"), Patch(facecolor=SAFFRON, label="Offset 5")])
    fig.suptitle("Synthetic seed 11  ·  the reference series", color=INK, fontsize=15, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.96, top=0.78, bottom=0.06)
    paths["synthetic"] = save_fig(fig, "synthetic_mse_3d.png")

    samples = one["burn_samples"]
    fig = plt.figure(figsize=(11.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    xs = np.arange(len(samples))
    bar3(ax, xs, 0, [s["power_w"] for s in samples], MAROON, depth=0.35, width=0.55)
    ticks = np.linspace(0, len(samples) - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{samples[i]['elapsed_s'] / 60:.1f}" for i in ticks])
    ax.set_xlabel("minutes", labelpad=14)
    ax.set_yticks([])
    ax.set_zlabel("watts", labelpad=12)
    ax.view_init(elev=25, azim=-52)
    fig.suptitle("One board  ·  five minutes on the tiled thermometer", color=INK, fontsize=15)
    fig.subplots_adjust(left=0.04, right=0.92, top=0.88, bottom=0.08)
    paths["one_power"] = save_fig(fig, "one_power_3d.png")

    fig = plt.figure(figsize=(11.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    bar3(ax, xs, 0, [s["temp_c"] for s in samples], GOLD, depth=0.35, width=0.55)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{samples[i]['elapsed_s'] / 60:.1f}" for i in ticks])
    ax.set_xlabel("minutes", labelpad=14)
    ax.set_yticks([])
    ax.set_zlabel("degrees C", labelpad=12)
    ax.view_init(elev=25, azim=-52)
    fig.suptitle("One board  ·  temperature during the real-series burn", color=INK, fontsize=15)
    fig.subplots_adjust(left=0.04, right=0.92, top=0.88, bottom=0.08)
    paths["one_temp"] = save_fig(fig, "one_temp_3d.png")

    fs = four["burn_samples"]
    fig = plt.figure(figsize=(11.4, 7.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    handles = []
    xf = np.arange(len(fs))
    for i in range(4):
        vals = [s["boards"][i]["power_w"] for s in fs]
        bar3(ax, xf, i, vals, palette[i], depth=0.4, width=0.5)
        handles.append(Patch(facecolor=palette[i], label=f"Board {i}"))
    step = max(1, len(fs) // 6)
    ax.set_xticks(xf[::step])
    ax.set_xticklabels([f"{fs[i]['elapsed_s'] / 60:.1f}" for i in range(0, len(fs), step)])
    ax.set_xlabel("minutes", labelpad=16)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("board", labelpad=14)
    ax.set_zlabel("watts", labelpad=12)
    ax.view_init(elev=24, azim=-58)
    legend_above(fig, handles)
    fig.suptitle("Four boards  ·  five minutes of the heavy job", color=INK, fontsize=15, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.94, top=0.78, bottom=0.08)
    paths["four_power"] = save_fig(fig, "four_power_3d.png")

    fig = plt.figure(figsize=(11.2, 7.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    ax.bar3d([0], [0], [0], 0.7, 0.45, [29.510], color=MAROON, shade=True)
    ax.bar3d([1.6], [0], [0], 0.7, 0.45, [8.283], color=FOREST, shade=True)
    ax.set_xticks([0.35, 1.95])
    ax.set_xticklabels(["four jobs\nqueued", "four jobs\nat once"], color=INK)
    ax.set_yticks([])
    ax.set_zlabel("milliseconds", labelpad=12)
    ax.view_init(elev=22, azim=-62)
    legend_above(fig, [Patch(facecolor=MAROON, label="Queued 29.51 ms"), Patch(facecolor=FOREST, label="Together 8.28 ms")])
    fig.suptitle("Four heavy jobs  ·  wall time", color=INK, fontsize=15, y=1.0)
    fig.subplots_adjust(top=0.78)
    paths["queue"] = save_fig(fig, "queue_3d.png")

    fig = plt.figure(figsize=(11.2, 7.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    roof = one["roofline"]
    for i, row in enumerate(roof):
        bar3(ax, [i], 0, [row["GBps_16byte"]], FOREST, depth=0.4, width=0.55)
    ax.set_xticks(np.arange(len(roof)) + 0.25)
    ax.set_xticklabels(["10,240", "1.05M", "10.5M", "50M"], color=INK)
    ax.set_yticks([])
    ax.set_zlabel("GB/s", labelpad=12)
    ax.view_init(elev=22, azim=-62)
    fig.suptitle("One board  ·  16 bytes per node  ·  peak 1,597 GB/s", color=INK, fontsize=15)
    fig.subplots_adjust(top=0.86, bottom=0.08)
    paths["roof"] = save_fig(fig, "roofline_3d.png")

    fig = plt.figure(figsize=(11.0, 6.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    uses = [("People", 672), ("Counsel", 160), ("Cloud", 144), ("Partners", 110), ("Bench", 50), ("Reserve", 114)]
    colors_u = [MAROON, FOREST, BLUE, SAFFRON, GOLD, "#6b3a22"]
    for i, ((label, amount), color) in enumerate(zip(uses, colors_u)):
        ax.bar3d([i], [0], [0], 0.6, 0.4, [amount], color=color, shade=True)
    ax.set_xticks(np.arange(len(uses)) + 0.3)
    ax.set_xticklabels([u[0] for u in uses], color=INK)
    ax.set_yticks([])
    ax.set_zlabel("thousand USD", labelpad=12)
    ax.view_init(elev=22, azim=-62)
    fig.suptitle("Working ask  ·  1.25 million USD over 18 months", color=INK, fontsize=15)
    fig.subplots_adjust(top=0.86, bottom=0.08)
    paths["funds"] = save_fig(fig, "funds_3d.png")
    return paths


def equation(name: str, line: str, caption: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 2.5), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.62, line, ha="center", va="center", fontsize=28, color=INK)
    ax.text(0.5, 0.22, caption, ha="center", va="center", fontsize=12, color=FOREST, fontstyle="italic")
    path = FIG / name
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def draw_equations() -> dict[str, Path]:
    return {
        "flux": equation("eq_flux.png", r"$A(y)=\mu+(y-\mu)\,e^{-\lambda\xi}$", "Flux damping. The center stays. Only the wiggle shrinks."),
        "raw": equation("eq_raw.png", r"$A(\theta)=\theta\,e^{-\lambda\xi}$", "Raw damping. The whole reading is shrunk, including a real temperature."),
        "closed": equation("eq_closed.png", r"$A_{40}=\mu+(y-\mu)\,e^{-40\xi}$", "Forty passes with a fixed center. The same-input residual was 0."),
    }


def flow_chain(name: str, title: str, nodes: list[tuple[str, str]]) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    n = len(nodes)
    fig, ax = plt.subplots(figsize=(2.7 * n, 3.7), facecolor="white")
    ax.set_xlim(0, n * 2.7)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    fills = [MAROON, FOREST, SAFFRON, BLUE]
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
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def flow_bounds(name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 4.5), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.25, 0.3), 5.5, 3.5, boxstyle="round,pad=0.04,rounding_size=0.1", facecolor=FOREST, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((6.25, 0.3), 5.5, 3.5, boxstyle="round,pad=0.04,rounding_size=0.1", facecolor=MAROON, edgecolor="none"))
    ax.text(3.0, 3.35, "Passed on 14 November", ha="center", color=CARD, fontsize=13, fontweight="bold")
    ax.text(9.0, 3.35, "Not claimed", ha="center", color=CARD, fontsize=13, fontweight="bold")
    left = "Ladder, gate, sign bit, derivative.\nFlux holds on a real thermometer.\nSame-input drift 0.\nFused kernel faster than eager.\nFour boards, about 4x the jobs."
    right = "A language-model exam.\nHalf the variance on this\nthermometer. The half bar\nwas missed, at about 51%.\nA silence gate for hallucinations."
    ax.text(3.0, 1.7, left, ha="center", va="center", color=CARD, fontsize=11, linespacing=1.4)
    ax.text(9.0, 1.7, right, ha="center", va="center", color=CARD, fontsize=11, linespacing=1.4)
    fig.suptitle("What the labelled exam can carry", color=INK, fontsize=14)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.22)
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


def pic(path: Path, max_w: float = 7.2 * inch, max_h: float = 5.2 * inch) -> Image:
    image = PILImage.open(path)
    wpx, hpx = image.size
    width = max_w
    height = width * (hpx / wpx)
    if height > max_h:
        height = max_h
        width = height * (wpx and hpx / wpx)
        width = height * (wpx / hpx)
    return Image(str(path), width=width, height=height)


def table(headers, rows, widths) -> Table:
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


def figure_block(title, path, caption):
    s = styles()
    return KeepTogether([CondPageBreak(5.5 * inch), Paragraph(title, s["H"]), pic(path), Paragraph(caption, s["Cap"])])


def how_story(figures, flows):
    s = styles()
    return [
        Paragraph("How the technology works", s["Cover"]),
        Paragraph("A plain-language brief for patent counsel. Ceraph Corporation, Kerala, India. 14 November 2026, labelled exam. chief@ceraphco.com · www.ceraphco.com", s["Sub"]),
        Paragraph("This brief explains an in-process way to keep the steady part of a numerical series and to quiet only the disturbance. The measurements below come from two notebooks run the same night on NVIDIA RTX PRO 6000 Blackwell Server Edition boards, one board and four boards. Each notebook answered a fixed exam whose answer key was written before the run, then spent five minutes under load. The one-board burn used a real thermometer series. The four-board burn used the same heavy random job as the earlier ten-minute record. No language model was downloaded, and no generation exam was scored.", s["Body"]),
        Paragraph("The idea, in one pass", s["H"]),
        pic(flows["path"], max_h=3.4 * inch),
        Paragraph("Read left to right. A noisy row arrives. A short neighborhood supplies a local center and a measure of how wild that neighborhood is. The map keeps the center and shrinks only the departure from it.", s["Cap"]),
        Paragraph("Three languages, one formula", s["H"]),
        pic(flows["languages"], max_h=3.4 * inch),
        Paragraph("Julia and the Python ladder both chose 128, 256, 512, and 1024 on the four labelled densities. Both returned the same gate answer at 0.05, 0.20, and 0.80. JAX matched the written derivative with residual 0. Triton matched the flux formula within 4.8x10^-7. Sign-bit negation residual was 0. This held on the one-board machine and on each of the four boards.", s["Cap"]),
        Paragraph("A real thermometer", s["H"]),
        pic(flows["thermo"], max_h=3.4 * inch),
        Paragraph("The series is the daily minimum temperature in Melbourne, 3,650 days, beginning 1 January 1981 at 20.7 C. The recording is the latent value. Sensor noise of 0.5 C and a 10 percent glitch were added, so the error has a known target. Seeds 11, 22, and 33 were fixed before the score.", s["Cap"]),
        Paragraph("The formula, written large", s["H"]),
        pic(figures["flux"]),
        Paragraph("Mu is the local center. The parenthesis is the wiggle. Xi is how wild the neighborhood is. Lambda was 1. On the thermometer, adding 5 C to the whole series left the flux error unchanged. The same shift made raw damping worse, because raw multiplies the temperature itself.", s["Cap"]),
        pic(figures["raw"]),
        Paragraph("A kitchen thermometer already sits far from zero. Raw damping pulled a reading near 11 C toward zero and produced an error near 135. After the extra 5 C, that error rose to about 266. Flux stayed near 5.82 to 5.87, just under the moving average, at both offsets.", s["Cap"]),
        Paragraph("Layman explainer", s["H"]),
        Paragraph("A song in a busy kitchen. The song is the steady thing. The clatter is the disturbance. A moving average turns both down. This method keeps the song at its own loudness and turns down only the clatter. On the thermometer, the song is the day's temperature. Flux kept it. Raw damping turned the temperature down along with the noise.", s["Body"]),
        Paragraph("A photograph on a shaky day. The person should stay. The shake should go. The five-minute run is that steadying, repeated 41,186 times on one board, on the same real series tiled out to 16,777,216 points, with every value remaining finite.", s["Body"]),
        Paragraph("Four musicians and a queue. Four copies of the heavy job, one after another on one board, took 29.51 milliseconds. The same four jobs at once, one per board, took 8.28 milliseconds. Over five minutes the one-board rate was 137.3 jobs a second and the four-board rate was 549.2, four times the work.", s["Body"]),
        Paragraph("The labelled exam", s["H"]),
        Paragraph("Eighteen questions on one board, seventeen on four. The key was in the cell before the score. One question missed on both machines: V1, which asked whether flux would keep at most half the variance of the contaminated thermometer. It kept 51.0, 51.8, and 51.0 percent on the three seeds. The miss is part of the record. A large-glitch screen, spikes of +8 standard deviations and a predeclared variance rule of 4, scored 24 of 24. That screen shows the rule can see a gross glitch. It is not a test of hallucination, and no language model was present.", s["Body"]),
        table(
            ["Question", "What was asked", "Result"],
            [
                ["B1 to B4", "Densities 0.30, 0.79, 1.50, 7.06 choose 128, 256, 512, 1024", "Passed. Julia and Python agreed."],
                ["G1 to G3", "Gate at 0.05, 0.20, 0.80", "Passed. Closed, open, open."],
                ["S1, J1, R0", "Sign bit, derivative, Triton versus the formula", "Passed. Residuals 0, 0, and 4.8x10^-7."],
                ["F0", "Synthetic seed 11, flux versus the average, at both offsets", "Passed. Flux 1.472. Raw 0.774, then 18.17."],
                ["F1, F2", "Melbourne, three seeds, flux holds at +5 C and beats raw", "Passed. Flux unchanged. Raw about 135, then about 266."],
                ["V1", "Flux keeps at most half the variance on Melbourne", "Missed. Remaining variance 0.510 to 0.518."],
                ["D1, C1", "Same input twice, and forty passes versus the closed form", "Passed. Drift 0. Closed form within 1x10^-6 on one board and 1.2x10^-5 on four."],
                ["T1, E1", "Ladder within 10 percent of the fastest block. Fused faster than eager.", "Passed. 256 was 3.6 percent behind 1024. Triton 7.38 ms, eager 17.13 ms."],
                ["Q1", "Four boards together at least twice as fast as the queue", "Passed. 8.28 ms together, 29.51 ms queued, 3.56 times."],
            ],
            [1.15 * inch, 3.15 * inch, 2.85 * inch],
        ),
        Spacer(1, 8),
        figure_block(
            "The thermometer, drawn as a 3D histogram",
            figures["melbourne"],
            "Seed 11. Front row is the recorded series plus noise. Back row is the same series shifted by 5 C. Raw is the tall bar and grows taller after the shift. Flux is the short bar in both rows, 5.819, a little under the moving average at 5.897. Lower is better. The drawing uses the one-board exam file.",
        ),
        figure_block(
            "The synthetic reference, for comparison",
            figures["synthetic"],
            "Seed 11, 8,192 points, the series used in the earlier sessions. Flux is 1.472 at both offsets. The moving average is 1.675. Raw is 0.774 near zero and 18.17 at offset 5. Variance remaining on this synthetic series is 11.7 percent, which does clear the half-variance bar. The thermometer did not.",
        ),
        Paragraph("Five minutes", s["H"]),
        Paragraph("Software on both machines: PyTorch 2.14.1 with CUDA 13.0, Triton 3.8.0, JAX 0.12.0.1, and Julia from juliaup. The boards are RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W limit, stated peak bandwidth 1,597 GB/s.", s["Body"]),
        table(
            ["Five-minute burn", "One board, tiled Melbourne", "Four boards, heavy random job"],
            [
                ["Length and passes", "16,777,216 points, 40 passes, block 1024", "16,777,216 points, 40 passes"],
                ["Jobs", "41,186 in 300.0 s", "164,748 in 300.0 s"],
                ["Jobs per second", "137.3", "549.2, which is 4.00 times one board"],
                ["Queue", "One job in the fused kernel, 7.38 ms", "29.51 ms queued, 8.28 ms together"],
                ["Mean power", "314.5 W", "279, 283, 279, and 277 W"],
                ["Peak power", "326.6 W", "300 to 305 W"],
                ["Temperature", "32 C to 44 C", "peaks 47 C and 48 C"],
                ["Clock", "2302 to 2317 MHz", "up to 2422 MHz on each board"],
                ["Occupancy", "98.6 percent mean", "94.8 to 99.9 percent"],
                ["Values finite", "yes", "yes, on every board"],
                ["Same-input drift", "0", "0 on board 0"],
            ],
            [1.55 * inch, 2.8 * inch, 2.8 * inch],
        ),
        Spacer(1, 8),
        figure_block(
            "One board, five minutes of power on the real series",
            figures["one_power"],
            "Twenty samples, about fifteen seconds apart. The opening sample is 111 W. The draw then sits near 325 W. The mean across the file is 314.5 W and the peak is 326.6 W, against a 600 W limit. Block 1024 was chosen because the mean absolute instability of the tiled thermometer sat in the top rung of the ladder.",
        ),
        figure_block(
            "One board, the heat",
            figures["one_temp"],
            "32 C at the start, a plateau at 42 to 44 C. The clock stayed between 2302 and 2317 MHz. Occupancy after the opening minute was 99 percent. Every stored value stayed finite.",
        ),
        figure_block(
            "Four boards, five minutes",
            figures["four_power"],
            "This burn is the heavy random job, not the thermometer. Each color is one board. Means were 279, 283, 279, and 277 W. Peaks were 300 to 305 W. Temperatures peaked at 47 and 48 C. Clocks reached 2422 MHz. The cluster finished 164,748 jobs.",
        ),
        figure_block(
            "The queue, drawn large",
            figures["queue"],
            "Maroon is four heavy jobs one after another: 29.51 ms. Forest is the same four jobs at once: 8.28 ms. The ratio is 3.56. The five-minute job rate, 549 against 137, is the four-times figure.",
        ),
        figure_block(
            "How full the memory pipe was",
            figures["roof"],
            "One-board roofline, counting three loads and one store. At 50 million nodes the median was 0.548 ms and 1,459 GB/s, 91.4 percent of 1,597. The four-board machine, measured on board 0, reached 1,457 GB/s at the same length. Short lengths stay near a hundredth of a millisecond because the launch dominates.",
        ),
        Paragraph("What counsel can say", s["H"]),
        pic(flows["bounds"], max_h=3.6 * inch),
        Paragraph("Counsel can describe the flux map, the density ladder, the gate predicate, the sign-bit negation, the agreement of Julia, JAX, and Triton, the thermometer result, the same-input drift of zero, the fused kernel against the eager loop, and the four-board rate. Counsel should also say that half the variance was not achieved on this thermometer, that the glitch screen used declared spikes of eight standard deviations, and that no language-model exam was run.", s["Cap"]),
        Paragraph("A sentence for the specification. The invention is an in-process map that preserves a local center and damps only the deviation from that center. On 14 November 2026 a fixed exam of that map passed 17 of 18 questions on one Blackwell board and 16 of 17 on four boards. The missed question was the half-variance bar on a real daily-temperature series, where about 51 percent of the variance remained. On that same series the flux error was unchanged when the readings were shifted by 5 C, and a method that multiplies the whole reading was not.", s["Body"]),
        Paragraph("Absolute remains untouched, only the error is taken away.", s["Cover"]),
        Paragraph("Ceraph Corporation · Kerala, India · chief@ceraphco.com · www.ceraphco.com", s["Sub"]),
    ]


def tech_story(figures, flows):
    s = styles()
    return [
        Paragraph("Technical record and investor memorandum", s["Cover"]),
        Paragraph("Ceraph Corporation, Kerala, India. 14 November 2026, labelled exam. chief@ceraphco.com · www.ceraphco.com. For the company file and for a first conversation with investors. Planning figures are estimates. They are not a forecast of revenue and they are not an offer of securities.", s["Sub"]),
        Paragraph("Abstract", s["H"]),
        Paragraph("This memorandum records a labelled exam of the flux form of Abhaya damping, run the same night on one NVIDIA RTX PRO 6000 Blackwell Server Edition and on four of them. The answer key was fixed in the notebook before the score. Julia and a Python twin chose the launch block. JAX checked an analytical derivative. Triton executed the map. A sign-bit exclusive-or negated float32 values. The real series was the daily minimum temperature in Melbourne, 3,650 days from 1 January 1981. No language model was present.", s["Body"]),
        Paragraph("Seventeen of eighteen questions passed on one board. Sixteen of seventeen passed on four boards. The miss, on both machines, was the pre-existing claim that flux would retain at most half the variance. On the thermometer the remaining fraction was 0.510, 0.518, and 0.510 for seeds 11, 22, and 33. Flux error stayed at 5.819, 5.874, and 5.820 when the series was shifted by 5 C, under the moving average, while raw damping rose from about 135 to about 266. Same-input drift was 0. A fused forty-pass kernel on the tiled thermometer took 7.38 ms. The eager loop took 17.13 ms. Four boards finished 164,748 heavy jobs in 300 seconds, 549.2 per second, four times the one-board rate of 137.3. At 50 million nodes the corrected traffic count was 1,459 GB/s on the one-board machine and 1,457 GB/s on board 0 of the four-board machine, about 91 percent of the stated 1,597 GB/s peak.", s["Body"]),
        Paragraph("The company", s["H"]),
        Paragraph("Ceraph Corporation is based in Kerala, India. Correspondence goes to chief@ceraphco.com. The public site is www.ceraphco.com. The company is at the stage of a provisional filing and a first institutional conversation. It does not have design-partner revenue as of this date. This night's exam does not create that revenue. It adds a real series, a fixed score, and an honest miss to the record a conversation can show.", s["Body"]),
        Paragraph("Mathematical framework", s["H"]),
        Paragraph("Flux damping keeps a local center and multiplies only the deviation by a shrinking exponential of the local instability. Raw damping multiplies the whole observation. Lambda was 1 in every test below. Forty iterations with a fixed center collapse to one exponential.", s["Body"]),
        pic(figures["flux"]),
        Paragraph("On a thermometer the center is the local temperature. Shrinking it would report a cooler day than the one that happened. That is what raw damping did. Flux did not.", s["Cap"]),
        pic(figures["raw"]),
        Paragraph("The closed form is the check that forty kernel passes did the algebra they were asked to do. On one board the maximum absolute difference was 9.5x10^-7. On board 0 of the four-board machine it was 1.2x10^-5, inside the exam bar of 1x10^-4.", s["Cap"]),
        pic(figures["closed"]),
        Paragraph("The ladder reads the mean absolute instability. Below 0.5 the block is 128. Below 1 it is 256. Below 2 it is 512. Otherwise it is 1024. The gate predicate is true when the stability index is at least 0.20. Below that horizon the predicate is false and the host is specified not to launch. Both rules were scored against labels, and Julia matched Python on every one.", s["Body"]),
        Paragraph("The labelled exam", s["H"]),
        pic(flows["languages"], max_h=3.4 * inch),
        Paragraph("The stack was PyTorch 2.14.1 with CUDA 13.0, Triton 3.8.0, JAX 0.12.0.1 on cuda, and Julia. The one-board machine reported cuda:0. The four-board machine reported cuda:0 through cuda:3, each an RTX PRO 6000 Blackwell Server Edition.", s["Cap"]),
        table(
            ["Check", "One board", "Four boards"],
            [
                ["Exam score", "17 of 18", "16 of 17"],
                ["Miss", "V1, variance remaining 0.510 to 0.518", "the same three numbers"],
                ["Block ladder", "Julia and Python, four densities", "the same"],
                ["Gate", "0.05 closed, 0.20 and 0.80 open", "the same"],
                ["Sign bit and derivative", "0 and 0", "0 and 0 on each board"],
                ["Triton versus flux", "4.8x10^-7", "4.8x10^-7 on each board"],
                ["Synthetic flux, both offsets", "1.472", "1.472"],
                ["Synthetic raw at offset 5", "18.17", "18.17"],
                ["Melbourne flux, seeds 11, 22, 33", "5.819, 5.874, 5.820", "the same"],
                ["Melbourne flux after +5 C", "unchanged on each seed", "unchanged"],
                ["Melbourne raw, then +5 C", "about 135, then about 266", "the same"],
                ["Same-input drift", "0", "0"],
                ["Closed form", "9.5x10^-7", "1.2x10^-5"],
                ["Eager versus fused, 40 passes", "17.13 ms and 7.38 ms", "queue 29.51 ms, together 8.28 ms"],
                ["Ladder versus fastest block", "256 was 3.6 percent behind 1024, inside 10 percent", "not re-swept"],
                ["50 million nodes", "0.548 ms, 1,459 GB/s, 91.4 percent", "0.549 ms, 1,457 GB/s, 91.2 percent"],
            ],
            [1.7 * inch, 2.7 * inch, 2.75 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The block sweep on one million random points gave 0.0157 ms at 128, 0.0120 ms at 256, 0.0131 ms at 512, and 0.0116 ms at 1024. The ladder chose 256 because the mean instability of that draw was below 1. The question asked only whether the choice was within 10 percent of the fastest. It was. It did not ask the ladder to win.", s["Body"]),
        Paragraph("The large-glitch screen used 24 clean windows and 24 copies with +8 standard deviations written onto 15 percent of each window. The rule, fixed in advance, called a window a glitch when its variance exceeded 4. All 24 glitches fired and none of the clean windows did, so the F1 was 1.0. The spikes were gross by construction. This is evidence that the predicate can see a declared glitch. It is not evidence about hallucinations, and the generation exam in the framework remains unrun.", s["Body"]),
        figure_block("Thermometer histogram", figures["melbourne"], "Seed 11 of the real series. Flux is short in both rows. Raw is tall, and taller after the shift. The three seeds agreed to the third significant figure."),
        figure_block("Synthetic histogram", figures["synthetic"], "The seed-11 series from the earlier sessions, reproduced in this exam. Flux 1.472. Variance remaining 11.7 percent. This is the series on which the half-variance bar is met. The thermometer is the series on which it is not."),
        figure_block("Real-series endurance", figures["one_power"], "41,186 jobs in 300.0 seconds, 137.3 per second, on the tiled thermometer. Mean draw 314.5 W, peak 326.6 W, about half the 600 W limit still unused. Energy was about 2,291 mJ per job."),
        figure_block("Real-series temperature", figures["one_temp"], "A rise from 32 C to a plateau of 42 to 44 C. Clock 2302 to 2317 MHz. The earlier random-series burn in the same notebook, block 256, finished 41,097 jobs at 137.0 per second, mean 288.6 W, 30 C to 42 C. The real series chose block 1024 and drew more power. Both stayed finite."),
        figure_block("Four-board endurance", figures["four_power"], "164,748 jobs, 549.2 per second. Cluster mean about 1,119 W. Per board, means 279 to 283 W and peaks 300 to 305 W. Peaks of temperature 47 and 48 C. Clocks to 2422 MHz. Energy about 2,037 mJ per job. This burn is the random heavy job. The thermometer on this machine is the exam table, not this picture."),
        figure_block("Queue histogram", figures["queue"], "29.51 ms against 8.28 ms. The buffers were already on the devices before the timer started, so the ratio is kernel time. 3.56 times on one wave, and 4.00 times on the five-minute rate."),
        figure_block("Roofline", figures["roof"], "16 bytes per node. 1,459 GB/s at 50 million nodes on the one-board machine. The twelve-byte count stored beside it is 1,094 GB/s. The sixteen-byte count is the one that includes the write."),
        Paragraph("Why the miss belongs in the memorandum", s["H"]),
        Paragraph("A raise that hides the question it failed will not survive a careful reader. V1 was written into the cell as a claim the company already carried: flux should keep at most half the variance. On the synthetic seed it does, at 11.7 percent remaining. On the Melbourne series, with half-degree noise and ten percent glitches, it kept about 51 percent. The error still fell below the moving average, and it did not move when the temperature was shifted. The invention's deployable claim on this night is the preservation of the center. The stronger variance claim is met on the synthetic series and not yet met on this thermometer. Both sentences can sit in a specification.", s["Body"]),
        Paragraph("What a raise is for", s["H"]),
        Paragraph("Two bands are enough for a first conversation. Salaries are loaded roughly and stated in US dollars. The company is in Kerala. These figures are a budget a conversation can edit. They are not a local payroll quotation.", s["Body"]),
        Paragraph("Working ask: 1.25 million US dollars for 18 months.", s["Body"]),
        table(
            ["Use", "Basis", "Amount"],
            [
                ["Founder", "18 months at 10,000 a month", "$180,000"],
                ["Kernel engineer", "18 months, loaded near 16,000 a month", "$288,000"],
                ["Research scientist", "12 months, loaded near 14,000 a month", "$168,000"],
                ["Finance and operations, fractional", "2,000 a month", "$36,000"],
                ["Counsel and the utility path", "Beyond the provisional", "$120,000"],
                ["Entity, insurance, accounting", "18 months", "$40,000"],
                ["Blackwell and other cloud", "Planned average 8,000 a month", "$144,000"],
                ["Bench held for later embodiments", "Not spent as if those devices exist", "$30,000"],
                ["Papers, site, two gatherings, travel", "A small public face", "$70,000"],
                ["Design-partner program", "Our cost to show up", "$40,000"],
                ["Recruiting", "Two hires", "$20,000"],
                ["Reserve", "About 9 percent", "$114,000"],
                ["Working ask", "18 months", "$1,250,000"],
            ],
            [2.4 * inch, 3.15 * inch, 1.6 * inch],
        ),
        Spacer(1, 8),
        figure_block("Where the working ask goes", figures["funds"], "People 672. Counsel and the entity 160. Cloud 144. The public face and the partner program 110. Bench and recruiting 50. Reserve 114. The sum is 1.25 million."),
        Paragraph("Lean ask: 700,000 US dollars for 12 months. Founder 120,000. Engineer 192,000. Scientist for 6 months 84,000. Counsel 80,000. Cloud at 6,000 a month 72,000. Public face and partner program 50,000. Operations 36,000. Reserve 66,000.", s["Body"]),
        Paragraph("Projected income over three years", s["H"]),
        Paragraph("One assumption comes before the numbers. A design partner who already runs a graphics pipeline pays on the order of 60,000 US dollars a year to put this map inside one stream, once a small library exists and the provisional is on file. None of this income exists today. The table is a planning band, not a booking.", s["Body"]),
        table(
            ["Year after the raise", "Quiet", "Working", "One wedge takes"],
            [
                ["Year 1", "No paid partner. Income about 0.", "Two paid pilots, about $120,000.", "The same as the working case."],
                ["Year 2", "One retainer, about $60,000.", "Four retainers, about $400,000.", "Six to eight, about $0.8 million to $1.2 million."],
                ["Year 3", "Two retainers, about $150,000.", "Eight retainers, about $1.2 million.", "One vertical. Planning band $2.5 million to $4 million."],
            ],
            [1.4 * inch, 1.9 * inch, 1.9 * inch, 1.95 * inch],
        ),
        Spacer(1, 8),
        Paragraph("Year 2 at about 400,000 does not cover a 1.25 million burn. The raise is the bridge. Year 3 in the working column is the first year a services practice of this shape could approach covering an ongoing team. A consumer product, a multi-tenant cloud, and a claim of thermodynamic advantage sit outside all three columns.", s["Body"]),
        Paragraph("Risks", s["H"]),
        Paragraph("The thermometer win is on a public daily series with noise we added, not inside a customer's sensor. The half-variance bar missed on that series. The glitch screen was easy by design. No language model was examined. An incumbent can ship a blur faster than a new company can ship a library. The advantage shown here is the placement, in the same pass, and the preservation of the center when the reading is shifted. Deep-tech schedules slip. The next money is for a library and for one partner's own series, not for another copy of this night.", s["Body"]),
        Paragraph("Conclusion", s["H"]),
        Paragraph("On 14 November 2026 the labelled exam was scored on Blackwell, on one board and on four. The map held a real temperature when a multiplying damper did not. The same input repeated with drift zero. The fused kernel was faster than the eager loop. Four boards did four times the jobs, near half the power limit, with temperatures under 50 C. One predeclared question missed, and the memorandum keeps the miss. The raise of 1.25 million dollars, or 700,000 in the lean band, buys the people and the months to turn the record into one library and one paid wedge, and to carry the provisional through to a utility filing.", s["Body"]),
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
    run = section.header.paragraphs[0].add_run("Ceraph Corporation  ·  Kerala, India  ·  chief@ceraphco.com  ·  www.ceraphco.com")
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
            for item in paragraph.runs:
                item.font.color.rgb = RGBColor(0x7C, 0x2D, 0x2D)
                item.font.name = "Times New Roman"
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
                        for item in paragraph.runs:
                            item.font.size = Pt(8)
                            item.font.name = "Times New Roman"
                            if r == 0:
                                item.bold = True
                                item.font.color.rgb = RGBColor(0xFB, 0xF6, 0xEC)
                    if r == 0:
                        shade_cell(grid.rows[r].cells[c], "7C2D2D")
                    else:
                        shade_cell(grid.rows[r].cells[c], "FFFFFF")
            document.add_paragraph("")
        elif kind == "cap":
            paragraph = document.add_paragraph(payload)
            for item in paragraph.runs:
                item.italic = True
                item.font.color.rgb = RGBColor(0x2C, 0x4A, 0x3E)
                item.font.size = Pt(10)
    fr = section.footer.paragraphs[0].add_run("Ceraph Corporation  ·  Kerala, India  ·  14 November 2026")
    fr.font.name = "Times New Roman"
    fr.font.size = Pt(9)
    fr.italic = True
    fr.font.color.rgb = RGBColor(0x2C, 0x4A, 0x3E)
    document.save(path)


def docx_from_story(path: Path, items: list) -> None:
    blocks = []
    for item in items:
        if isinstance(item, Paragraph):
            text = item.getPlainText()
            style = item.style.name if item.style is not None else ""
            blocks.append(("h" if style in {"Cover", "H"} else "cap" if style in {"Cap", "Sub"} else "p", text))
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
                    blocks.append(("h" if style == "H" else "cap" if style == "Cap" else "p", inner.getPlainText()))
                elif isinstance(inner, Image):
                    blocks.append(("img", inner.filename))
    add_docx(path, blocks)


def build() -> None:
    figures = draw_figures()
    figures.update(draw_equations())
    flows = {
        "path": flow_chain("flow_path.png", "Keep the center, quiet the wiggle", [
            ("Series", "A noisy row arrives. A thermometer is one such row."),
            ("Neighborhood", "A short window gives a center and a wildness."),
            ("Flux map", "Only the departure from the center is shrunk."),
            ("Reading", "The temperature remains. The glitch is quieter."),
        ]),
        "languages": flow_chain("flow_languages.png", "Julia, JAX, and Triton", [
            ("Density", "Mean absolute size of the disturbance."),
            ("Julia", "The ladder chooses 128, 256, 512, or 1024."),
            ("JAX", "The map and its derivative, on the graphics chip."),
            ("Triton", "The fused kernel runs at that block."),
        ]),
        "thermo": flow_chain("flow_thermo.png", "A real daily temperature", [
            ("Recording", "Melbourne, 3,650 days, from 20.7 C."),
            ("Noise", "0.5 C of noise and a 10 percent glitch."),
            ("Score", "Flux, raw, and a moving average, seeds 11, 22, 33."),
            ("Shift", "Add 5 C. Flux holds. Raw does not."),
        ]),
        "bounds": flow_bounds("flow_bounds.png"),
    }
    how_pdf = OUT / "nov14-counsel-exam.pdf"
    tech_pdf = OUT / "nov14-investor-exam.pdf"
    how_doc = SimpleDocTemplate(str(how_pdf), pagesize=letter, leftMargin=0.68 * inch, rightMargin=0.68 * inch, topMargin=0.72 * inch, bottomMargin=0.62 * inch, title="How the technology works", author="Ceraph Corporation")
    tech_doc = SimpleDocTemplate(str(tech_pdf), pagesize=letter, leftMargin=0.68 * inch, rightMargin=0.68 * inch, topMargin=0.72 * inch, bottomMargin=0.62 * inch, title="Technical record and investor memorandum", author="Ceraph Corporation")
    how_doc.build(how_story(figures, flows), onFirstPage=lambda c, d: paint(c, d, FOOTER_C), onLaterPages=lambda c, d: paint(c, d, FOOTER_C))
    tech_doc.build(tech_story(figures, flows), onFirstPage=lambda c, d: paint(c, d, FOOTER_I), onLaterPages=lambda c, d: paint(c, d, FOOTER_I))
    docx_from_story(OUT / "nov14-counsel-exam.docx", how_story(figures, flows))
    docx_from_story(OUT / "nov14-investor-exam.docx", tech_story(figures, flows))
    print({"pdf": [str(how_pdf), str(tech_pdf)]})


if __name__ == "__main__":
    build()
