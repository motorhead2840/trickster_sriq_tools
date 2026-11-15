"""Detailed L40S evidence pack with 3D figures from the 14 Nov 2026 notebook."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
NAVY = colors.HexColor("#1c2430")
RULE = colors.HexColor("#c5cdd8")
ROW = colors.HexColor("#f4f6f8")

CPU = [
    (4096, 0.01428),
    (16384, 0.03384),
    (65536, 0.12465),
    (262144, 0.47779),
    (1048576, 1.88534),
]
GPU = {4096: 0.04096, 65536: 0.04096, 1048576: 0.04608}
IDLE_W, LOAD_W = 38.2735, 84.719
IDLE_C, LOAD_C = 30.5, 36.0
LIMIT_W = 350.0


def _style():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cover", fontName="Times-Bold", fontSize=16, leading=20, textColor=NAVY, spaceAfter=6))
    base.add(ParagraphStyle("Sub", fontName="Times-Italic", fontSize=10, leading=13, textColor=colors.HexColor("#3d4754"), spaceAfter=8))
    base.add(ParagraphStyle("H", fontName="Times-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=4))
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=10, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=7))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8, leading=10.4, textColor=NAVY))
    base.add(ParagraphStyle("Head", fontName="Times-Bold", fontSize=8, leading=10.4, textColor=colors.white))
    return base


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#5c6773"))
    canvas.drawString(0.7 * inch, 0.38 * inch, "Ceraph Corporation  ·  L40S evidence pack  ·  14 Nov 2026")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.38 * inch, str(doc.page))
    canvas.restoreState()


def _table(headers, rows, widths):
    s = _style()
    data = [[Paragraph(h, s["Head"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["Cell"]) for c in row])
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def make_figures() -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    lengths = [n for n, _ in CPU if n in GPU]
    fig = plt.figure(figsize=(10.2, 6.2), facecolor="#10141c")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#10141c")
    xpos = np.arange(len(lengths))
    cpu_h = np.array([dict(CPU)[n] for n in lengths])
    gpu_h = np.array([GPU[n] for n in lengths])
    ax.bar3d(xpos - 0.15, np.zeros(len(lengths)), np.zeros(len(lengths)), 0.3, 0.45, cpu_h, color="#e09a55", shade=True, label="CPU")
    ax.bar3d(xpos + 0.15, np.ones(len(lengths)), np.zeros(len(lengths)), 0.3, 0.45, gpu_h, color="#3ecf8e", shade=True)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{n // 1024}k" for n in lengths], color="#e8edf5")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["CPU", "L40S"], color="#e8edf5")
    ax.set_zlabel("Median ms", color="#e8edf5")
    ax.tick_params(colors="#e8edf5")
    ax.view_init(elev=24, azim=-58)
    fig.suptitle("Same-host latency by length", color="#f4f7fb")
    latency = FIG / "latency_3d.png"
    fig.savefig(latency, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig = plt.figure(figsize=(8.4, 5.6), facecolor="#10141c")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#10141c")
    labels = ["Idle power", "Loaded power", "Idle temp", "Loaded temp"]
    heights = [IDLE_W, LOAD_W, IDLE_C, LOAD_C]
    colors_b = ["#6ec8ff", "#ff6b6b", "#7d8cff", "#ffb703"]
    ax.bar3d(np.arange(4), np.zeros(4), np.zeros(4), 0.55, 0.55, heights, color=colors_b, shade=True)
    ax.set_xticks(np.arange(4) + 0.25)
    ax.set_xticklabels(labels, color="#e8edf5", rotation=15)
    ax.set_zlabel("W or °C", color="#e8edf5")
    ax.tick_params(colors="#e8edf5")
    ax.view_init(elev=22, azim=-62)
    fig.suptitle("Idle versus loaded board readings", color="#f4f7fb")
    power = FIG / "power_temp_3d.png"
    fig.savefig(power, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"latency": latency, "power": power}


def detailed(path: Path, figures: dict[str, Path]) -> None:
    s = _style()
    ratio = 1.88534 / 0.04608
    story = [
        Paragraph("L40S evidence pack, 14 November 2026", s["Cover"]),
        Paragraph("Telemetry from one NVIDIA L40S on DGX Cloud. Synthetic map-and-reduce and a synthetic filter. Not a facility audit and not the unpublished method.", s["Sub"]),
        Paragraph("1. Device", s["H"]),
        Paragraph("The card identified itself as an NVIDIA L40S, driver 565.57.01, 46,068 MiB, power limit 350 W. PyTorch 2.14.1 was built for CUDA 12.6, matching that driver. A first probe read 32.40 W and 30 °C. Twenty idle samples averaged 38.27 W, with temperature between 30 °C and 31 °C.", s["Body"]),
        Paragraph("2. Latency", s["H"]),
        Paragraph(f"CPU median time rose from 0.014 ms at 4,096 elements to 1.885 ms at 1,048,576. The log-log slope was 0.895. The L40S median was 0.041 ms at 4,096 and at 65,536, and 0.046 ms at 1,048,576. At the largest length the GPU was {ratio:.0f} times quicker than the CPU in the same machine. At 4,096 elements the CPU was quicker, because the GPU launch costs more than that small amount of arithmetic.", s["Body"]),
        Image(str(figures["latency"]), width=6.6 * inch, height=4.0 * inch),
        Spacer(1, 6),
        _table(
            ["Length", "CPU ms", "L40S ms", "Quicker device"],
            [
                ["4,096", "0.014", "0.041", "CPU"],
                ["16,384", "0.034", "not timed", "CPU only"],
                ["65,536", "0.125", "0.041", "L40S"],
                ["262,144", "0.478", "not timed", "CPU only"],
                ["1,048,576", "1.885", "0.046", "L40S, about 41×"],
            ],
            [1.5 * inch, 1.4 * inch, 1.5 * inch, 2.2 * inch],
        ),
        Paragraph("3. Power and heat during the large GPU loop", s["H"]),
        Paragraph("Thirty launches of the 1,048,576-element GPU kernel, with a driver query between launches, averaged 84.72 W and reached 36 °C. That is 2.21 times the 38.27 W idle average, and 24 percent of the 350 W limit. The temperature rose about 5 degrees from the idle band. The driver query is slow beside the kernel, so 84.72 W is the board during that mixed loop, not joules inside the 0.046 ms kernel alone.", s["Body"]),
        Image(str(figures["power"]), width=6.2 * inch, height=4.1 * inch),
        Spacer(1, 6),
        Paragraph("4. Synthetic error", s["H"]),
        Paragraph("On 80 synthetic series of length 128, a moving average of width 9 had mean squared error 0.0324. A median filter of the same width had 0.0469. The average was more accurate. This cell does not run the unpublished method.", s["Body"]),
        Paragraph("5. What a customer can be told", s["H"]),
        Paragraph("For this synthetic job on this machine, a long vector finished much sooner on the L40S than on the CPU, and the board stayed near 85 W and 36 °C rather than near its 350 W cap. A short vector was faster on the CPU. That is the efficiency statement the session supports. It does not price a data-center bill, it does not rank the L40S against the 7 November RTX PRO 6000 Blackwell kernel, and it does not show that every Ceraph workload is constant-time. The rental on the order screen was $1.74 per hour. The cells took minutes.", s["Body"]),
        Paragraph("A heavier kernel that approaches the power limit, and the same script on a second card, are the measurements that would extend this note from one host to a purchase comparison.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="L40S evidence pack", author="Ceraph Corporation")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def brief(path: Path, figures: dict[str, Path]) -> None:
    s = _style()
    story = [
        Paragraph("One graphics card, one afternoon", s["Cover"]),
        Paragraph("Ceraph Corporation  ·  14 November 2026  ·  Plain language", s["Sub"]),
        Paragraph("We rented one NVIDIA L40S. While it waited it used about 38 watts and stayed near 30 °C. While it repeated a large made-up arithmetic job it used about 85 watts and reached 36 °C. Its maximum is 350 watts.", s["Body"]),
        Paragraph("At about one million numbers, the card finished in 0.046 thousandths of a second. The ordinary processor in the same computer took 1.9 thousandths of a second, about 41 times longer. At only a few thousand numbers, the ordinary processor was faster, because starting the card costs more than the tiny job.", s["Body"]),
        Image(str(figures["latency"]), width=6.4 * inch, height=3.9 * inch),
        Spacer(1, 6),
        Paragraph("On a separate made-up noisy series, a simple average was more accurate than a median filter. That test is not the private method. This note shows a faster large job on this card at modest power. It does not yet show a data-center bill or a win over a different graphics card.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="One graphics card, one afternoon", author="Ceraph Corporation")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    figures = make_figures()
    detailed(OUT / "l40s-evidence-detailed.pdf", figures)
    brief(OUT / "l40s-evidence-brief.pdf", figures)
    print("wrote", OUT / "l40s-evidence-detailed.pdf")


if __name__ == "__main__":
    main()
