"""PDFs for the 14 Nov 2026 RTX PRO 6000 Blackwell kernel timing."""
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

ROWS = [
    (16384, 1.061457, 0.149632, 7.094),
    (65536, 8.450132, 0.152480, 55.418),
    (262144, 35.982704, 0.241824, 148.797),
    (1048576, 168.046119, 1.415104, 118.752),
]
LOAD_W, LOAD_C = 104.9785, 30.0


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cover", fontName="Times-Bold", fontSize=16, leading=20, textColor=NAVY, spaceAfter=6))
    base.add(ParagraphStyle("Sub", fontName="Times-Italic", fontSize=10, leading=13, textColor=colors.HexColor("#3d4754"), spaceAfter=8))
    base.add(ParagraphStyle("H", fontName="Times-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=4))
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=10, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=7))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8, leading=10.4, textColor=NAVY))
    base.add(ParagraphStyle("Head", fontName="Times-Bold", fontSize=8, leading=10.4, textColor=colors.white))
    return base


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#5c6773"))
    canvas.drawString(0.7 * inch, 0.38 * inch, "Ceraph Corporation  ·  RTX PRO 6000 Blackwell session  ·  14 Nov 2026")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.38 * inch, str(doc.page))
    canvas.restoreState()


def table(headers, rows, widths):
    s = styles()
    data = [[Paragraph(h, s["Head"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["Cell"]) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def figures() -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = [str(n) for n, *_ in ROWS]
    xs = np.arange(len(ROWS))
    cpu = np.array([r[1] for r in ROWS])
    gpu = np.array([r[2] for r in ROWS])
    speed = np.array([r[3] for r in ROWS])
    fig = plt.figure(figsize=(10.2, 6.2), facecolor="#10141c")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#10141c")
    ax.bar3d(xs - 0.15, np.zeros(len(xs)), np.zeros(len(xs)), 0.28, 0.4, cpu, color="#e09a55", shade=True)
    ax.bar3d(xs + 0.15, np.ones(len(xs)), np.zeros(len(xs)), 0.28, 0.4, gpu, color="#3ecf8e", shade=True)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color="#e8edf5", rotation=15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["CPU", "Blackwell"], color="#e8edf5")
    ax.set_zlabel("Median ms", color="#e8edf5")
    ax.tick_params(colors="#e8edf5")
    ax.view_init(elev=24, azim=-58)
    fig.suptitle("RTX PRO 6000 Blackwell vs same-host CPU", color="#f4f7fb")
    latency = FIG / "blackwell_latency_3d.png"
    fig.savefig(latency, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig = plt.figure(figsize=(9.2, 5.8), facecolor="#10141c")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#10141c")
    ax.bar3d(xs, np.zeros(len(xs)), np.zeros(len(xs)), 0.45, 0.45, speed, color=["#7d8cff", "#6ec8ff", "#3ecf8e", "#ffb703"], shade=True)
    ax.set_xticks(xs + 0.2)
    ax.set_xticklabels(labels, color="#e8edf5", rotation=15)
    ax.set_zlabel("CPU time / GPU time", color="#e8edf5")
    ax.tick_params(colors="#e8edf5")
    ax.view_init(elev=22, azim=-62)
    fig.suptitle("How many times sooner the GPU finished", color="#f4f7fb")
    ratio = FIG / "blackwell_speedup_3d.png"
    fig.savefig(ratio, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"latency": latency, "speed": ratio}


def detailed(path: Path, pics: dict[str, Path]) -> None:
    s = styles()
    story = [
        Paragraph("RTX PRO 6000 Blackwell kernel session, 14 November 2026", s["Cover"]),
        Paragraph("Same code on the GPU and the CPU of one DGX Cloud machine. PyTorch 2.14.1+cu130. Not a second-card comparison and not a facility bill.", s["Sub"]),
        Paragraph("Device", s["H"]),
        Paragraph("The GPU was an NVIDIA RTX PRO 6000 Blackwell Server Edition. During forty repeats of the one-million-element kernel, with a driver query between launches, the board averaged 104.98 W and the reported temperature stayed at 30 °C. The query is slow next to the kernel, so this watt reading belongs to that mixed loop.", s["Body"]),
        Paragraph("Timing", s["H"]),
        Paragraph("At every length the GPU finished sooner than the CPU. The ratio rose from 7.1 times at 16,384 elements to 149 times at 262,144, then eased to 119 times at 1,048,576 because the GPU time itself rose from 0.242 ms to 1.415 ms. CPU time kept growing, from 1.06 ms to 168 ms. The GPU is faster across this range. It is not constant-time: a 64-fold increase in length, from 16,384 to 1,048,576, multiplied GPU time by about 9.5.", s["Body"]),
        Image(str(pics["latency"]), width=6.5 * inch, height=3.9 * inch),
        Spacer(1, 6),
        table(
            ["Length", "CPU ms", "GPU ms", "CPU / GPU"],
            [[f"{n:,}", f"{c:.3f}", f"{g:.3f}", f"{r:.1f}×"] for n, c, g, r in ROWS],
            [1.6 * inch, 1.5 * inch, 1.5 * inch, 1.8 * inch],
        ),
        Spacer(1, 8),
        Image(str(pics["speed"]), width=6.2 * inch, height=3.9 * inch),
        Paragraph("What can be said", s["H"]),
        Paragraph("On this machine, this kernel finished 7 to 149 times sooner on the Blackwell GPU than on the CPU, while the board sat near 105 W and 30 °C. That is the supported speed and power statement. It does not meter the CPU, so it does not close a joule comparison. It does not repeat the 7 November JAX session, which timed a related kernel once at length 16,384 with a median of 0.08884 ms and no power reading. This PyTorch run at that length was 0.150 ms. Different software, so the two numbers are not a ranking. It also does not show a data-center price, a second accelerator, or a result for any workload other than this kernel.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="RTX PRO 6000 Blackwell kernel session", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def brief(path: Path, pics: dict[str, Path]) -> None:
    s = styles()
    story = [
        Paragraph("Blackwell card versus the processor beside it", s["Cover"]),
        Paragraph("Ceraph Corporation  ·  14 November 2026  ·  Plain language", s["Sub"]),
        Paragraph("We ran the same numerical kernel on an NVIDIA RTX PRO 6000 Blackwell and on the ordinary processor in that computer. At every size we tried, the graphics card finished first. At about a quarter million numbers it was about 149 times sooner. At about one million numbers it was about 119 times sooner. The card used about 105 watts during that large loop, and the reported temperature stayed at 30 °C.", s["Body"]),
        Image(str(pics["speed"]), width=6.3 * inch, height=4.0 * inch),
        Spacer(1, 6),
        Paragraph("The card did get slower as the job grew. It was not instant at every size. We did not measure the processor’s own electricity, and we did not run a second brand of card. So this shows a large speed advantage on this computer at about 105 watts. It does not yet show a data-center bill or a winner between two graphics cards.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="Blackwell card versus the processor beside it", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    pics = figures()
    detailed(OUT / "blackwell-session-detailed.pdf", pics)
    brief(OUT / "blackwell-session-brief.pdf", pics)
    print("ok")
