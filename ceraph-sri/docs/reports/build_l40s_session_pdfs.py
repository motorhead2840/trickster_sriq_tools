"""Session note from the 14 November 2026 L40S notebook run."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#1c2430")
RULE = colors.HexColor("#c5cdd8")
ROW = colors.HexColor("#f4f6f8")


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cover", fontName="Times-Bold", fontSize=16, leading=20, textColor=NAVY, spaceAfter=8))
    base.add(ParagraphStyle("Sub", fontName="Times-Italic", fontSize=10, leading=13, textColor=colors.HexColor("#3d4754"), spaceAfter=10))
    base.add(ParagraphStyle("H", fontName="Times-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=4))
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=10, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=8))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8, leading=10.5, textColor=NAVY))
    base.add(ParagraphStyle("CellHead", fontName="Times-Bold", fontSize=8, leading=10.5, textColor=colors.white))
    return base


def table(headers, rows, widths):
    s = styles()
    data = [[Paragraph(h, s["CellHead"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["Cell"]) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#5c6773"))
    canvas.drawString(0.75 * inch, 0.4 * inch, "Ceraph Corporation  ·  L40S session 14 Nov 2026  ·  Synthetic host measurement")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.4 * inch, str(doc.page))
    canvas.restoreState()


def detailed(path: Path) -> None:
    s = styles()
    story = [
        Paragraph("L40S session record, 14 November 2026", s["Cover"]),
        Paragraph("One NVIDIA L40S on DGX Cloud. Synthetic workloads. Not the unpublished method, and not a facility power bill.", s["Sub"]),
        Paragraph("Machine", s["H"]),
        Paragraph("The notebook confirmed an NVIDIA L40S, driver 565.57.01, 46,068 MiB of memory, a 350 W board limit, Python 3.12.16, and PyTorch 2.14.1 built for CUDA 12.6. A first probe read 32.40 W and 30 °C. Twenty idle samples over the next 20 seconds averaged 38.27 W, with temperature between 30 °C and 31 °C. A later loop that launched the one-million-length GPU kernel 30 times, and asked the driver for power between launches, averaged 84.72 W and reached 36 °C. That is about 2.2 times the idle draw and about a quarter of the 350 W limit. The driver query is slow relative to the kernel, so 84.72 W is the board during that mixed loop, not a pure kernel joule count.", s["Body"]),
        Paragraph("Speed on this host", s["H"]),
        Paragraph("The CPU test used a synthetic float32 map-and-reduce. Median time rose from 0.014 ms at length 4,096 to 1.885 ms at length 1,048,576. The slope of log time against log length was 0.895, close to 1, so CPU time grew almost in proportion to length. The same shape of work on the L40S stayed near 0.041 ms at lengths 4,096 and 65,536, and 0.046 ms at length 1,048,576. At the largest length the GPU median was about 41 times shorter than the CPU median on this same machine. At length 4,096 the CPU was faster, 0.014 ms against 0.041 ms, which is what a short launch looks like when the arithmetic is too small to hide the launch.", s["Body"]),
        table(
            ["Length", "CPU median (ms)", "GPU median (ms)"],
            [
                ["4,096", "0.014", "0.041"],
                ["16,384", "0.034", "not timed"],
                ["65,536", "0.125", "0.041"],
                ["262,144", "0.478", "not timed"],
                ["1,048,576", "1.885", "0.046"],
            ],
            [2.2 * inch, 2.2 * inch, 2.2 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The flat GPU times are a real measurement of this synthetic kernel. They are not a proof that every Ceraph workload is constant-time, and they are not a comparison with the RTX PRO 6000 Blackwell session from 7 November. That earlier session timed a different kernel, at one length, with a median of 0.08884 ms.", s["Body"]),
        Paragraph("Error check", s["H"]),
        Paragraph("On 80 synthetic test series of length 128, a moving average of width 9 had mean squared error 0.0324. A median filter of the same width had 0.0469. The average was more accurate. This check does not run the unpublished method.", s["Body"]),
        Paragraph("What this does not yet say", s["H"]),
        Paragraph("The loaded reading does not fill a data-center bill. 84.72 W and 36 °C describe this short synthetic loop on one L40S. They do not describe a rack, a second accelerator, or the unpublished method. The rental rate on the screen was $1.74 per hour. The cells themselves took minutes. That is the cost of this session, not a customer price.", s["Body"]),
        Paragraph("A customer sentence that is supported today is narrow. For this synthetic map-and-reduce, once the vector is about a million long, the L40S finished about 41 times sooner than the CPU in the same machine, while the board sat near 85 W and 36 °C rather than near its 350 W cap. For a vector of a few thousand, the CPU was faster. Quality on the synthetic series favored the moving average over the median. A cooler, cheaper facility claim still needs the same loop on a second card and a workload heavy enough to approach the power limit.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch, title="L40S session record", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def brief(path: Path) -> None:
    s = styles()
    story = [
        Paragraph("What the L40S test showed", s["Cover"]),
        Paragraph("Ceraph Corporation  ·  14 November 2026  ·  Plain language", s["Sub"]),
        Paragraph("We rented one NVIDIA L40S graphics card. It was identified correctly. While waiting it drew about 38 watts and stayed near 30 °C, against a 350 watt maximum. While repeating the large synthetic job it drew about 85 watts and reached 36 °C.", s["Body"]),
        Paragraph("On a made-up arithmetic job, the card's time stayed near 0.05 thousandths of a second as the job grew from about four thousand numbers to about one million. The ordinary processor in the same machine slowed as the job grew, from about 0.014 thousandths of a second to about 1.9. At the large size the card was about 41 times quicker. At the small size the ordinary processor was quicker, because starting the card costs more than the tiny bit of arithmetic.", s["Body"]),
        Paragraph("On a separate made-up noisy series, a simple average was more accurate than a median filter. That test is not the private method.", s["Body"]),
        Paragraph("This is evidence that this card can finish a large synthetic job much sooner than its own CPU, at about 85 watts and 36 °C rather than at its 350 watt maximum. It is not yet a data-center power bill, a rack design, or a win over the earlier Blackwell card. Those need the same job on a second card, and a heavier job that actually presses the power limit.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch, title="What the L40S test showed", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    detailed(OUT / "l40s-session-detailed.pdf")
    brief(OUT / "l40s-session-brief.pdf")
    print("ok")
