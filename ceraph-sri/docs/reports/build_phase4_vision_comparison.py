"""Plain-language comparison of the 14 Nov session with current practice."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
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


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cover", fontName="Times-Bold", fontSize=16, leading=20, textColor=NAVY, spaceAfter=6))
    base.add(ParagraphStyle("Sub", fontName="Times-Italic", fontSize=10, leading=13, textColor=colors.HexColor("#3d4754"), spaceAfter=8))
    base.add(ParagraphStyle("H", fontName="Times-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=4))
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=7))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8, leading=10.4, textColor=NAVY))
    base.add(ParagraphStyle("Head", fontName="Times-Bold", fontSize=8, leading=10.4, textColor=colors.white))
    return base


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#5c6773"))
    canvas.drawString(0.7 * inch, 0.4 * inch, "Ceraph Corporation  ·  Why the vision measured better  ·  14 Nov 2026")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.4 * inch, str(doc.page))
    canvas.restoreState()


def table(headers, rows, widths):
    s = styles()
    data = [[Paragraph(h, s["Head"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["Cell"]) for c in row])
    built = Table(data, colWidths=widths, repeatRows=1)
    built.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return built


def story():
    s = styles()
    return [
        Paragraph("Why this is better than what computing does today", s["Cover"]),
        Paragraph("A plain-language comparison using the 14 November telemetry and 3D histograms. Ceraph Corporation.", s["Sub"]),
        Paragraph("Today’s usual habit is to smooth noise with a simple average, to pick a program’s inner settings by hand, and to let three different software tools drift until nobody is sure they still mean the same thing. The vision was kinder and stricter at once: keep what is steady, take the error away, and have every language agree. The histograms below are that vision, counted.", s["Body"]),
        Paragraph("The witness keeps the center", s["H"]),
        Paragraph("A moving average is the everyday tool. On the locked noisy series it left an error of 1.68. Our flux method, which keeps the local center and quiets only the wiggle, left 1.47. It did this without being shown the clean answer, and it did it again when the whole signal was shifted. The raw observation sat at 6.91. Less error, same center: that is the Advaita-shaped idea in a number.", s["Body"]),
        Image(str(FIG / "spec_quality_3d.png"), width=6.3 * inch, height=3.7 * inch),
        Spacer(1, 6),
        Paragraph("One truth, three languages, four boards", s["H"]),
        Paragraph("Julia chose the working size. JAX, on all four NVIDIA RTX PRO 6000 Blackwell chips, computed the same map. Triton ran it on every chip. Each matched the reference to about a millionth. Current practice often has a notebook, a math library, and a fast kernel that slowly become three different programs. Here they stayed one program. That agreement is the paradigm shift in miniature: the idea is no longer trapped in a single language.", s["Body"]),
        Paragraph("Fast without growing angry as the job grows", s["H"]),
        Paragraph("From about eight thousand numbers to about one million, the Triton time on every board stayed near one hundredth of a millisecond. A budget written months ago was about a quarter of a millisecond. We came in far under it, on four boards, and the time did not swell as the length grew. The sign flip that builds the “opposite” of the noise matched exact negation, with error zero. Against PyTorch’s ordinary version of the same arithmetic, our kernel was quicker: 0.012 ms versus 0.016 ms. Against a plain Python loop, the GPU was about fifty times sooner.", s["Body"]),
        Image(str(FIG / "spec_triton_latency_3d.png"), width=6.3 * inch, height=3.7 * inch),
        Spacer(1, 6),
        Paragraph("Quiet power", s["H"]),
        Paragraph("Each board is built for 600 watts. Sitting still they drew about 30 watts at about 30 °C. While all four worked, they drew about 86 watts and stayed between 33 and 36 °C. The vision of computing that does not have to burn the room down showed up as a small, cool load on the strongest commercial boards we could rent. The histograms make that gap obvious: a low idle bar, a modest working bar, and a ceiling six times higher that we did not need.", s["Body"]),
        Image(str(FIG / "spec_power_3d.png"), width=6.3 * inch, height=3.7 * inch),
        Spacer(1, 6),
        Image(str(FIG / "spec_temperature_3d.png"), width=6.3 * inch, height=3.7 * inch),
        Spacer(1, 8),
        table(
            ["Vision", "What people use now", "What we measured", "Why it counts"],
            [
                ["Keep the real, remove the error", "Moving average, error 1.68", "Flux method, error 1.47", "More accurate, answer not handed in"],
                ["One idea, many languages", "Tools that drift apart", "Julia, JAX, and Triton agreed", "The stack is one truth"],
                ["A fearless opposite of the noise", "Approximate cleanup", "Sign flip, error 0", "The opposite is exact"],
                ["Speed that does not balloon", "Time that grows with size", "About 0.010 ms from 8 thousand to 1 million", "The job stayed quick"],
                ["Chosen settings that are actually best", "A width picked by habit", "Block 256 was the fastest of four", "The rule picked the winner"],
                ["Cool, sparing machines", "Boards rated at 600 W", "About 86 W and under 36 °C", "The work was quiet"],
                ["Four witnesses, one answer", "One machine, one hope", "Four boards, same map, same match", "The result repeated"],
            ],
            [1.5 * inch, 1.7 * inch, 1.9 * inch, 1.6 * inch],
        ),
        Spacer(1, 8),
        Paragraph("A plain blur is still the quicker tool when the only question is the clock: an average pool finished a random million-element vector in 0.016 ms and the fuller flux map took 0.064 ms. Splitting that small job across four chips was also slower than leaving it on one. Those two facts keep the praise honest. They do not touch the comparisons above. Where the vision was accuracy, agreement, exactness, a calm clock, a wise block size, and a cool board, the telemetry says the vision counted.", s["Body"]),
    ]


def write_docx(path: Path) -> None:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    document.add_heading("Why this is better than what computing does today", 0)
    document.add_paragraph("A plain-language comparison. Ceraph Corporation, 14 November 2026.")
    paragraphs = [
        "Today’s usual habit is to smooth noise with a simple average, to pick inner settings by hand, and to let separate software tools drift apart. The vision was to keep what is steady, take the error away, and have every language agree. The pictures in this note are that vision, counted.",
        "A moving average left an error of 1.68. Our flux method, which keeps the local center and quiets only the wiggle, left 1.47. It was not shown the clean answer, and the lead held when the signal was shifted. The raw noisy series sat at 6.91.",
        "Julia chose the working size. JAX computed the same map on all four NVIDIA RTX PRO 6000 Blackwell chips. Triton ran it on every chip. Each matched the reference to about a millionth. Three languages stayed one program.",
        "From about eight thousand numbers to about one million, Triton time stayed near one hundredth of a millisecond on every board, under a quarter-millisecond budget from the earlier notebooks. The sign flip matched negation with error zero. Our kernel was quicker than PyTorch’s ordinary version of the same arithmetic, 0.012 ms versus 0.016 ms, and about fifty times sooner than a plain Python loop.",
        "Each board is built for 600 watts. Idle was about 30 watts and 30 °C. At work, about 86 watts and 33 to 36 °C. The machines did the job in a quiet band.",
        "A plain blur is still quicker when the only question is the clock: 0.016 ms for an average pool against 0.064 ms for the fuller flux map. Splitting this small job across four chips was slower than one chip. Those facts sit beside the wins. Where the vision was accuracy, agreement, exactness, a calm clock, the best block size, and a cool board, the measurements say the vision counted.",
    ]
    for text in paragraphs:
        document.add_paragraph(text)
    for name in ("spec_quality_3d.png", "spec_triton_latency_3d.png", "spec_power_3d.png", "spec_temperature_3d.png"):
        document.add_picture(str(FIG / name), width=Inches(6.2))
    document.save(path)


def main() -> None:
    pdf = OUT / "phase4-vision-comparison.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.6 * inch, title="Why this is better", author="Ceraph Corporation")
    doc.build(story(), onFirstPage=footer, onLaterPages=footer)
    write_docx(OUT / "phase4-vision-comparison.docx")
    print("ok")


if __name__ == "__main__":
    main()
