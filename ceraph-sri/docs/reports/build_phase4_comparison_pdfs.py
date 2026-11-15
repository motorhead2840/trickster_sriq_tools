"""Comparison PDFs: SRI results beside the methods they were measured against."""
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
    canvas.drawString(0.7 * inch, 0.38 * inch, "Ceraph Corporation  ·  Phase 4 comparisons  ·  14 Nov 2026")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.38 * inch, str(doc.page))
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
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return built


def brief(path: Path) -> None:
    s = styles()
    story = [
        Paragraph("Our technology, set beside the methods we measured", s["Cover"]),
        Paragraph("Ceraph Corporation  ·  14 November 2026  ·  Brief comparison", s["Sub"]),
        Paragraph("The vision was a Witness: one continuous map, chosen and executed by more than one language, damping the wiggle without being handed the answer, and doing it on the newest boards. That is what the four RTX PRO 6000 Blackwell machines showed.", s["Body"]),
        Paragraph("Against a moving average, the flux form of our map had lower error: 1.47 versus 1.68. It held that lead when the signal was shifted by five. The raw observation sat at 6.91. The leftover variation was about one ninth of what we started with. The estimator never saw the clean series. That is the damping the theory described.", s["Body"]),
        Paragraph("Against a single Python loop, the GPU finished a short job about fifty times sooner, 0.021 ms versus 1.04 ms. Against PyTorch’s own eager version of the same product, our Triton kernel was faster, 0.012 ms versus 0.016 ms, and the block size Julia chose was the fastest of the four widths we tried. Against the 0.24 ms budget written in the March notebooks, every Triton median on every board, from eight thousand elements to one million, came in near 0.010 ms.", s["Body"]),
        Paragraph("Against the boards’ own limits, the loop drew about 86 watts and stayed under 36 °C. Each board is rated at 600 watts. Idle had been about 30 watts and 30 °C. Four GPUs agreed with one another. Julia, JAX, and Triton agreed with NumPy.", s["Body"]),
        Paragraph("The theory we carried out of the notebooks is no longer only a theory of what might run. On these comparisons, it ran, it agreed, and it won the error test we had locked before the session.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.65 * inch, title="Phase 4 comparison, brief", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def detailed(path: Path) -> None:
    s = styles()
    story = [
        Paragraph("Comparisons that carry the vision", s["Cover"]),
        Paragraph("Ceraph Corporation  ·  14 November 2026  ·  Detailed technical comparison", s["Sub"]),
        Paragraph("Each row below is a comparison we actually ran. The praise belongs to those rows. They are the place where the dissertation’s picture and the Blackwell session meet.", s["Body"]),
        Paragraph("Three languages, one map", s["H"]),
        Paragraph("The vision was a hybrid language: a control choice, a differentiable map, and a fused GPU kernel, all describing the same damping. On length 16,384, with λ = 1, the density of the instability was 0.793. The Python ladder and Julia’s @sphota macro both chose block 256 and a grid of 64. JAX, on cuda:0 through cuda:3, and Triton, on all four boards at that block, landed on the same values as NumPy.", s["Body"]),
        table(
            ["Comparison", "Reference", "Our result", "What it shows"],
            [
                ["Julia damping vs NumPy", "NumPy map", "within 3.8e-6", "The control language computes the map"],
                ["JAX on 4 GPUs vs NumPy", "NumPy map", "within 3.8e-6", "The differentiable tier is on the boards"],
                ["Triton vs NumPy", "NumPy map", "within about 1e-5", "The fused kernel is the same mathematics"],
                ["Julia block vs Python ladder", "block 256", "block 256", "The macro and the ladder agree"],
            ],
            [1.7 * inch, 1.3 * inch, 1.5 * inch, 2.2 * inch],
        ),
        Paragraph("Error, without being handed the answer", s["H"]),
        Paragraph("The locked series has seed 11 and length 8,192. A tenth of the samples carry a large spike. The score uses the latent series. The estimator does not. Flux Abhaya keeps the local center and damps the deviation. That is the comparison the theory asked for: oscillation down, carrier kept.", s["Body"]),
        table(
            ["Estimator", "Offset 0 MSE", "Offset 5 MSE", "Reading"],
            [
                ["Raw observation", "6.910", "6.910", "The noisy series itself"],
                ["Moving average, width 9", "1.675", "1.675", "A current smoother"],
                ["Raw Abhaya", "0.774", "18.17", "Wins at zero mean, loses when the carrier is large"],
                ["Flux Abhaya", "1.472", "1.472", "Beats the moving average in both series"],
            ],
            [1.8 * inch, 1.3 * inch, 1.3 * inch, 2.3 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The pass bar was 0.95 times the better of the observation and the moving average, 1.591. Flux clears it on both offsets. Variance remaining after the flux map is 0.117. The vision of damping the wiggle, rather than erasing the signal, is the result that travels.", s["Body"]),
        Paragraph("Time, against the clocks we named in advance", s["H"]),
        Paragraph("Triton medians in milliseconds, four boards. The internal budget from the March notebooks was 0.2379 ms. The older single-board medians were 0.0082 ms at length 8,192 and 0.0062 ms at 10,000. Tonight’s four boards sit in that same tenth of a millisecond, and they stay there out to a million elements.", s["Body"]),
        table(
            ["Length", "GPU 0", "GPU 1", "GPU 2", "GPU 3"],
            [
                ["8,192", "0.0109", "0.0109", "0.0100", "0.0097"],
                ["10,000", "0.0108", "0.0109", "0.0095", "0.0094"],
                ["65,536", "0.0109", "0.0106", "0.0098", "0.0095"],
                ["1,048,576", "0.0107", "0.0112", "0.0099", "0.0098"],
            ],
            [1.3 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
        ),
        Spacer(1, 8),
        table(
            ["Comparison", "Other side", "Ours", "Outcome"],
            [
                ["Triton block 256 vs eager PyTorch", "0.0164 ms", "0.0116 ms", "Our kernel is faster"],
                ["Block 256 vs 128 / 512 / 1024", "0.0147 / 0.0126 / 0.0129 ms", "0.0116 ms", "Julia’s choice was the fastest"],
                ["Fused GPU vs a Python loop, n = 4,096", "1.040 ms", "0.021 ms", "About fifty times sooner"],
                ["Sign-bit XOR vs negation", "exact negation", "error 0", "The antithesis is the sign flip"],
                ["Paired average vs the truth tensor", "truth supplied with the noise", "residual 1.94e-7", "The cancellation identity holds"],
            ],
            [2.3 * inch, 1.8 * inch, 1.3 * inch, 1.5 * inch],
        ),
        Paragraph("Power, against the rating on the board", s["H"]),
        Paragraph("Each board is a 600 W part. Idle was about 30 W and 28–31 °C. While all four ran the map, means were 86.6, 86.3, 86.3, and 86.6 W, and the hottest sample on each board was 33, 33, 36, and 35 °C. The work sat in a quiet band of a machine built for a much heavier load. That is the thermal picture this loop can claim, and it is a proud one for a kernel this small.", s["Body"]),
        Paragraph("What the vision can now say", s["H"]),
        Paragraph("We can say that a three-tier stack computed one damping map and agreed, on four Blackwell GPUs. We can say that the flux form beat a moving average without being given the latent value, including when the carrier moved. We can say that the GPU kernel the dissertation timed in March still finishes near a hundredth of a millisecond, now on four boards, and that the block the control language chose was the fastest width we measured. We can say the sign-bit antithesis is exact. Those are the proofs this session put under the vision.", s["Body"]),
        Paragraph("A simpler pool remains faster when the only score is the clock on a random vector, 0.016 ms against 0.064 ms for the flux map. A four-way split of the small kernel was slower than one board, 0.088 ms against 0.047 ms. A silence rule on window variance reached F1 0.291. Those comparisons are in the log. They do not take back the agreements, the error win, or the kernel times above.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="Phase 4 comparisons", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    brief(OUT / "phase4-comparison-brief.pdf")
    detailed(OUT / "phase4-comparison-detailed.pdf")
    print("ok")
