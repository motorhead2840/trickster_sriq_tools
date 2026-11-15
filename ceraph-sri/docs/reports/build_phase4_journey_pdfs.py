"""Brief and detailed PDFs for the Phase 4 journey through 14 Nov 2026."""
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
    canvas.drawString(0.7 * inch, 0.38 * inch, "Ceraph Corporation  ·  Phase 4 journey  ·  14 Nov 2026")
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
        Paragraph("Phase 4: what the four Blackwell boards showed", s["Cover"]),
        Paragraph("Ceraph Corporation  ·  14 November 2026  ·  A short account", s["Sub"]),
        Paragraph("The research began as notebooks and a dissertation. It became a stack that runs. On 14 November, four NVIDIA RTX PRO 6000 Blackwell boards ran the same damping map three ways: Julia chose the block size, JAX computed the map on all four GPUs, and Triton launched that block on every board. Each result matched the NumPy reference.", s["Body"]),
        Paragraph("The map that keeps the center and damps the wiggle beat an ordinary moving average. Error was 1.47 against 1.68, whether the signal sat near zero or was shifted by five. The leftover variation was about one ninth of the original. A sign-bit flip on the GPU matched negation exactly, in about a hundredth of a millisecond.", s["Body"]),
        Paragraph("Across lengths from about eight thousand numbers to about one million, the Triton times stayed near 0.010 ms on every board, under a 0.24 ms budget carried forward from the March notebooks. The boards idled near 30 watts and 30 °C. Under the test loop they drew about 86 watts and stayed under 36 °C, against a 600 watt limit.", s["Body"]),
        Paragraph("Two comparisons stayed in the open. A silence rule based on local variation did not yet separate spikes from ordinary samples. Splitting one small job across four boards was slower than leaving it on one board. An average pool was still faster than the fuller damping map when the only question was the clock. Those results sit beside the wins. They make the record something a partner can trust.", s["Body"]),
        Paragraph("A heat meter, an optical crystal, and a fabricated quasicrystal were not in the room. The digital form the dissertation called an asymptote of that future did run, on the hardware the notebooks had been reaching for.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.65 * inch, title="Phase 4 brief", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def detailed(path: Path) -> None:
    s = styles()
    story = [
        Paragraph("Phase 4 technical record", s["Cover"]),
        Paragraph("Ceraph Corporation  ·  14 November 2026  ·  From locked mathematics to four Blackwell GPUs", s["Sub"]),
        Paragraph("Journey", s["H"]),
        Paragraph("The published unveiling rounds, the E1 lock, the observable-only benchmark, and conjectures C1–C4 are held by regression tests on CPU. Phase 3 then passed a preregistered sensing test: the deployable dual-mode path, algebraically the Wiener filter, cut mean error 12.05% versus single-mode Wiener. The bar was 8%. An L40S session and a one-board Blackwell session measured real GPUs against the CPUs beside them. A same-GPU average pool beat an earlier, heavier kernel on time. Phase 4 kept that lesson and asked a sharper question: can Julia, JAX, and Triton agree on one map, on Blackwell, and can the flux form of Abhaya beat a moving average without seeing the latent value?", s["Body"]),
        Paragraph("The machine", s["H"]),
        Paragraph("Four NVIDIA RTX PRO 6000 Blackwell Server Edition boards. Each has 97,887 MiB and a 600 W limit. Software was Python 3.12, PyTorch 2.14.1+cu130, Triton, and JAX with the CUDA 13 plugin. Idle readings were about 30.5, 30.0, 29.9, and 30.4 W, at 28, 29, 31, and 30 °C.", s["Body"]),
        Paragraph("The stack", s["H"]),
        Paragraph("The map is A = theta times exp(−λ ξ), with λ = 1. On a length-16,384 draw, the mean absolute value of ξ was 0.793, so both the Python ladder and Julia’s @sphota macro chose block 256 and a grid of 64. Julia’s result and JAX’s result each differed from NumPy by at most 3.8e-6. Triton, launched at that block on cuda:0 through cuda:3, differed by at most about 7.6e-6. JAX’s first import was a CPU build. After jax[cuda13] and a kernel restart, jax.devices() reported cuda:0 through cuda:3 and the same error bound. That is the hybrid stack the notebooks described.", s["Body"]),
        table(
            ["Tier", "Where it ran", "Match to NumPy"],
            [
                ["Julia @sphota + damping", "CPU process, block 256", "3.8e-6"],
                ["JAX", "cuda:0–cuda:3", "3.8e-6"],
                ["Triton", "all four boards, block 256", "about 7.6e-6"],
            ],
            [2.2 * inch, 2.4 * inch, 1.8 * inch],
        ),
        Paragraph("Quality", s["H"]),
        Paragraph("Seed 11, length 8,192, a window of 9. The estimator sees only the observation. The latent series is used for the score. The pass bar is 0.95 times the better of the raw observation and a moving average.", s["Body"]),
        table(
            ["Series", "Identity", "Moving average", "Raw Abhaya", "Flux Abhaya"],
            [
                ["Offset 0", "6.910", "1.675", "0.774 supported", "1.472 supported"],
                ["Offset 5", "6.910", "1.675", "18.17 not supported", "1.472 supported"],
            ],
            [1.2 * inch, 1.2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch],
        ),
        Spacer(1, 8),
        Paragraph("Flux variance remaining is 0.117 of the observation variance. The omega-dit sample on the offset-0 series recorded sigma 1.448, upsilon −0.040, and xi 7.056. The flux form is the one that survives a shifted carrier. That is the dissertation’s damping of the oscillation, not of the whole signal.", s["Body"]),
        Paragraph("Triton at the dissertation lengths", s["H"]),
        Paragraph("Medians in milliseconds. Every launch matched the reference and sat under the 0.2379 ms budget. The March notebook medians were 0.0082 ms at 8,192 and 0.0062 ms at 10,000. Tonight is the same neighborhood, on four boards.", s["Body"]),
        table(
            ["GPU", "8,192", "10,000", "65,536", "1,048,576", "Sign XOR"],
            [
                ["0", "0.0109", "0.0108", "0.0109", "0.0107", "0 error, 0.0104 ms"],
                ["1", "0.0109", "0.0109", "0.0106", "0.0112", "0 error, 0.0112 ms"],
                ["2", "0.0100", "0.0095", "0.0098", "0.0099", "0 error, 0.0093 ms"],
                ["3", "0.0097", "0.0094", "0.0095", "0.0098", "0 error, 0.0096 ms"],
            ],
            [0.6 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 1.1 * inch, 1.6 * inch],
        ),
        Spacer(1, 8),
        Paragraph("A separate PyTorch map, y times exp(−y²), stayed between 0.017 and 0.019 ms from 16,384 to 1,048,576 elements on every board. Slopes of log time against log length were −0.0095, 0.0004, −0.0008, and 0.0043. These kernels are launch-bound. The flatness is a property of these maps at these sizes.", s["Body"]),
        Paragraph("Block size, dtypes, one board and four", s["H"]),
        Paragraph("At one million elements on GPU 0, Triton medians were 0.0147, 0.0116, 0.0126, and 0.0129 ms for blocks 128, 256, 512, and 1024. Block 256, the one Julia chose, was the fastest and beat eager PyTorch at 0.0164 ms. Float32, float16, and bfloat16 medians were 0.0217, 0.0202, and 0.0201 ms. Float8 storage allocated and was not timed.", s["Body"]),
        Paragraph("One board finished 4,194,304 elements in 0.047 ms. The same count split across four boards took 0.088 ms. For this small kernel the split is not a win. The single-board number is the one to carry forward. A Python loop of 4,096 elements took 1.040 ms. The fused GPU map of that length took 0.021 ms.", s["Body"]),
        Paragraph("On a million-element random vector the flux map took 0.064 ms and an average pool took 0.016 ms. The pool wins that clock. The locked series still gives flux the lower error. Speed and accuracy are both measured, and they answer different questions.", s["Body"]),
        Paragraph("Power and the algebra that had both tensors", s["H"]),
        Paragraph("Thirty samples during a four-board loop averaged 86.6, 86.3, 86.3, and 86.6 W, with peaks of 33, 33, 36, and 35 °C. That is headroom against 600 W. It is a light loop, not a board pushed to its limit.", s["Body"]),
        Paragraph("Paired cancellation on 10,000 elements, truth and noise both supplied, left a residual of 1.94e-7. The entropy term in the written bracket is then zero because the residual is under 1e-6. That is an algebraic identity on those inputs. A Fibonacci window of 13 summed to weight 2.61, and the filtered energy was 1.56 times the noise energy, so that window did not shrink this noise. Circuit A printed 1.09 and circuit B sat at its cap of 50.", s["Body"]),
        Paragraph("The silence gate", s["H"]),
        Paragraph("Eight thresholds on window variance. The best was threshold 8: precision 0.199, recall 0.544, F1 0.291, below the 0.5 bar. Of 812 spikes, 442 were caught and 370 missed, with 1,782 clean samples also flagged. The wish to stay silent rather than guess remains a product idea. This feature does not yet carry it.", s["Body"]),
        Paragraph("What this proves, in the sense the journey asked for", s["H"]),
        Paragraph("It proves a hybrid implementation: one damping map, three runtimes, four Blackwell GPUs, numerical agreement, a block size chosen by the Julia macro that was also the fastest Triton width, a flux estimator that beats a moving average without the latent value, a sign-bit antithesis with zero error, and a power envelope far under the board limit for this work. It proves the March timing neighborhood again, at larger scale. It also proves the value of the misses. The record can say what ran, what won, and what is still ahead: a sharper silence feature, a labeled language exam, and the physical instruments that were never on this machine.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.65 * inch, title="Phase 4 technical record", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    brief(OUT / "phase4-journey-brief.pdf")
    detailed(OUT / "phase4-journey-detailed.pdf")
    print("ok")
