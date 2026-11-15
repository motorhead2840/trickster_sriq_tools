"""Technical specification and patent specification, PDF and DOCX, from the 14 Nov 2026 session."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import numpy as np
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#1c2430")
RULE = colors.HexColor("#c5cdd8")
ROW = colors.HexColor("#f4f6f8")


def pdf_styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("Cover", fontName="Times-Bold", fontSize=16, leading=20, textColor=NAVY, spaceAfter=6))
    base.add(ParagraphStyle("Sub", fontName="Times-Italic", fontSize=10, leading=13, textColor=colors.HexColor("#3d4754"), spaceAfter=8))
    base.add(ParagraphStyle("H", fontName="Times-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=4))
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=10, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=6))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8, leading=10.2, textColor=NAVY))
    base.add(ParagraphStyle("Head", fontName="Times-Bold", fontSize=8, leading=10.2, textColor=colors.white))
    return base


def pdf_table(headers, rows, widths):
    s = pdf_styles()
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


def footer(label):
    def _draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Italic", 8)
        canvas.setFillColor(colors.HexColor("#5c6773"))
        canvas.drawString(0.65 * inch, 0.38 * inch, label)
        canvas.drawRightString(letter[0] - 0.65 * inch, 0.38 * inch, str(doc.page))
        canvas.restoreState()
    return _draw


def make_figures() -> dict[str, Path]:
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def canvas(name, title, draw):
        fig = plt.figure(figsize=(10.2, 6.0), facecolor="#10141c")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#10141c")
        draw(ax)
        ax.tick_params(colors="#e8edf5")
        ax.xaxis.label.set_color("#e8edf5")
        ax.yaxis.label.set_color("#e8edf5")
        ax.zaxis.label.set_color("#e8edf5")
        fig.suptitle(title, color="#f4f7fb")
        path = folder / name
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths[name] = path

    def quality(ax):
        colors = ["#8aa0b8", "#e09a55", "#7d8cff", "#3ecf8e"]
        offset0 = [6.910, 1.675, 0.774, 1.472]
        offset5 = [6.910, 1.675, 18.17, 1.472]
        for col, heights in enumerate((offset0, offset5)):
            ax.bar3d(np.arange(4), np.full(4, col), np.zeros(4), 0.45, 0.45, heights, color=colors, shade=True)
        ax.set_xticks(np.arange(4) + 0.2)
        ax.set_xticklabels(["observation", "average", "raw", "flux"], color="#e8edf5", rotation=15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["offset 0", "offset 5"], color="#e8edf5")
        ax.set_zlabel("MSE")
        ax.view_init(elev=22, azim=-58)

    def latency(ax):
        series = [
            [0.0109, 0.0108, 0.0109, 0.0107],
            [0.0109, 0.0109, 0.0106, 0.0112],
            [0.0100, 0.0095, 0.0098, 0.0099],
            [0.0097, 0.0094, 0.0095, 0.0098],
        ]
        palette = ["#7d8cff", "#3ecf8e", "#ffb703", "#e09a55"]
        for device, heights in enumerate(series):
            ax.bar3d(np.arange(4), np.full(4, device), np.zeros(4), 0.4, 0.45, heights, color=palette[device], shade=True)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["8k", "10k", "65k", "1M"], color="#e8edf5")
        ax.set_ylabel("GPU")
        ax.set_zlabel("ms")
        ax.view_init(elev=24, azim=-60)

    def telemetry(ax):
        idle = [30.45, 29.95, 29.91, 30.44]
        load = [86.59, 86.30, 86.28, 86.55]
        ax.bar3d(np.arange(4) - 0.15, np.zeros(4), np.zeros(4), 0.3, 0.4, idle, color="#6ec8ff", shade=True)
        ax.bar3d(np.arange(4) + 0.15, np.ones(4), np.zeros(4), 0.3, 0.4, load, color="#ff7a59", shade=True)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["GPU 0", "GPU 1", "GPU 2", "GPU 3"], color="#e8edf5")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["idle", "load"], color="#e8edf5")
        ax.set_zlabel("Watts")
        ax.view_init(elev=22, azim=-58)

    def temperature(ax):
        idle_c = [28, 29, 31, 30]
        load_c = [33, 33, 36, 35]
        ax.bar3d(np.arange(4) - 0.15, np.zeros(4), np.zeros(4), 0.3, 0.4, idle_c, color="#3ecf8e", shade=True)
        ax.bar3d(np.arange(4) + 0.15, np.ones(4), np.zeros(4), 0.3, 0.4, load_c, color="#c77dff", shade=True)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["GPU 0", "GPU 1", "GPU 2", "GPU 3"], color="#e8edf5")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["idle", "peak"], color="#e8edf5")
        ax.set_zlabel("°C")
        ax.view_init(elev=22, azim=-58)

    canvas("spec_quality_3d.png", "Error comparison, latent value withheld", quality)
    canvas("spec_triton_latency_3d.png", "Triton median milliseconds on four boards", latency)
    canvas("spec_power_3d.png", "Telemetry: idle and load power", telemetry)
    canvas("spec_temperature_3d.png", "Telemetry: idle and peak temperature", temperature)
    return paths


def picture(path: Path):
    return Image(str(path), width=6.4 * inch, height=3.75 * inch)


def tech_story(figs: dict[str, Path]):
    s = pdf_styles()
    return [
        Paragraph("Technical specification", s["Cover"]),
        Paragraph("SRI Witness damping stack on NVIDIA RTX PRO 6000 Blackwell × 4<br/>Ceraph Corporation · 14 November 2026 · Specification of the measured system", s["Sub"]),
        Paragraph("1. Scope", s["H"]),
        Paragraph("This specification describes the system that ran on 14 November 2026: a hybrid implementation of one damping map, executed by NumPy, Julia, JAX, and Triton, on four NVIDIA RTX PRO 6000 Blackwell Server Edition graphics processors. It records the interfaces, the equations as implemented, the test protocol, and the measured results. It specifies that system. It does not specify a calorimeter, an optical crystal, or a fabricated quasicrystal.", s["Body"]),
        Paragraph("2. Hardware", s["H"]),
        Paragraph("Four processors, each an NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB of memory, 600 W board limit. Idle, before the suite: 30.45 W at 28 °C, 29.95 W at 29 °C, 29.91 W at 31 °C, and 30.44 W at 30 °C. Clocks at idle were 180 MHz and utilization was zero.", s["Body"]),
        Paragraph("3. Software stack", s["H"]),
        Paragraph("Python 3.12. PyTorch 2.14.1 built for CUDA 13.0. Triton. JAX with the CUDA 13 plugin, backend gpu, devices cuda:0 through cuda:3. Julia 1.13.4, invoked as a subprocess. The control macro @sphota selects a launch block from the mean absolute value of the instability: 128 below 0.5, 256 below 1, 512 below 2, otherwise 1024. On the session draw that density was 0.793, so the block was 256 and the grid was 64 for a length of 16,384.", s["Body"]),
        Paragraph("4. The map", s["H"]),
        Paragraph("Raw form: each element of the observation is multiplied by exp(−λ ξ), with λ = 1 and ξ the variance of a window of width 9. Flux form: the local mean of that window is kept, and the same exponential multiplies only the deviation from the mean. The latent series is never an input to either form. A third identity, used only when both a truth tensor and a noise tensor are supplied, replaces the pair (truth+noise, truth−noise) by their average. A sign-bit kernel flips the IEEE-754 sign of a float32 value by an unsigned XOR with the high bit and is checked against arithmetic negation.", s["Body"]),
        Paragraph("5. Agreement of the three tiers", s["H"]),
        pdf_table(
            ["Tier", "Device", "Absolute error vs NumPy"],
            [
                ["Julia damping and @sphota", "Host process", "3.8e-6; block 256 matches the Python ladder"],
                ["JAX", "cuda:0 through cuda:3", "3.8e-6"],
                ["Triton, block 256", "Each of the four boards", "at most about 7.6e-6 on the stack check"],
            ],
            [2.2 * inch, 2.2 * inch, 2.4 * inch],
        ),
        Spacer(1, 8),
        Paragraph("6. Quality protocol and results", s["H"]),
        Paragraph("Seed 11. Length 8,192. Ten percent of the samples receive an added spike of standard deviation 8, on top of background noise of standard deviation 0.5. Pass if the estimator’s mean squared error is at most 0.95 times the better of the raw observation and a width-9 moving average.", s["Body"]),
        pdf_table(
            ["Estimator", "MSE, offset 0", "MSE, offset 5"],
            [
                ["Observation", "6.910", "6.910"],
                ["Moving average", "1.675", "1.675"],
                ["Raw map", "0.774", "18.17"],
                ["Flux map", "1.472", "1.472"],
            ],
            [2.2 * inch, 2.2 * inch, 2.2 * inch],
        ),
        Spacer(1, 6),
        Paragraph("Flux variance remaining is 0.117 of the observation variance. The flux map meets the bar at both offsets. The raw map meets it only at offset 0.", s["Body"]),
        Paragraph("7. Timing", s["H"]),
        Paragraph("Triton medians on the four boards, in milliseconds, for the damping product. Sign-bit XOR error was 0 on every board.", s["Body"]),
        pdf_table(
            ["Length", "GPU 0", "GPU 1", "GPU 2", "GPU 3"],
            [
                ["8,192", "0.0109", "0.0109", "0.0100", "0.0097"],
                ["10,000", "0.0108", "0.0109", "0.0095", "0.0094"],
                ["65,536", "0.0109", "0.0106", "0.0098", "0.0095"],
                ["1,048,576", "0.0107", "0.0112", "0.0099", "0.0098"],
            ],
            [1.4 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch],
        ),
        Spacer(1, 6),
        Paragraph("At one million elements on GPU 0, Triton block medians were 0.0147, 0.0116, 0.0126, and 0.0129 ms for widths 128, 256, 512, and 1024. Eager PyTorch of the same product was 0.0164 ms. Float32, float16, and bfloat16 medians of a related one-line map were 0.0217, 0.0202, and 0.0201 ms. One board completed 4,194,304 elements in 0.047 ms. The same count split across four boards took 0.088 ms. A host Python loop of 4,096 elements took 1.040 ms; the fused GPU map of that length took 0.021 ms. On a separate million-element vector the flux map took 0.064 ms and an average pool took 0.016 ms.", s["Body"]),
        Paragraph("8. Telemetry", s["H"]),
        Paragraph("Power and temperature were read from the NVIDIA driver on each board, once at idle and across thirty samples while all four boards ran the map. The load samples include a driver query between launches. They describe this light loop.", s["Body"]),
        pdf_table(
            ["GPU", "Idle W", "Idle °C", "Load W", "Peak °C", "Limit W"],
            [
                ["0", "30.45", "28", "86.6", "33", "600"],
                ["1", "29.95", "29", "86.3", "33", "600"],
                ["2", "29.91", "31", "86.3", "36", "600"],
                ["3", "30.44", "30", "86.6", "35", "600"],
            ],
            [0.8 * inch, 1.1 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch, 1.0 * inch],
        ),
        Spacer(1, 8),
        picture(figs["spec_quality_3d.png"]),
        Spacer(1, 6),
        picture(figs["spec_triton_latency_3d.png"]),
        Spacer(1, 6),
        picture(figs["spec_power_3d.png"]),
        Spacer(1, 6),
        picture(figs["spec_temperature_3d.png"]),
        Spacer(1, 6),
        Paragraph("Mean power 86.6, 86.3, 86.3, and 86.6 W. Maximum temperature 33, 33, 36, and 35 °C. The boards were not held at the 600 W limit.", s["Body"]),
        Paragraph("9. Acceptance of this specification", s["H"]),
        Paragraph("A reproduction accepts this specification when, on the same seed and sizes, the three tiers match NumPy within 1e-3, the flux mean squared error is at most the stated bar on both offsets, the sign-bit kernel matches negation, and a Triton median at length 10,000 is under 0.2379 ms. The silence-gate F1 on this series was 0.291 and is not an acceptance item.", s["Body"]),
    ]


def patent_story(figs: dict[str, Path]):
    s = pdf_styles()
    return [
        Paragraph("Specification for a hybrid damping system", s["Cover"]),
        Paragraph("Ceraph Corporation · 14 November 2026 · Draft specification grounded in the four-GPU Blackwell session<br/>For counsel. Not a filed application and not a grant.", s["Sub"]),
        Paragraph("Field", s["H"]),
        Paragraph("The invention concerns a computer-implemented method and system for damping a numerical series. A control program selects a launch width. A differentiable program and a fused graphics-processor kernel compute the same elementwise map. The map may be applied to a whole series or only to its deviation from a local center.", s["Body"]),
        Paragraph("Background", s["H"]),
        Paragraph("Smoothing a noisy series is ordinarily done by a local average. Launching a custom kernel ordinarily requires a programmer to pick a block size by hand, and a research notebook, a differentiable runtime, and a kernel language often drift apart. The measured system keeps those three descriptions on one formula and checks them against a single reference.", s["Body"]),
        Paragraph("Summary", s["H"]),
        Paragraph("A host process computes a density from the series and selects one of four block widths. A Julia program computes the same width and the same damped series. A JAX program computes the damped series on one or more graphics processors. A Triton kernel, launched at the selected width, computes it again. Agreement with a NumPy reference is the acceptance test. In a preferred mode the local mean is preserved and only the deviation is scaled by an exponential of a local variance. In another mode, used when a paired noise tensor is already isolated, a sign bit is inverted and the pair is averaged.", s["Body"]),
        Paragraph("Example", s["H"]),
        Paragraph("On 14 November 2026 the method ran on four RTX PRO 6000 Blackwell processors. Density 0.793 selected block 256. Julia, JAX on cuda:0–cuda:3, and Triton each matched NumPy within 1e-5. On a locked spike series of length 8,192 the flux mode had mean squared error 1.472 against 1.675 for a moving average, at carrier offset 0 and at carrier offset 5. The raw mode had error 0.774 at offset 0 and 18.17 at offset 5. Triton medians from length 8,192 to 1,048,576 remained near 0.010 ms. Sign-bit inversion matched negation with error 0. Block 256 was faster than blocks 128, 512, and 1024, and faster than an eager product on the same processor. Board power during a light four-processor loop was about 86 W, with temperature at most 36 °C.", s["Body"]),
        Paragraph("Telemetry drawings from the example", s["H"]),
        Paragraph("The following drawings are the measured histograms from the example: error, Triton latency on four processors, power, and temperature.", s["Body"]),
        picture(figs["spec_quality_3d.png"]),
        Spacer(1, 6),
        picture(figs["spec_triton_latency_3d.png"]),
        Spacer(1, 6),
        picture(figs["spec_power_3d.png"]),
        Spacer(1, 6),
        picture(figs["spec_temperature_3d.png"]),
        Spacer(1, 6),
        Paragraph("Draft claims, limited to what the example enables", s["H"]),
        Paragraph("1. A method comprising: determining a density of a numerical series; selecting a block width from a fixed ladder of widths according to that density; computing a damped series in a host language and in a graphics-processor kernel launched at the selected width; and accepting the result when the two computations agree with a reference implementation within a predetermined tolerance.", s["Body"]),
        Paragraph("2. The method of claim 1, wherein the damped series preserves a local mean and scales the deviation from that mean by an exponential of a local variance, and wherein the latent series used only for scoring is not supplied to the computation.", s["Body"]),
        Paragraph("3. The method of claim 1, wherein a sign bit of a floating-point value is inverted by an unsigned exclusive-or and the inverted value matches arithmetic negation.", s["Body"]),
        Paragraph("4. A system comprising four graphics processors, each executing the kernel of claim 1, and a host process that records per-processor time, power, and temperature.", s["Body"]),
        Paragraph("Matter not enabled by the example", s["H"]),
        Paragraph("The example does not enable a claim that heat per erased bit was measured, that an optical crystal performed the damping, that a physical quasicrystal localized a wave, that splitting this kernel across four processors is faster than one processor, or that a language model was shown to be more factual. A silence rule based on window variance, on this series, reached an F1 of 0.291 and is not claimed. An average-pool timing of 0.016 ms against a flux-map timing of 0.064 ms is recorded and is not claimed as a speed advantage.", s["Body"]),
    ]


def write_pdf(path: Path, story, title: str, label: str) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch, title=title, author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer(label), onLaterPages=footer(label))


def add_docx_from_blocks(path: Path, title: str, blocks: list[tuple[str, str]], images: list[Path]) -> None:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    document.add_heading(title, 0)
    for kind, text in blocks:
        if kind == "h":
            document.add_heading(text, level=1)
        else:
            document.add_paragraph(text)
    document.add_heading("Telemetry and 3D histograms", level=1)
    document.add_paragraph("Idle and load readings from the four RTX PRO 6000 boards, and the color histograms of error, Triton latency, power, and temperature.")
    for image in images:
        document.add_picture(str(image), width=Inches(6.2))
    document.save(path)


def tech_docx_blocks():
    return [
        ("h", "1. Scope"),
        ("p", "This specification describes the system that ran on 14 November 2026: one damping map executed by NumPy, Julia, JAX, and Triton on four NVIDIA RTX PRO 6000 Blackwell Server Edition processors. It does not specify a calorimeter, an optical crystal, or a fabricated quasicrystal."),
        ("h", "2. Hardware"),
        ("p", "Four processors, 97,887 MiB and a 600 W limit each. Idle power and temperature were approximately 30.5 W at 28 °C, 30.0 W at 29 °C, 29.9 W at 31 °C, and 30.4 W at 30 °C."),
        ("h", "3. Software"),
        ("p", "Python 3.12, PyTorch 2.14.1+cu130, Triton, JAX with the CUDA 13 plugin on cuda:0 through cuda:3, and Julia 1.13.4. The @sphota ladder selects block 128, 256, 512, or 1024 from the mean absolute instability. Density 0.793 selected block 256 and grid 64 at length 16,384."),
        ("h", "4. The map"),
        ("p", "Raw form multiplies the observation by exp(−λξ) with λ = 1 and ξ a window variance of width 9. Flux form keeps the local mean and scales only the deviation. The latent series is not an input. When both a truth tensor and a noise tensor are supplied, their paired sum and difference are averaged. A sign-bit kernel flips the float32 sign by an unsigned XOR of the high bit."),
        ("h", "5. Agreement"),
        ("p", "Julia and JAX each matched NumPy within 3.8e-6. Triton on each board matched within about 7.6e-6 on the stack check."),
        ("h", "6. Quality"),
        ("p", "Seed 11, length 8192. Observation MSE 6.910. Moving average 1.675. Raw map 0.774 at offset 0 and 18.17 at offset 5. Flux map 1.472 at both offsets. Flux variance remaining 0.117. The flux map meets a bar of 0.95 times the better of observation and moving average. The raw map meets it only at offset 0."),
        ("h", "7. Timing"),
        ("p", "Triton medians stayed near 0.009 to 0.011 ms from length 8192 to 1048576 on all four boards, under a 0.2379 ms budget. Sign-bit error was 0. Block 256 at one million elements took 0.0116 ms, faster than blocks 128, 512, and 1024 and faster than eager PyTorch at 0.0164 ms. One board finished 4194304 elements in 0.047 ms; four boards took 0.088 ms. A Python loop of 4096 elements took 1.040 ms; the fused GPU map took 0.021 ms. On another million-element vector the flux map took 0.064 ms and an average pool took 0.016 ms."),
        ("h", "8. Telemetry"),
        ("p", "Idle: GPU0 30.45 W / 28 °C, GPU1 29.95 W / 29 °C, GPU2 29.91 W / 31 °C, GPU3 30.44 W / 30 °C. Load means: 86.6, 86.3, 86.3, and 86.6 W. Peaks: 33, 33, 36, and 35 °C. Limit 600 W. Samples include a driver query between launches."),
        ("h", "9. Acceptance"),
        ("p", "A reproduction meets this specification when the three tiers match NumPy within 1e-3, the flux error meets the bar on both offsets, the sign-bit kernel matches negation, and a Triton median at length 10000 is under 0.2379 ms. The silence-gate F1 of 0.291 is not an acceptance item."),
    ]


def patent_docx_blocks():
    return [
        ("p", "Draft specification for counsel. Not a filed application and not a grant. Ceraph Corporation, 14 November 2026."),
        ("h", "Field"),
        ("p", "A computer-implemented method and system for damping a numerical series, in which a control program selects a launch width and a differentiable program and a fused graphics-processor kernel compute the same map."),
        ("h", "Background"),
        ("p", "A local average is the ordinary smoother. A kernel width is ordinarily chosen by hand. A notebook, a differentiable runtime, and a kernel language often diverge. The measured system keeps them on one formula and checks them against one reference."),
        ("h", "Summary"),
        ("p", "A host process selects one of four block widths from a density. Julia computes that width and the damped series. JAX computes the damped series on one or more graphics processors. Triton launches at the selected width. Agreement with NumPy is the test. A preferred mode preserves a local mean and scales only the deviation by an exponential of a local variance. Another mode inverts a sign bit when a paired noise value is already isolated."),
        ("h", "Example"),
        ("p", "On 14 November 2026 the method ran on four RTX PRO 6000 Blackwell processors. Density 0.793 selected block 256. Julia, JAX on four GPUs, and Triton matched NumPy within 1e-5. Flux-mode error was 1.472 against 1.675 for a moving average, at offsets 0 and 5. Raw-mode error was 0.774 at offset 0 and 18.17 at offset 5. Triton medians near 0.010 ms spanned lengths 8192 to 1048576. Sign-bit error was 0. Block 256 was the fastest width and beat an eager product. A light four-processor loop drew about 86 W and stayed at or below 36 °C."),
        ("h", "Draft claims limited to the example"),
        ("p", "1. A method comprising determining a density of a numerical series, selecting a block width from a fixed ladder according to that density, computing a damped series in a host language and in a graphics-processor kernel at that width, and accepting the result when both agree with a reference within a predetermined tolerance."),
        ("p", "2. The method of claim 1, wherein a local mean is preserved and the deviation is scaled by an exponential of a local variance, and the latent series used for scoring is not an input."),
        ("p", "3. The method of claim 1, wherein a floating-point sign bit is inverted by an unsigned exclusive-or and matches arithmetic negation."),
        ("p", "4. A system comprising four graphics processors each executing the kernel of claim 1, and a host that records time, power, and temperature."),
        ("h", "Not enabled by the example"),
        ("p", "No claim is made here that heat per erased bit was measured, that an optical crystal or a physical quasicrystal performed the map, that four processors were faster than one on this kernel, or that a language model was shown to be more factual. A silence rule on this series reached F1 0.291 and is not claimed. An average pool at 0.016 ms against the flux map at 0.064 ms is not claimed as a speed advantage."),
    ]


def main() -> None:
    figs = make_figures()
    images = [figs["spec_quality_3d.png"], figs["spec_triton_latency_3d.png"], figs["spec_power_3d.png"], figs["spec_temperature_3d.png"]]
    write_pdf(OUT / "phase4-technical-specification.pdf", tech_story(figs), "SRI technical specification", "Ceraph Corporation  ·  Technical specification  ·  14 Nov 2026")
    write_pdf(OUT / "phase4-patent-specification.pdf", patent_story(figs), "SRI patent specification draft", "Ceraph Corporation  ·  Draft patent specification  ·  14 Nov 2026")
    add_docx_from_blocks(OUT / "phase4-technical-specification.docx", "Technical specification — SRI Witness damping stack, 14 November 2026", tech_docx_blocks(), images)
    add_docx_from_blocks(OUT / "phase4-patent-specification.docx", "Draft specification — hybrid damping system, 14 November 2026", patent_docx_blocks(), images)
    print("ok")


if __name__ == "__main__":
    main()
