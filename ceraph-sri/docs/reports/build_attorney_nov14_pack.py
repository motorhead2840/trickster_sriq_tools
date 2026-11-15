"""Attorney pack: how-it-works flowcharts, and a detailed spec with telemetry and histograms."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Inches, Pt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    base.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=10.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
    base.add(ParagraphStyle("Cap", fontName="Times-Italic", fontSize=9, leading=12, textColor=colors.HexColor("#3d4754"), spaceBefore=2, spaceAfter=8))
    base.add(ParagraphStyle("Cell", fontName="Times-Roman", fontSize=8, leading=10.2, textColor=NAVY))
    base.add(ParagraphStyle("Head", fontName="Times-Bold", fontSize=8, leading=10.2, textColor=colors.white))
    return base


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#5c6773"))
    canvas.drawString(0.65 * inch, 0.38 * inch, "Ceraph Corporation  ·  Attorney filing pack  ·  14 Nov 2026")
    canvas.drawRightString(letter[0] - 0.65 * inch, 0.38 * inch, str(doc.page))
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


def flow(name: str, title: str, nodes: list[tuple[str, str]], edges: list[tuple[int, int]]) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor="#f7f4ee")
    ax.set_xlim(0, len(nodes) * 2.2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    palette = ["#1c2430", "#245c4a", "#8a4b08", "#243f73", "#6b3fa0", "#7a2e2e"]
    centers = []
    for i, (head, body) in enumerate(nodes):
        x = 0.25 + i * 2.2
        box = FancyBboxPatch((x, 0.35), 1.9, 1.3, boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor=palette[i % len(palette)], edgecolor="none")
        ax.add_patch(box)
        ax.text(x + 0.95, 1.25, head, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(x + 0.95, 0.8, body, ha="center", va="center", color="#f4f7fb", fontsize=7.5, wrap=True)
        centers.append((x + 1.9, 1.0))
    for a, b in edges:
        ax.add_patch(FancyArrowPatch((centers[a][0], 1.0), (centers[b][0] - 1.65, 1.0),
                                     arrowstyle="-|>", mutation_scale=12, color="#1c2430", lw=1.4))
    fig.suptitle(title, color="#1c2430", fontsize=12)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def histograms() -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = {}

    def save(name, title, draw):
        fig = plt.figure(figsize=(10.2, 6.0), facecolor="#10141c")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#10141c")
        draw(ax)
        ax.tick_params(colors="#e8edf5")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color("#e8edf5")
        fig.suptitle(title, color="#f4f7fb")
        path = FIG / name
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths[name] = path

    def quality(ax):
        from matplotlib.patches import Patch
        labels = ["raw series", "moving average", "raw Abhaya", "flux Abhaya"]
        colors_ = ["#8aa0b8", "#e09a55", "#7d8cff", "#3ecf8e"]
        for col, heights in enumerate(([6.910, 1.675, 0.774, 1.472], [6.910, 1.675, 18.17, 1.472])):
            ax.bar3d(np.arange(4), np.full(4, col), np.zeros(4), 0.45, 0.45, heights, color=colors_, shade=True)
        ax.set_xticks(np.arange(4) + 0.2)
        ax.set_xticklabels(labels, color="#e8edf5", rotation=15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["signal near 0", "signal shifted by 5"], color="#e8edf5")
        ax.set_zlabel("Error (lower is better)")
        ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors_, labels)], loc="upper left", fontsize=8)
        ax.view_init(elev=22, azim=-58)

    def latency(ax):
        from matplotlib.patches import Patch
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
        ax.set_xticklabels(["8,192", "10,000", "65,536", "1,048,576"], color="#e8edf5", rotation=15)
        ax.set_ylabel("Board")
        ax.set_zlabel("Milliseconds")
        ax.legend(handles=[Patch(facecolor=palette[i], label=f"Board {i}") for i in range(4)], loc="upper left", fontsize=8)
        ax.view_init(elev=24, azim=-60)

    def power(ax):
        from matplotlib.patches import Patch
        idle = [30.45, 29.95, 29.91, 30.44]
        load = [86.59, 86.30, 86.28, 86.55]
        ax.bar3d(np.arange(4) - 0.15, np.zeros(4), np.zeros(4), 0.3, 0.4, idle, color="#6ec8ff", shade=True)
        ax.bar3d(np.arange(4) + 0.15, np.ones(4), np.zeros(4), 0.3, 0.4, load, color="#ff7a59", shade=True)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["Board 0", "Board 1", "Board 2", "Board 3"], color="#e8edf5")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["idle", "working"], color="#e8edf5")
        ax.set_zlabel("Watts")
        ax.legend(handles=[Patch(facecolor="#6ec8ff", label="Idle, about 30 W"), Patch(facecolor="#ff7a59", label="Working, about 86 W")], loc="upper left", fontsize=8)
        ax.view_init(elev=22, azim=-58)

    def temperature(ax):
        from matplotlib.patches import Patch
        idle_c = [28, 29, 31, 30]
        load_c = [33, 33, 36, 35]
        ax.bar3d(np.arange(4) - 0.15, np.zeros(4), np.zeros(4), 0.3, 0.4, idle_c, color="#3ecf8e", shade=True)
        ax.bar3d(np.arange(4) + 0.15, np.ones(4), np.zeros(4), 0.3, 0.4, load_c, color="#c77dff", shade=True)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(["Board 0", "Board 1", "Board 2", "Board 3"], color="#e8edf5")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["idle", "hottest sample"], color="#e8edf5")
        ax.set_zlabel("Degrees Celsius")
        ax.legend(handles=[Patch(facecolor="#3ecf8e", label="Idle, 28–31 °C"), Patch(facecolor="#c77dff", label="Working peak, 33–36 °C")], loc="upper left", fontsize=8)
        ax.view_init(elev=22, azim=-58)

    save("attorney_quality_3d.png", "How much error was left  ·  lower bars are better", quality)
    save("attorney_latency_3d.png", "How long the GPU kernel took  ·  four boards", latency)
    save("attorney_power_3d.png", "Electricity drawn  ·  idle versus working", power)
    save("attorney_temperature_3d.png", "Temperature  ·  idle versus the hottest working sample", temperature)
    return paths


def pic(path: Path):
    return Image(str(path), width=6.3 * inch, height=3.7 * inch)


def how_pdf(flows: dict[str, Path]) -> None:
    s = styles()
    story = [
        Paragraph("How the technology works", s["Cover"]),
        Paragraph("A simple explanation with flowcharts, for counsel. Ceraph Corporation · 14 November 2026. This is the picture of the invention. The measured numbers are in the detailed specification.", s["Sub"]),
        Paragraph("In ordinary words", s["H"]),
        Paragraph("A computer often finishes a messy answer and only afterwards tries to clean it up. That late cleanup is slow, and it cannot undo work that already went wrong. This invention cleans while the answer is being formed. It keeps what is steady and reduces the disturbance. It can do that without being handed a secret copy of the true signal.", s["Body"]),
        Paragraph("Flowchart 1. The old path and the new path", s["H"]),
        pic(flows["flow_old_new.png"]),
        Paragraph("Left to right: a person asks a question. In the usual path the machine finishes, then a separate filter judges the result. In this path the scores are steadied before the next piece of the answer is chosen. The person still receives an answer. The cleanup happened inside the work, not after it.", s["Cap"]),
        Paragraph("Flowchart 2. Keep the center, quiet the wiggle", s["H"]),
        pic(flows["flow_flux.png"]),
        Paragraph("The machine looks at a short neighborhood of the noisy series. It remembers the local center. It measures how wild the neighborhood is. It shrinks only the departure from the center. The clean hidden series is used later, only to grade the result. It is not an input. On the test, this left less error than a moving average.", s["Cap"]),
        Paragraph("Flowchart 3. Three languages, one formula", s["H"]),
        pic(flows["flow_stack.png"]),
        Paragraph("A density number picks a block width. Julia computes that width and the damped series. JAX computes the same series on the graphics chips. Triton runs the fast kernel at that width. All three are accepted only when they match a simple reference program. On 14 November they did, on four RTX PRO 6000 Blackwell boards, and Julia’s width was also the fastest width tried.", s["Cap"]),
        Paragraph("Flowchart 4. What the operator records", s["H"]),
        pic(flows["flow_telemetry.png"]),
        Paragraph("Each board reports its name, watts, temperature, and how long the kernel took. The host writes those readings down. Idle was about 30 watts and 30 °C. Working was about 86 watts and at most 36 °C. The boards are rated at 600 watts. The invention’s example ran in a quiet band of that hardware.", s["Cap"]),
        Paragraph("What counsel can protect from this picture", s["H"]),
        Paragraph("Protect the in-process cleanup, the version that keeps a local center and damps only the deviation, the rule that picks a block width from a density, and the requirement that the host program and the graphics kernel agree. Protect the sign-bit flip when a paired noise value is already in hand. Describe an optical crystal, a heat meter, and a physical quasicrystal as future embodiments. They are not in these flowcharts because they were not in the laboratory.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(OUT / "phase4-how-it-works.pdf"), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch, title="How the technology works", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def detailed_pdf(hists: dict[str, Path]) -> None:
    s = styles()
    story = [
        Paragraph("Detailed specification of the measured invention", s["Cover"]),
        Paragraph("Core mathematics, technical description, and the full test record of 14 November 2026. Ceraph Corporation. Draft for counsel. Not a filed application.", s["Sub"]),
        Paragraph("1. Core mathematics", s["H"]),
        Paragraph("Let y be an observed series. A window of width 9 has a local mean μ and a local variance Ξ. The damping coefficient is λ = 1. The raw map is A = y · exp(−λ Ξ). It multiplies every sample, so a large carrier is pulled toward zero. The flux map is A = μ + (y − μ) · exp(−λ Ξ). It keeps the center and scales only the deviation. Neither map receives the latent series. When a truth vector v and a noise vector ξ are both supplied, the paired identity is ½((v+ξ)+(v−ξ)) = v. A sign-bit map replaces ξ by the same bits with the IEEE-754 sign flipped, which for finite float32 values is arithmetic negation.", s["Body"]),
        Paragraph("A density is the mean of the absolute values of the instability series. The launch ladder is 128 if the density is below 0.5, 256 if below 1, 512 if below 2, and 1024 otherwise. The grid is the length divided by the block, rounded up. On the agreement test the density was 0.793, so the block was 256 and the grid was 64 for length 16,384.", s["Body"]),
        Paragraph("An earlier, separate result remains part of the mathematical bedrock. On a preregistered dual-mode sensing model, the deployable estimator matches the Wiener filter and reduced mean error 12.05 percent versus a single-mode Wiener filter. The pass bar was 8 percent. That path uses two readings and does not require the hidden noise.", s["Body"]),
        Paragraph("2. Technical embodiment", s["H"]),
        Paragraph("Host: Python 3.12, NumPy as the reference, Julia 1.13.4 for the @sphota schedule and a binary float32 damping check. Differentiable tier: JAX with the CUDA 13 plugin, backend gpu, devices cuda:0 through cuda:3. Kernel tier: Triton on PyTorch 2.14.1+cu130. Hardware: four NVIDIA RTX PRO 6000 Blackwell Server Edition boards, 97,887 MiB and 600 W each. The kernel loads a tile of the series and of the instability, applies the exponential product, and stores the result. The tile is staged for that program. The reads and the write use device memory.", s["Body"]),
        Paragraph("3. Tests, telemetry, and what they mean", s["H"]),
        Paragraph("This section collects the tests that bear on the invention, then the 14 November session in full. Each histogram has a legend. The sentence under it says what a non-specialist should see.", s["Body"]),
        Paragraph("3.1 Earlier tests that the invention stands on", s["H"]),
        Paragraph("Regression tests on ordinary processors lock the published numerical results of the unveiling rounds, the observable-only benchmark, and conjectures C1 through C4. Phase 3, also on a processor, passed the dual-mode comparison above. An earlier single-board Blackwell run of a heavier kernel finished between 7 and 149 times sooner than the processor in that same machine, near 105 W and 30 °C. On that same board an ordinary average pool was faster than that heavier kernel. An L40S session, a different and lighter synthetic job, finished about 41 times sooner than its neighboring processor at a million elements, near 85 W and 36 °C, against a 350 W cap. Those runs taught the project to separate an accuracy win from a speed win. The 14 November session is the one that puts the hybrid stack and the flux map on four boards.", s["Body"]),
        Paragraph("3.2 Agreement of the stack", s["H"]),
        Paragraph("Length 16,384, seed 11, λ = 1. Julia matched NumPy within 3.8e-6 and chose block 256, the same block as the Python ladder. After the CUDA 13 plugin was installed and the notebook restarted, JAX’s backend was gpu on cuda:0 through cuda:3 and the map matched NumPy within 3.8e-6. The first JAX import that day had been a CPU build. Triton on each of the four boards matched within about 7.6e-6, with medians near 0.012 ms. Meaning: the invention is not three stories. It is one formula checked three ways, and the graphics chips did the JAX and Triton parts.", s["Body"]),
        Paragraph("3.3 Error histogram", s["H"]),
        pic(hists["attorney_quality_3d.png"]),
        Paragraph("Legend: gray is the noisy series, amber is a moving average, blue is the raw damping map, green is the flux map. The front row is a signal near zero. The back row is the same noise on a signal shifted by 5. Lower bars are better. The green bars are 1.47 in both rows. The amber bars are 1.68. The gray bars are 6.91. The blue bar is excellent in front (0.77) and poor in back (18.17), because multiplying the whole series drags a large signal toward zero. Meaning: the invention’s preferred mode beats a standard smoother without seeing the answer, and it still does so when the signal moves. That is why the flux mode is the one to claim.", s["Cap"]),
        table(
            ["Estimator", "Offset 0", "Offset 5", "What it means"],
            [
                ["Noisy series", "6.910", "6.910", "The problem, uncleaned"],
                ["Moving average", "1.675", "1.675", "A current method"],
                ["Raw map", "0.774", "18.17", "Wins only near zero"],
                ["Flux map", "1.472", "1.472", "Wins in both rows"],
            ],
            [1.5 * inch, 1.2 * inch, 1.2 * inch, 2.6 * inch],
        ),
        Spacer(1, 6),
        Paragraph("Flux variance remaining was 0.117, under a half-variance bar. Eight hundred twelve spikes were in the series. A silence rule that flags high window variance reached a best F1 of 0.291 at threshold 8 (precision 0.199, recall 0.544). That rule is not good enough to claim. The error result does not depend on it.", s["Body"]),
        Paragraph("3.4 Latency histogram", s["H"]),
        pic(hists["attorney_latency_3d.png"]),
        Paragraph("Legend: one color per board. The floor labels are the length of the series. The height is milliseconds. Every bar is about 0.010, from 8,192 elements through 1,048,576, on every board, and every bar is under the 0.2379 ms budget carried from the March notebooks. The March single-board medians were 0.0082 ms and 0.0062 ms. Tonight is the same neighborhood, repeated four times. Meaning: for this kernel the time stays in a hundredth of a millisecond while the job grows by more than a hundredfold. The launch cost dominates the arithmetic. A heavier program can still slow down. This histogram is the timing example for this kernel, and it is why the in-process path is in a different time band from a filter that waits until a full answer exists.", s["Cap"]),
        Paragraph("Sign-bit XOR medians were 0.0104, 0.0112, 0.0093, and 0.0096 ms, with error 0 on every board. At one million elements, Triton widths 128, 256, 512, and 1024 took 0.0147, 0.0116, 0.0126, and 0.0129 ms. Width 256, the one the density rule chose, was the fastest, and it beat eager PyTorch at 0.0164 ms. Float32, float16, and bfloat16 medians of a related one-line map were 0.0217, 0.0202, and 0.0201 ms. A Python loop of 4,096 elements took 1.040 ms. The fused GPU map took 0.021 ms. One board finished 4,194,304 elements in 0.047 ms. Four boards together took 0.088 ms, so splitting this small kernel is not a speed claim. A separate average pool took 0.016 ms against the flux map at 0.064 ms. The pool wins that clock. The flux map wins the error table. Claim the error. Do not claim the pool’s clock.", s["Body"]),
        Paragraph("A one-line PyTorch map stayed between 0.017 and 0.019 ms from length 16,384 to 1,048,576. Log-log slopes were −0.0095, 0.0004, −0.0008, and 0.0043. Paired cancellation, both tensors supplied, left a residual of 1.94e-7. A Fibonacci window of 13 summed to weight 2.61 and did not reduce this particular noise energy (filtered energy was 1.56 times the noise). Circuit A printed 1.09 and circuit B sat at its cap of 50. Those last two are recorded behavior, not the claims to lead with.", s["Body"]),
        Paragraph("3.5 Power and temperature histograms", s["H"]),
        pic(hists["attorney_power_3d.png"]),
        Paragraph("Legend: blue is idle, coral is the working loop. Heights are watts. Idle is about 30 W on each board. Working is about 86 W. The boards are rated at 600 W. Meaning: the example did its work in a small slice of the machine’s electrical budget. It is not a measurement of a data-center bill, and it is not a claim that every heavier job will stay at 86 W.", s["Cap"]),
        pic(hists["attorney_temperature_3d.png"]),
        Paragraph("Legend: green is idle temperature, purple is the hottest sample while working. Idle was 28, 29, 31, and 30 °C. Working peaks were 33, 33, 36, and 35 °C. Meaning: the boards stayed cool during this loop. A heat-per-bit experiment was not performed. The histogram shows thermal headroom, which is the honest thermal claim.", s["Cap"]),
        table(
            ["Board", "Idle watts", "Idle °C", "Working watts", "Hottest °C"],
            [
                ["0", "30.45", "28", "86.6", "33"],
                ["1", "29.95", "29", "86.3", "33"],
                ["2", "29.91", "31", "86.3", "36"],
                ["3", "30.44", "30", "86.6", "35"],
            ],
            [1.1 * inch, 1.3 * inch, 1.2 * inch, 1.5 * inch, 1.3 * inch],
        ),
        Paragraph("4. Why these tests make the invention better", s["H"]),
        Paragraph("Better than a moving average, on the locked series, because the error is lower and the clean answer was not an input. Better than three disconnected software tools, because Julia, JAX, and Triton had to match one reference or the stack was not accepted. Better than a hand-picked kernel width, because the density rule selected the width that also measured fastest. Better than a late cleanup that waits for a finished answer, because this kernel’s own time sits near 0.010 ms, which is the in-process example counsel can point to. Better than an approximate “opposite” of the noise, because the sign flip was exact. Quieter than the rating plate of the boards it ran on, for this loop, at about one seventh of 600 W.", s["Body"]),
        Paragraph("5. Conclusion", s["H"]),
        Paragraph("The enabled invention is the hybrid damping method and system measured above: density-chosen block width, flux-mode damping without the latent series, sign-bit conjugation when a paired value is in hand, and acceptance by agreement with a reference, demonstrated on four RTX PRO 6000 Blackwell processors with the telemetry in this section. Future embodiments, not enabled here, include a heat-per-bit measurement, an optical crystal, and a physical quasicrystal. A labeled question exam was not run. The silence gate did not pass. Counsel should draft from sections 1, 2, and 4, and use section 3 as the example, including the comparisons that did not win.", s["Body"]),
    ]
    doc = SimpleDocTemplate(str(OUT / "phase4-attorney-detailed-specification.pdf"), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch, topMargin=0.6 * inch, bottomMargin=0.58 * inch, title="Detailed specification for counsel", author="Ceraph Corporation")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def how_docx(flows: dict[str, Path]) -> None:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    document.add_heading("How the technology works", 0)
    document.add_paragraph("A simple explanation with flowcharts, for counsel. Ceraph Corporation, 14 November 2026.")
    document.add_paragraph("A computer often finishes a messy answer and only afterwards tries to clean it up. This invention cleans while the answer is being formed. It keeps what is steady and reduces the disturbance, without being handed a secret copy of the true signal.")
    captions = [
        ("flow_old_new.png", "Flowchart 1. Usual path: finish, then filter. This path: steady the scores before the next piece of the answer is chosen."),
        ("flow_flux.png", "Flowchart 2. Remember the local center, measure how wild the neighborhood is, shrink only the departure from the center. The hidden clean series is used only to grade the result."),
        ("flow_stack.png", "Flowchart 3. A density picks a block width. Julia, JAX, and Triton each compute the map. They are accepted only when they match a reference. On 14 November they did, on four Blackwell boards."),
        ("flow_telemetry.png", "Flowchart 4. Each board reports watts, temperature, and kernel time. Idle was about 30 watts and 30 °C. Working was about 86 watts and at most 36 °C, on 600 watt boards."),
    ]
    for name, caption in captions:
        document.add_picture(str(flows[name]), width=Inches(6.4))
        document.add_paragraph(caption)
    document.add_paragraph("Protect the in-process cleanup, the center-keeping flux mode, the density-chosen block width, and agreement between the host program and the graphics kernel. Describe an optical crystal, a heat meter, and a physical quasicrystal as future embodiments.")
    document.save(OUT / "phase4-how-it-works.docx")


def detailed_docx(hists: dict[str, Path]) -> None:
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    document.add_heading("Detailed specification of the measured invention", 0)
    document.add_paragraph("Ceraph Corporation, 14 November 2026. Draft for counsel. Not a filed application.")
    sections = [
        ("1. Core mathematics", "Raw map: observation times exp(−λ times window variance), λ = 1, window 9. Flux map: keep the local mean and scale only the deviation. The latent series is not an input. Paired identity, when both truth and noise are supplied: one half of (truth+noise) plus (truth−noise) equals truth. Sign-bit flip of a finite float32 value matches arithmetic negation. Density is the mean absolute instability. Block ladder: 128, 256, 512, 1024 at thresholds 0.5, 1, and 2. The dual-mode sensing result remains: 12.05 percent lower error than single-mode Wiener, bar 8 percent."),
        ("2. Technical embodiment", "Python 3.12, NumPy reference, Julia 1.13.4, JAX CUDA 13 on cuda:0 through cuda:3, Triton on PyTorch 2.14.1+cu130. Four RTX PRO 6000 Blackwell boards, 97,887 MiB, 600 W. The kernel loads a tile, applies the product, and stores it."),
        ("3. Tests and telemetry", "Regression tests lock the earlier published numbers. Phase 3 passed the dual-mode comparison. A prior one-board Blackwell kernel was 7 to 149 times sooner than its neighboring processor, near 105 W and 30 °C, and lost a pure speed test to an average pool. The 14 November session is the hybrid-stack example. Julia and JAX matched NumPy within 3.8e-6. Triton matched within about 7.6e-6 at block 256. Flux MSE 1.472 versus moving average 1.675 at offsets 0 and 5. Raw map 0.774 and 18.17. Silence-gate best F1 0.291, not claimed. Triton medians near 0.010 ms from length 8192 to 1048576, under 0.2379 ms. Sign-bit error 0. Block 256 at 0.0116 ms beat 128, 512, 1024 and eager PyTorch at 0.0164 ms. One board 0.047 ms versus four boards 0.088 ms at 4194304 elements. Average pool 0.016 ms versus flux map 0.064 ms on a separate timing vector. Python loop 1.040 ms versus fused GPU 0.021 ms at length 4096. Paired residual 1.94e-7."),
        ("Why the tests make the invention better", "Lower error than a moving average without the latent series. Three runtimes forced to agree. The density rule selected the fastest measured width. The kernel’s own time is near 0.010 ms, which is the in-process example. The sign flip is exact. The loop used about 86 W and stayed at or below 36 °C on 600 W boards. Do not claim the average pool’s clock, the four-way split, the silence gate, a heat-per-bit result, an optical crystal, or a physical quasicrystal."),
        ("4. Conclusion", "Draft from the flux mode, the density-chosen width, host-and-kernel agreement, and sign-bit conjugation, using section 3 as the example, including the comparisons that did not win."),
    ]
    for title, text in sections:
        document.add_heading(title, level=1)
        document.add_paragraph(text)
    document.add_heading("Histograms", level=1)
    captions = [
        ("attorney_quality_3d.png", "Error histogram. Gray: noisy series. Amber: moving average. Blue: raw map. Green: flux map. Front row near zero, back row shifted by 5. Lower is better. Green is 1.47 in both rows."),
        ("attorney_latency_3d.png", "Latency histogram. One color per board. Heights are milliseconds. Every bar is about 0.010 from 8,192 to 1,048,576 elements."),
        ("attorney_power_3d.png", "Power histogram. Blue idle about 30 W. Coral working about 86 W. Boards are rated at 600 W."),
        ("attorney_temperature_3d.png", "Temperature histogram. Green idle 28–31 °C. Purple working peaks 33–36 °C."),
    ]
    for name, caption in captions:
        document.add_picture(str(hists[name]), width=Inches(6.2))
        document.add_paragraph(caption)
    document.save(OUT / "phase4-attorney-detailed-specification.docx")


def main() -> None:
    flows = {
        "flow_old_new.png": flow(
            "flow_old_new.png",
            "Flowchart 1.  Finish-then-filter, or steady the work as it happens",
            [("Ask", "A person\nsends a question"), ("Usual", "Finish the\nanswer, then filter"), ("This path", "Steady the scores\nbefore the next step"), ("Answer", "The person\nreceives the result")],
            [(0, 1), (1, 2), (2, 3)],
        ),
        "flow_flux.png": flow(
            "flow_flux.png",
            "Flowchart 2.  Keep the center, quiet only the wiggle",
            [("Series", "Noisy samples\nonly"), ("Window", "Local center\nand wildness"), ("Dampen", "Shrink the\ndeparture"), ("Grade", "Hidden answer\nused only to score")],
            [(0, 1), (1, 2), (2, 3)],
        ),
        "flow_stack.png": flow(
            "flow_stack.png",
            "Flowchart 3.  Three languages must agree",
            [("Density", "Pick a\nblock width"), ("Julia", "Same width,\nsame map"), ("JAX", "Same map\non the chips"), ("Triton", "Fast kernel\nat that width")],
            [(0, 1), (1, 2), (2, 3)],
        ),
        "flow_telemetry.png": flow(
            "flow_telemetry.png",
            "Flowchart 4.  What gets written down",
            [("Boards", "Four RTX\nPRO 6000s"), ("Run", "Kernel on\neach board"), ("Read", "Watts, heat,\nand time"), ("Record", "Idle ~30 W\nWork ~86 W")],
            [(0, 1), (1, 2), (2, 3)],
        ),
    }
    hists = histograms()
    how_pdf(flows)
    detailed_pdf(hists)
    how_docx(flows)
    detailed_docx(hists)
    print("ok")


if __name__ == "__main__":
    main()
