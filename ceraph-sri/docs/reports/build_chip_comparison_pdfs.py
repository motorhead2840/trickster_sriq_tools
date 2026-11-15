"""Build two evidence-bounded PDFs: postulated chip vs simulations to date.

The documents do not assert unmeasured power, thermal, or cross-chip speed claims.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#1c2430")
RULE = colors.HexColor("#c5cdd8")
ROW = colors.HexColor("#f4f6f8")
WARN = colors.HexColor("#6b3a2a")


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="Cover", fontName="Times-Bold", fontSize=16, leading=20,
        textColor=NAVY, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="Sub", fontName="Times-Italic", fontSize=10, leading=13,
        textColor=colors.HexColor("#3d4754"), spaceAfter=12,
    ))
    base.add(ParagraphStyle(
        name="H", fontName="Times-Bold", fontSize=12, leading=15,
        textColor=NAVY, spaceBefore=12, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="Body", fontName="Times-Roman", fontSize=10, leading=13.5,
        alignment=TA_JUSTIFY, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="Small", fontName="Times-Roman", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#333a44"), spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="Cell", fontName="Times-Roman", fontSize=8, leading=10.5,
        textColor=NAVY,
    ))
    base.add(ParagraphStyle(
        name="CellHead", fontName="Times-Bold", fontSize=8, leading=10.5,
        textColor=colors.white,
    ))
    return base


def _table(headers, rows, widths):
    styles = _styles()
    data = [[Paragraph(h, styles["CellHead"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(cell), styles["Cell"]) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(0.75 * inch, 0.55 * inch, letter[0] - 0.75 * inch, 0.55 * inch)
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#5c6773"))
    canvas.drawString(0.75 * inch, 0.38 * inch, "Ceraph Corporation  ·  Internal evidence note  ·  Not a power or thermal claim")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.38 * inch, str(doc.page))
    canvas.restoreState()


def build_detailed(path: Path) -> None:
    styles = _styles()
    story = []
    story.append(Paragraph("Postulated chip and simulations to date", styles["Cover"]))
    story.append(Paragraph(
        "An evidence comparison for Ceraph Corporation. November 2026. "
        "This note uses committed run records. It does not add a power, thermal, or data-center cost model.",
        styles["Sub"],
    ))
    story.append(Paragraph("1. How to read this note", styles["H"]))
    story.append(Paragraph(
        "The architecture is described, in the project source material, as a hardware-native design "
        "that would keep a large computation stable at a cost that does not grow in the usual way "
        "with problem size, and that would damp unwanted variation without a large energy bill. "
        "That description is the postulated chip. The repository is the record of what has been "
        "simulated. The two are not the same object. A data-center decision needs the second.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "Nothing in the committed GPU run, the classical sensing runs, or the later campaign notes "
        "measures watts, joules on a die, inlet temperature, or a power bill. Where this note says "
        "a saving is not shown, that is a gap in the evidence, not a hidden result.",
        styles["Body"],
    ))

    story.append(Paragraph("2. What was actually run", styles["H"]))
    story.append(Paragraph(
        "On 7 November 2026 a JAX kernel was executed on an NVIDIA RTX PRO 6000 Blackwell Server Edition. "
        "The record in sri_gpu_run.json uses float32, seed 108, one vector length (16,384), 25 cycles, "
        "and 10 synchronized repeats after compilation and warm-up. The median synchronized kernel time "
        "was 0.08884 ms. The largest projection error, with the clean vector and the injected noise both "
        "supplied to the calculation, was 1.33049 × 10⁻¹³. The timing is one kernel configuration. "
        "It is not end-to-end inference latency, and it is not a comparison with another GPU.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "The same record, and the README that quotes it, states that the software still builds and "
        "reduces length-N arrays, so the work grows with N. A single length cannot establish constant-time "
        "behavior. Separate classical simulations compare an observable-only estimator with simple filters "
        "on synthetic series, and score a stability statistic on a 100-item multiple-choice proxy. "
        "Campaign notes in the repository mark physical circumvention of the Landauer bound as not proven, "
        "and mark a tensor-memory constant-time path as not instrumented. Those notes also warn that "
        "published accelerator figures from other groups must not be pasted in as Ceraph measurements.",
        styles["Body"],
    ))

    story.append(Paragraph("3. Claim against record", styles["H"]))
    story.append(_table(
        ["Postulated chip", "What the simulations show", "Data-center reading"],
        [
            ["Faster than a conventional processor at the same job",
             "One Blackwell kernel, median 0.08884 ms, one length. No second device, no locked-clock comparison.",
             "Not a procurement result. Host timing only."],
            ["Cost stays flat as the problem grows",
             "Software work grows with vector length. The campaign verdict on constant-time software is negative.",
             "Do not size a cluster on a flat-cost assumption."],
            ["Lower energy per useful result",
             "No joule meter, no power rail, no watt-hour total. Landauer appears as a bound check, not a measured bypass.",
             "No basis yet for a power-budget cut."],
            ["Better thermal behavior in a rack",
             "A damped variance near 10⁻⁴¹ is the float32 floor of a synthetic run, not heat leaving a package.",
             "No inlet, diode, or coolant measurement exists."],
            ["More reliable outputs under noise",
             "Known-noise arithmetic matches to about 10⁻¹³. On a synthetic denoising test a moving average won. A small language proxy scored the project's statistic below chance and below model confidence.",
             "Not a reliability product result."],
        ],
        [1.7 * inch, 2.7 * inch, 2.3 * inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4. Speed", styles["H"]))
    story.append(Paragraph(
        "The Blackwell number is real and narrow. Ten repeats of one batched kernel, compilation excluded, "
        "landed at a median of 0.08884 ms. That shows the configured kernel ran on that card. It does not "
        "show that the postulated chip is faster than a CPU server, a Hopper GPU, or an ASIC built for the "
        "same job. A fair speed claim needs the same binary, the same sizes, named comparison hardware, "
        "and a clock policy. The host-timing cell added for external discussion is a synthetic map-and-reduce "
        "used to name the device. It is not the research kernel, and a ratio against the same machine's CPU "
        "is only a same-host check.",
        styles["Body"],
    ))

    story.append(Paragraph("5. Power, thermals, and the data-center bill", styles["H"]))
    story.append(Paragraph(
        "A data center cuts its power bill only if a workload uses fewer watt-hours for the same accepted "
        "output, or if it finishes in less time at the same power so the facility can run more work. "
        "Neither quantity is in the repository. There is no current sensor, no energy counter, no "
        "comparison of idle and loaded board power, and no facility model (PUE, chilled water, or rack "
        "airflow). Thermal efficiency is a ratio of heat removed to heat produced, or a temperature rise "
        "per watt. The simulations do not record temperature.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "The very small variance after the configured damping is a numerical floor. The project README "
        "says so: it is not physical cooling, and the stability statistic in that run is not held inside "
        "a unit interval by the implemented formula. Treating that floor as evidence of a cooler chip "
        "would reverse the record.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "The Landauer bound is a physics lower bound on the energy of erasing a bit. Tabulating the bound "
        "does not show that a device has reached it or gone around it. The campaign code states that "
        "physical circumvention remains unproven until there is calorimetry on silicon. That is the "
        "correct status for any conversation about power cost.",
        styles["Body"],
    ))

    story.append(Paragraph("6. Where the simulations are genuinely useful", styles["H"]))
    story.append(Paragraph(
        "They are useful as a bound on language. The paired calculation reproduces an identity when both "
        "the clean input and the noise are supplied. That is a software check, and the error is at the "
        "level of float32 roundoff. The observable-only path is a classical estimator for a stated Gaussian "
        "model with known parameters. Closed-form checks in that model are reproducible on CPU. Conjecture "
        "labels in the repository mean the project's own pass criteria were met inside those models. "
        "They are not a chip measurement and not a customer result.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "The negative results are part of the value. A moving-average baseline beat the paired method on "
        "the synthetic series. The language-proxy statistic did not beat confidence (about 0.47 against "
        "about 0.62 on the committed 100-item sample). A damping setting that drives the vector to the "
        "numeric floor removes signal together with noise. A later chip proposal has to survive those "
        "findings rather than average them away.",
        styles["Body"],
    ))

    story.append(Paragraph("7. What a power and thermal case would require", styles["H"]))
    story.append(Paragraph(
        "Before anyone tells an operator that this design cuts facility cost, the following have to exist "
        "as paired artifacts, not as a narrative:",
        styles["Body"],
    ))
    items = [
        "Board power and energy for the real kernel and for a named baseline, same host, same sizes, idle subtracted or separately reported.",
        "A temperature record (package or board) over the same interval, so thermal efficiency is a measured rise per watt, not a small residual in a float.",
        "At least two vector lengths, so any claim that cost does not grow can be checked against the bytes moved.",
        "A second implementation target only if the goal is a cross-chip ranking. One Blackwell session cannot supply that ranking.",
        "An output quality metric defined before the run. Speed at a worse answer is not a data-center saving.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(item, styles["Body"]), leftIndent=12) for item in items],
        bulletType="1",
        start="1",
    ))
    story.append(Paragraph(
        "Until those artifacts exist, the honest comparison is: the postulated chip is a design hypothesis; "
        "the simulations are a classical and single-GPU record; power, heat, and facility cost are unmeasured. "
        "The Blackwell median is a timing of one kernel. It is not a reason to revise a data-center power budget.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "Sources: sri_gpu_run.json, README.md, docs/vc-feasibility-report.md, docs/technical-foundation.md, "
        "and the campaign notes that mark Landauer circumvention and constant-time silicon paths as unproven. "
        "This PDF does not restate unpublished update rules.",
        styles["Small"],
    ))

    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Postulated chip and simulations to date",
        author="Ceraph Corporation",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def build_brief(path: Path) -> None:
    styles = _styles()
    story = []
    story.append(Paragraph("A plain-language note on the chip idea and the tests so far", styles["Cover"]))
    story.append(Paragraph(
        "Ceraph Corporation  ·  November 2026  ·  For readers who are not implementing the method",
        styles["Sub"],
    ))
    story.append(Paragraph(
        "There is a design on paper for a future chip, and there is a set of computer tests. "
        "This note says what those tests support. It does not say that a data center would use less "
        "electricity or run cooler. That has not been measured.",
        styles["Body"],
    ))
    story.append(Paragraph("The idea and the tests", styles["H"]))
    story.append(Paragraph(
        "The design idea is that a specialized chip could do a hard job quickly and with little wasted energy. "
        "The tests are ordinary programs. One of them ran on a current NVIDIA Blackwell server graphics card. "
        "Others ran on a normal processor and checked arithmetic. None of them is the chip. None of them "
        "includes a power meter.",
        styles["Body"],
    ))
    story.append(Paragraph("What the graphics-card test showed", styles["H"]))
    story.append(Paragraph(
        "On 7 November 2026 the card finished one prepared calculation in about 0.089 thousandths of a second, "
        "in the middle of ten repeats, at one problem size. The arithmetic matched the inputs it was given "
        "to a very small rounding error. The test used a made-up signal and made-up noise, and both were "
        "handed to the program. It does not show the machine discovering an unknown answer. It also used "
        "only one size, so it does not show that a bigger job would take the same time.",
        styles["Body"],
    ))
    story.append(Paragraph("Speed, power, and heat", styles["H"]))
    story.append(_table(
        ["Question a buyer would ask", "Answer from the tests"],
        [
            ["Is it faster than other chips?",
             "Not shown. Only one card was timed, at one size. There is no head-to-head result."],
            ["Will the electric bill fall?",
             "Not shown. No watts or energy were recorded."],
            ["Will the racks run cooler?",
             "Not shown. No temperature was recorded. A tiny leftover in the arithmetic is not heat leaving the chip."],
            ["Does the software stay cheap as the job grows?",
             "The programs still do more work as the input gets longer. A flat cost is not established."],
            ["Are the answers more trustworthy?",
             "Not as a product claim. On a simple noisy series, an ordinary average did better. On a small quiz, the project's score did not beat the model's own confidence."],
        ],
        [2.3 * inch, 4.4 * inch],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "A lower power bill in a data center would require a measured drop in energy for the same accepted "
        "work, or the same energy with more accepted work. Cooler racks would require a measured temperature. "
        "Those instruments were not part of these tests. A physics formula for the smallest energy of erasing "
        "a bit was written down as a reference. Writing the formula down is not a measurement of the card.",
        styles["Body"],
    ))
    story.append(Paragraph("What the tests are good for", styles["H"]))
    story.append(Paragraph(
        "They keep the project honest. They show that a prepared calculation can run on the Blackwell card, "
        "and they show several places where a strong claim would be premature. That is useful laboratory "
        "discipline. It is not a reason to change a facility power plan.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "A later note could talk about savings only after a run records power and temperature on the real "
        "workload, at more than one size, against a named alternative, with the quality of the output defined "
        "in advance. Until then, the accurate sentence is: the chip is still a proposal; the tests are "
        "simulations and one graphics-card timing; power and heat are unknown.",
        styles["Body"],
    ))
    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Plain-language note on the chip idea and the tests",
        author="Ceraph Corporation",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    detailed = OUT / "sri-chip-comparison-detailed.pdf"
    brief = OUT / "sri-chip-comparison-brief.pdf"
    build_detailed(detailed)
    build_brief(brief)
    print(detailed, detailed.stat().st_size)
    print(brief, brief.stat().st_size)


if __name__ == "__main__":
    main()
