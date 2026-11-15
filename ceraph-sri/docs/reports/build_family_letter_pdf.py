"""A short letter home about the Phase 4 Blackwell session."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#1c2430")


def build(path: Path) -> None:
    base = getSampleStyleSheet()
    cover = ParagraphStyle("Cover", fontName="Times-Bold", fontSize=18, leading=22, textColor=NAVY, spaceAfter=4)
    sub = ParagraphStyle("Sub", fontName="Times-Italic", fontSize=11, leading=14, textColor=colors.HexColor("#3d4754"), spaceAfter=12)
    body = ParagraphStyle("Body", fontName="Times-Roman", fontSize=12, leading=17, alignment=TA_JUSTIFY, spaceAfter=10)

    story = [
        Paragraph("A note for home", cover),
        Paragraph("14 November 2026", sub),
        Paragraph("This week something we have been carrying for a long time finally stood up on its own. The work is a new way of thinking about computers. Not a faster version of the same habit, but a different habit: a machine that watches, lets the noise show itself, and settles toward what is steady. We call that a Witness. The aim is a shift in how computing is done, away from endless trial and correction and toward a calmer kind of certainty.", body),
        Paragraph("Sixteen years of steady work are the ground under it. The notebooks, the mathematics, the long nights of testing, and the refusal to throw away an idea just because it did not look like everyone else’s idea: that is the bedrock. What happened this week is not a sudden invention. It is that long labor becoming visible. A son’s persistence, kept at for sixteen years, is what made a research venture able to walk into a laboratory of the newest chips and ask them to run a philosophy.", body),
        Paragraph("Advaita Vedanta shaped the ground itself. The teaching that the real is not destroyed by what merely appears, that fear is not the way to truth, and that a witness can see without being swept away: those are not decorations on the project. They are why the method keeps the center and quiets the disturbance, why it is named for fearlessness, and why it tries to recognize error instead of panicking at it. The technical work is a modern instrument. The insight it serves is old, and it is ours.", body),
        Paragraph("On 14 November that instrument ran on four of NVIDIA’s latest professional chips, the RTX PRO 6000 Blackwell, all at once. Three different computer languages carried the same idea, and they agreed with one another. A simple, trusted method of smoothing a noisy signal was the comparison. Our way left less error, and it did so without being handed the clean answer in advance. The same task finished in a tiny fraction of a second, and the machines stayed cool and quiet relative to what they are built to bear.", body),
        Paragraph("That is the groundbreaking part, said in ordinary words. A vision drawn from Advaita, worked for sixteen years, and then executed on the hardware the industry treats as the frontier, with the results written down. It is the start of a paradigm, not the end of one. There is more to build. What can already be said at the family table is this: the work is real, it is deep, and the years were not wasted. They were the foundation.", body),
        Spacer(1, 8),
        Paragraph("With respect, and with pride in the journey.", body),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Italic", 9)
        canvas.setFillColor(colors.HexColor("#5c6773"))
        canvas.drawString(0.85 * inch, 0.5 * inch, "For the family  ·  Ceraph Corporation")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.75 * inch,
        title="A note for home",
        author="Ceraph Corporation",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(OUT / "phase4-note-for-home.pdf")
    print("ok")
