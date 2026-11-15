"""Turn the three reader documents into PDFs.

White page, maroon header, gold rule. Mermaid blocks are rendered to images.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "DejaVuSerif.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "DejaVuSerif-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "DejaVuSansMono.ttf")))

MAROON = colors.HexColor("#7c2d2d")
GOLD = colors.HexColor("#a67c42")
FOREST = colors.HexColor("#2c4a3e")
INK = colors.HexColor("#3a2416")
CARD = colors.HexColor("#fbf6ec")

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures" / "reader-pdfs"


def styles():
    return {
        "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=16, leading=20, textColor=MAROON, spaceAfter=8, spaceBefore=4),
        "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=13, leading=16, textColor=MAROON, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=11, leading=14, textColor=FOREST, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=14, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=8, leading=10, textColor=INK),
        "head": ParagraphStyle("head", fontName="Body-Bold", fontSize=8, leading=10, textColor=CARD),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10, textColor=INK),
    }


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    text = esc(text)
    text = re.sub(r"`([^`]+)`", r'<font face="Mono" size="8">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def paint(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.setFillColor(MAROON)
    canvas.rect(0, letter[1] - 0.38 * inch, letter[0], 0.38 * inch, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, letter[1] - 0.44 * inch, letter[0], 0.06 * inch, fill=1, stroke=0)
    canvas.setFillColor(CARD)
    canvas.setFont("Body", 8)
    canvas.drawString(0.55 * inch, letter[1] - 0.24 * inch, "Ceraph Corporation   ·   Kerala, India   ·   www.ceraphco.com")
    canvas.setFillColor(FOREST)
    canvas.setFont("Body", 8)
    canvas.drawString(0.55 * inch, 0.34 * inch, "Ceraph Corporation  ·  chief@ceraphco.com")
    canvas.drawRightString(letter[0] - 0.55 * inch, 0.34 * inch, str(doc.page))
    canvas.restoreState()


def render_mermaid(source: str, name: str) -> Path | None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.png"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "chart.mmd"
        src.write_text(source, encoding="utf-8")
        done = subprocess.run(
            ["npx", "-y", "@mermaid-js/mermaid-cli@11.4.0", "-i", str(src), "-o", str(out), "-b", "white", "-s", "2"],
            capture_output=True,
            text=True,
        )
    if done.returncode != 0 or not out.exists():
        print(done.stderr[-500:])
        return None
    return out


def blocks(markdown: str, stem: str):
    s = styles()
    lines = markdown.splitlines()
    i = 0
    chart = 0
    story = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("```mermaid"):
            i += 1
            body = []
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            chart += 1
            image = render_mermaid("\n".join(body), f"{stem}-{chart}")
            if image:
                story.append(Spacer(1, 4))
                story.append(Image(str(image), width=6.6 * inch, height=3.6 * inch, kind="proportional"))
                story.append(Spacer(1, 8))
            continue
        if line.startswith("```"):
            i += 1
            body = []
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            story.append(Preformatted("\n".join(body), s["code"]))
            story.append(Spacer(1, 6))
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*-+", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            header, data = rows[0], rows[2:]
            style_head = s["head"]
            style_cell = s["cell"]
            table_data = [[Paragraph(inline(c), style_head) for c in header]]
            for row in data:
                table_data.append([Paragraph(inline(c), style_cell) for c in row])
            width = 7.1 * inch / max(1, len(header))
            table = Table(table_data, colWidths=[width] * len(header), repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), MAROON),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, GOLD),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(table)
            story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), s["h1"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), s["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), s["h3"]))
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(ListItem(Paragraph(inline(lines[i][2:]), s["body"])))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12))
            continue
        elif line.strip() == "":
            pass
        else:
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "-", "|", "```")):
                para.append(lines[i])
                i += 1
            story.append(Paragraph(inline(" ".join(para)), s["body"]))
            continue
        i += 1
    return story


def build(markdown_path: Path, pdf_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    story = blocks(text, markdown_path.stem)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.62 * inch,
        title=markdown_path.stem,
        author="Ceraph Corporation",
    )
    doc.build(story, onFirstPage=paint, onLaterPages=paint)
    print(pdf_path)


def main() -> None:
    pairs = [
        (ROOT / "swami-sandeepananda-giri-note.md", ROOT / "swami-sandeepananda-giri-note.pdf"),
        (ROOT / "patent" / "PATENT_SPECIFICATION_FLOWCHARTS.md", ROOT / "patent" / "PATENT_SPECIFICATION_FLOWCHARTS.pdf"),
        (ROOT / "patent" / "PROVISIONAL_WHITEPAPER.md", ROOT / "patent" / "PROVISIONAL_WHITEPAPER.pdf"),
    ]
    for src, dst in pairs:
        build(src, dst)


if __name__ == "__main__":
    main()
