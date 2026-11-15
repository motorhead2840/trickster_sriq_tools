"""Build the investor project report from the markdown in this folder.

Histograms are taken from figures/. Display math is rendered with KaTeX.
Flowcharts are rendered with Mermaid. Both run in headless Chrome.
"""
from __future__ import annotations

import http.server
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
VENDOR_SRC = Path("/tmp/pdfbuild/node_modules")
PREVIEW = ROOT / "_preview"
CHROME = "/usr/local/bin/google-chrome"

DOCS = [
    ("project-report.md", "Project report for investors"),
]

PAGE_CSS = """
@page {
  size: letter;
  margin: 0.78in 0.62in 0.62in 0.62in;
  @top-left {
    content: "Ceraph  ·  15 November 2026";
    font-family: "Liberation Serif", "Noto Serif", serif;
    font-size: 9pt;
    color: #7c2d2d;
  }
  @top-right {
    content: string(doctitle);
    font-family: "Liberation Serif", "Noto Serif", serif;
    font-size: 9pt;
    color: #2c4a3e;
  }
  @bottom-right {
    content: counter(page);
    font-family: "Liberation Serif", "Noto Serif", serif;
    font-size: 9pt;
    color: #2c4a3e;
  }
}
html { string-set: doctitle attr(data-running-title); }
body {
  font-family: "Liberation Serif", "Noto Serif", serif;
  font-size: 11.5pt;
  line-height: 1.60;
  color: #3a2416;
  margin: 0;
}
h1 {
  font-size: 18pt;
  line-height: 1.25;
  color: #7c2d2d;
  font-weight: 700;
  margin: 0 0 0.45em;
  string-set: doctitle content();
}
h2 {
  font-size: 13.5pt;
  color: #7c2d2d;
  margin: 1.15em 0 0.35em;
  page-break-after: avoid;
}
h3 {
  font-size: 12pt;
  color: #2c4a3e;
  margin: 0.9em 0 0.3em;
  page-break-after: avoid;
}
p { margin: 0 0 0.55em; text-align: justify; hyphens: auto; }
ul, ol { margin: 0.2em 0 0.7em; padding-left: 1.2em; }
li { margin: 0.15em 0; }
code, pre {
  font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
  font-size: 8.5pt;
}
code { background: #fbf6ec; padding: 0 0.15em; }
pre {
  background: #fbf6ec;
  border-left: 3px solid #a67c42;
  padding: 0.45em 0.6em;
  white-space: pre-wrap;
  page-break-inside: avoid;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.4em 0 0.8em;
  page-break-inside: avoid;
  font-size: 8.4pt;
  line-height: 1.3;
}
th {
  background: #7c2d2d;
  color: #fbf6ec;
  font-weight: 700;
  text-align: left;
}
th, td {
  border: 0.4pt solid #a67c42;
  padding: 3px 4px;
  vertical-align: top;
  word-break: break-word;
}
figure {
  margin: 0.6em auto 0.9em;
  page-break-inside: avoid;
  text-align: center;
}
figure img {
  width: auto;
  max-width: 100%;
  max-height: 4.3in;
}
figcaption {
  font-family: "Liberation Sans", "Noto Sans", sans-serif;
  font-size: 9pt;
  color: #2c4a3e;
  margin-top: 0.25em;
}
.math-display {
  margin: 0.45em 0 0.7em;
  text-align: center;
  page-break-inside: avoid;
  overflow-x: auto;
}
.mermaid {
  margin: 0.35em auto 0.7em;
  text-align: center;
  page-break-inside: avoid;
}
.mermaid svg { max-width: 100%; max-height: 8.15in; height: auto; }
"""


def protect(text: str) -> tuple[str, list[str], list[str], list[str]]:
    displays: list[str] = []
    inlines: list[str] = []
    charts: list[str] = []

    def take_chart(match: re.Match[str]) -> str:
        charts.append(match.group(1).strip("\n"))
        return f"\n\nMERMAIDBLOCK{len(charts) - 1}END\n\n"

    def take_display(match: re.Match[str]) -> str:
        displays.append(match.group(1).strip())
        return f"\n\nMATHDISPLAY{len(displays) - 1}END\n\n"

    def take_inline(match: re.Match[str]) -> str:
        inlines.append(match.group(1).strip())
        return f"MATHINLINE{len(inlines) - 1}END"

    text = re.sub(r"```mermaid\n(.*?)```", take_chart, text, flags=re.S)
    text = re.sub(r"\\\[(.*?)\\\]", take_display, text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", take_inline, text, flags=re.S)
    return text, displays, inlines, charts


def restore(html: str, displays: list[str], inlines: list[str], charts: list[str]) -> str:
    def esc(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    for i, latex in enumerate(displays):
        html = html.replace(
            f"MATHDISPLAY{i}END",
            f'<div class="math-display">{esc(latex)}</div>',
        )
    for i, latex in enumerate(inlines):
        html = html.replace(
            f"MATHINLINE{i}END",
            f'<span class="math-inline">{esc(latex)}</span>',
        )
    for i, source in enumerate(charts):
        html = html.replace(
            f"MERMAIDBLOCK{i}END",
            f'<div class="mermaid">{esc(source)}</div>',
        )
    html = re.sub(
        r'<img alt="([^"]*)" src="([^"]+)"\s*/?>',
        r'<figure><img alt="\1" src="../\2" /><figcaption>\1</figcaption></figure>',
        html,
    )
    return html


def html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en" data-running-title="{title}">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<link rel="stylesheet" href="vendor/katex.min.css" />
<style>{PAGE_CSS}</style>
</head>
<body>
{body}
<script src="vendor/katex.min.js"></script>
<script src="vendor/mermaid.min.js"></script>
<script>
document.querySelectorAll(".math-inline").forEach((el) => {{
  katex.render(el.textContent, el, {{throwOnError: false, displayMode: false}});
}});
document.querySelectorAll(".math-display").forEach((el) => {{
  katex.render(el.textContent, el, {{throwOnError: false, displayMode: true}});
}});
mermaid.initialize({{
  startOnLoad: false,
  theme: "neutral",
  securityLevel: "loose",
  flowchart: {{htmlLabels: true, curve: "linear", nodeSpacing: 16, rankSpacing: 22, padding: 6}}
}});
mermaid.run({{querySelector: ".mermaid"}}).then(() => {{
  document.querySelectorAll(".mermaid svg").forEach((svg) => {{
    const box = svg.getBoundingClientRect();
    if (box.height > 760) {{
      svg.style.maxHeight = "760px";
      svg.style.width = "auto";
    }}
  }});
  document.body.setAttribute("data-ready", "1");
}});
</script>
</body>
</html>
"""


def stage_vendor() -> None:
    dest = PREVIEW / "vendor"
    dest.mkdir(parents=True, exist_ok=True)
    katex = VENDOR_SRC / "katex" / "dist"
    shutil.copy(katex / "katex.min.js", dest / "katex.min.js")
    shutil.copy(katex / "katex.min.css", dest / "katex.min.css")
    fonts = dest / "fonts"
    if fonts.exists():
        shutil.rmtree(fonts)
    shutil.copytree(katex / "fonts", fonts)
    shutil.copy(VENDOR_SRC / "mermaid" / "dist" / "mermaid.min.js", dest / "mermaid.min.js")


def build_html() -> None:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    stage_vendor()
    converter = markdown.Markdown(extensions=["extra", "sane_lists", "tables", "fenced_code"])
    for name, title in DOCS:
        source = (ROOT / name).read_text(encoding="utf-8")
        protected, displays, inlines, charts = protect(source)
        converter.reset()
        body = converter.convert(protected)
        body = restore(body, displays, inlines, charts)
        page = html_page(title, body)
        (PREVIEW / name.replace(".md", ".html")).write_text(page, encoding="utf-8")
        print(f"html {name} math={len(displays)}+{len(inlines)} charts={len(charts)}")


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(ROOT), **kwargs
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def print_one(port: int, name: str, index: int) -> None:
    html_name = name.replace(".md", ".html")
    pdf_path = ROOT / name.replace(".md", ".pdf")
    if pdf_path.exists():
        pdf_path.unlink()
    url = f"http://127.0.0.1:{port}/_preview/{html_name}"
    profile = Path(f"/tmp/pdf-chrome-{index}")
    if profile.exists():
        shutil.rmtree(profile)
    proc = subprocess.Popen(
        [
            "/opt/google/chrome/chrome",
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 60
    last = -1
    stable = 0
    while time.time() < deadline:
        if pdf_path.exists():
            size = pdf_path.stat().st_size
            tail = pdf_path.read_bytes()[-16:] if size > 16 else b""
            if size > 2000 and size == last and b"%%EOF" in tail:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
                last = size
        time.sleep(0.35)
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)
    if not pdf_path.exists() or b"%%EOF" not in pdf_path.read_bytes()[-32:]:
        raise SystemExit(f"chrome did not finish {name}")
    print(f"pdf {pdf_path.name} {pdf_path.stat().st_size} bytes")


def print_pdfs(port: int) -> None:
    for index, (name, _title) in enumerate(DOCS):
        print_one(port, name, index)


def main() -> None:
    if not (VENDOR_SRC / "katex").exists():
        raise SystemExit(f"missing {VENDOR_SRC}; npm install katex mermaid there first")
    build_html()
    server, port = serve()
    try:
        print_pdfs(port)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
