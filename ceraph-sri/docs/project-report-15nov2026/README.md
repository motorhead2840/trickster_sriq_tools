# Project report, 15 November 2026

A report for investors, built from the two forty-minute soak logs and checked against them.

| File | What it is |
|---|---|
| `project-report.md`, `project-report.pdf` | The report. |
| `data/` | The two soak logs the report quotes. |
| `figures/` | The five charts of those logs, and three charts copied from the afternoon documents. The copied three are marked in the report as not recomputed. |
| `check_soak.py` | Recomputes the headline numbers from the logs. |
| `projection.py` | The funding lines and the fifteen-year arithmetic. Both are assumptions. |
| `build_pdfs.py` | Rebuilds the PDF. Needs the local KaTeX and Mermaid already installed for the 15 November documents. |

From the repository root:

```
python docs/project-report-15nov2026/check_soak.py
python docs/project-report-15nov2026/build_pdfs.py
```
