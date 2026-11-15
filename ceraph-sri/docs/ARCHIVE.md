# Files removed from the working tree

These files were deleted from the tree without rewriting Git history. Each one
can be recovered from the last commit that contains it:

```bash
git show <commit>:"<path>" > "<filename>"
```

No Python module, test, notebook, or remaining document references these
paths, so none were kept under `tests/data/`.

| Path | Size (bytes) | Last commit |
|---|---:|---|
| `Instructions for Gemini - Project Ceraph.md.docx` | 6226441 | `a90033c1d1be14dd78d6cdd450a3987fd90a3737` |
| `Instructions for Gemini - Project Ceraph.md.rtf` | 999134 | `e4b05507059686e178fff3a6f9bdde31649a05a0` |
| `ceraph_simulation_output.wav` | 88244 | `032506d9b1a9c15b5b9bfb45cfa384a101533949` |
| `dummy_input_image.png` | 10168 | `032506d9b1a9c15b5b9bfb45cfa384a101533949` |
| `swami_sign.40.02 PM.jpeg` | 35455 | `032506d9b1a9c15b5b9bfb45cfa384a101533949` |

`swami_sign.40.02 PM.jpeg` is a personal signature image. It is no longer in
the working tree. The blob is still reachable from the commit above.

Recovery commands:

```bash
git show a90033c1d1be14dd78d6cdd450a3987fd90a3737:"Instructions for Gemini - Project Ceraph.md.docx" > "Instructions for Gemini - Project Ceraph.md.docx"
git show e4b05507059686e178fff3a6f9bdde31649a05a0:"Instructions for Gemini - Project Ceraph.md.rtf" > "Instructions for Gemini - Project Ceraph.md.rtf"
git show 032506d9b1a9c15b5b9bfb45cfa384a101533949:ceraph_simulation_output.wav > ceraph_simulation_output.wav
git show 032506d9b1a9c15b5b9bfb45cfa384a101533949:dummy_input_image.png > dummy_input_image.png
git show 032506d9b1a9c15b5b9bfb45cfa384a101533949:"swami_sign.40.02 PM.jpeg" > "swami_sign.40.02 PM.jpeg"
```
