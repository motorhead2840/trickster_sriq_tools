#!/usr/bin/env bash
# Render patent FIG. 1–8 as landscape PNG + SVG for Google Docs.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../.." && pwd)"
MMDC=(npx --yes -p @mermaid-js/mermaid-cli mmdc)
# Wide canvas keeps left→right flow readable
WIDTH=2400
HEIGHT=900
SCALE=2

cd "$DIR"
for mmd in fig-*.mmd; do
  base="${mmd%.mmd}"
  echo "Rendering $base ..."
  "${MMDC[@]}" -i "$mmd" -o "${base}.svg" -b white -w "$WIDTH" -H "$HEIGHT" -s "$SCALE"
  "${MMDC[@]}" -i "$mmd" -o "${base}.png" -b white -w "$WIDTH" -H "$HEIGHT" -s "$SCALE"
done

zip -j -q sri-patent-figures-landscape.zip fig-*.png fig-*.svg
echo "Wrote PNG/SVG + sri-patent-figures-landscape.zip in $DIR"
