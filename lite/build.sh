#!/usr/bin/env bash
# Build the JupyterLite (WASM) static site locally.
#
#   bash lite/build.sh
#
# Output is written to lite/_output/ — open it with:
#   python3 -m http.server -d lite/_output 8000
#   # then visit http://localhost:8000/lab/index.html?path=document_similarity.ipynb
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Use an isolated virtual environment so the build tooling doesn't pollute
# (or get confused by) your system Python.
if [ ! -d ".venv" ]; then
  echo "Creating build virtualenv at lite/.venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Building JupyterLite site ..."
jupyter lite build

echo
echo "Done. The static site is in lite/_output/"
echo "Preview locally with:"
echo "  python3 -m http.server -d lite/_output 8000"
echo "  open http://localhost:8000/lab/index.html?path=document_similarity.ipynb"
