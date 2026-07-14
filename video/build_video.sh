#!/usr/bin/env bash
# One-shot: start the engine, record the live UI, render cards + paper scroll, add music, assemble.
# Run from the package root's video/ directory on a machine where Chromium works (your laptop).
set -e
cd "$(dirname "$0")"
PORT=$(python3 -c "import json;print(json.load(open('storyboard.json'))['meta']['server_port'])")

echo "[1/6] deps"
pip install -q playwright pypdfium2 pillow numpy >/dev/null
playwright install chromium >/dev/null

echo "[2/6] copy manuscript into assets"
mkdir -p assets
cp ../../paper/enactome_manuscript.pdf assets/manuscript.pdf 2>/dev/null || \
  cp "${ENACTOME_PAPER:?set ENACTOME_PAPER to the manuscript PDF path}" assets/manuscript.pdf

echo "[3/6] start engine on :$PORT"
: "${ENACTOME_NODES:?set ENACTOME_NODES to neurons.csv.gz}"
: "${ENACTOME_EDGES:?set ENACTOME_EDGES to connections_princeton.csv.gz}"
( cd .. && PYTHONPATH=. python3 -m uvicorn server.app:app --port "$PORT" >/tmp/enactome_srv.log 2>&1 & echo $! > /tmp/enactome_srv.pid )
sleep 6
# load the connectome once so the UI's Load button and experiments have data
curl -s -X POST "localhost:$PORT/load_connectome" -H 'Content-Type: application/json' \
  -d "{\"nodes_path\":\"$ENACTOME_NODES\",\"edges_path\":\"$ENACTOME_EDGES\"}" >/dev/null

echo "[4/6] record live UI"
python3 record_ui.py

echo "[5/6] render cards + paper scroll + music"
python3 make_frames.py
python3 make_music.py

echo "[6/6] assemble"
python3 assemble.py

kill "$(cat /tmp/enactome_srv.pid)" 2>/dev/null || true
echo "DONE -> video/enactome_demo.mp4"
