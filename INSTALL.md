# Install

Requires Python 3.10+ and (for the desktop app) Node 16+.

## Engine

```bash
pip install -e .[server]        # add [gpu] for CUDA, [server] for the HTTP API
```

## Quick start

```bash
# 1. start the engine
uvicorn server.app:app --port 8799

# 2. load the connectome and run every experiment
curl -X POST localhost:8799/load_connectome \
  -d '{"nodes_path":"neurons.csv.gz","edges_path":"connections_princeton.csv.gz"}' \
  -H 'Content-Type: application/json'
curl -X POST localhost:8799/experiments/run -d '{}' -H 'Content-Type: application/json'
```

Expected: `{"passed": 37, "total": 37}`.

## Desktop app

```bash
cd electron && npm install && npm start
```

`main.js` starts the engine automatically. Set the connectome paths once via the
`ENACTOME_NODES` / `ENACTOME_EDGES` environment variables, or from the Load panel.

## Tests

```bash
pytest tests/                                                    # engine, no data needed
ENACTOME_NODES=neurons.csv.gz ENACTOME_EDGES=connections_princeton.csv.gz pytest tests/   # + BANC regression
cd electron/test && npm install jsdom@22 node-fetch@2 && node test_frontend.js            # headless frontend
```

## Claude API key (optional)

The engine reads `ANTHROPIC_API_KEY` from the environment for LLM-driven analysis.
It is never required for any experiment or data screen; the `/tools` manifest lets any
agent drive the same endpoints.
