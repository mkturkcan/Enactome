# Enactome explainer video

Produces a ~2-minute narrated walkthrough: the premise (fly, connectome) for a
lay-scientist, the live software running the paper's experiments, and a scroll through
the manuscript at the end. Subtitles are burned in; a soft royalty-free music bed is
generated locally.

## What it does

`build_video.sh` runs six steps:
1. installs Playwright + Chromium and the frame libraries;
2. copies the manuscript PDF into `assets/`;
3. starts the engine and loads the connectome;
4. drives the **real studio UI** with Playwright, recording frames of each action
   (load connectome, view the olfactory circuit, run all 35 experiments, lateral-horn
   results, disease atlas, perturbation prediction);
5. renders the title/section/outro cards, the paper-scroll frames, and the music;
6. burns subtitles and stitches everything into `enactome_demo.mp4`.

The scene order, durations, and every subtitle live in `storyboard.json` — edit that one
file to change pacing or wording.

## Run it (on your laptop, where Chromium works)

```bash
cd enactome/video
export ENACTOME_NODES=/path/to/neurons.csv.gz
export ENACTOME_EDGES=/path/to/connections_princeton.csv.gz
export ENACTOME_PAPER=../paper/enactome_manuscript.pdf   # optional; auto-found if adjacent
./build_video.sh
# -> enactome/video/enactome_demo.mp4
```

Requires: Python 3.10+, Node not needed, ffmpeg on PATH, ~2 GB free for Chromium.

## The pieces (all editable)

- `storyboard.json` — scenes, durations, subtitles, resolution, port.
- `record_ui.py` — Playwright driver; each `ui` scene maps to a UI action.
- `make_frames.py` — title/section/outro cards + manuscript scroll (no browser needed).
- `make_music.py` — generates a quiet I-vi-IV-V sine-pad bed sized to the video.
- `assemble.py` — burns subtitles, encodes per-scene clips, concatenates, muxes audio.

## Prompt for VS Code / Claude Code

Paste this into Claude Code (or your VS Code AI assistant) from the package root to
generate and refine the video interactively:

> Build the Enactome explainer video. Read `video/storyboard.json` for the scene plan.
> Start the engine (`PYTHONPATH=. uvicorn server.app:app --port 8799`), load the BANC
> connectome from my `neurons.csv.gz` and `connections_princeton.csv.gz`, then run
> `video/build_video.sh` to record the live UI, render the paper scroll, add the music
> bed, and assemble `video/enactome_demo.mp4`. The video must: (1) open with the premise
> of the fly and the connectome for a lay-scientist, (2) show the software being used to
> reproduce the paper's results (loading the connectome, running the 35 experiments,
> the lateral-horn and disease-genetics screens), and (3) end by scrolling through the
> manuscript PDF to show every figure. Keep subtitles on every scene and the elevator
> music soft. If any UI selector in `record_ui.py` no longer matches, fix it against
> `electron/src/index.html`. Target about two minutes total.

Author: Mehmet Kerem Turkcan, Columbia University (mkt2126@columbia.edu).
