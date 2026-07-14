#!/usr/bin/env python3
"""Drive the live Enactome studio UI with Playwright and capture per-scene frames.

Run locally (Chrome/Chromium works there):
    pip install playwright && playwright install chromium
    python record_ui.py

Assumes the engine server is already running on the storyboard port and that the
studio HTML is served (build_video.sh starts both). Writes PNG frames to frames/ui_<id>/.
"""
import json, os, time, http.server, socketserver, threading, functools
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "electron" / "src"
SB = json.load(open(HERE / "storyboard.json"))
PORT = SB["meta"]["server_port"]
W, H, FPS = SB["meta"]["width"], SB["meta"]["height"], SB["meta"]["fps"]
UI_PORT = 8123

def serve_ui():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SRC))
    httpd = socketserver.TCPServer(("127.0.0.1", UI_PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd

def grab(page, scene_id, seconds):
    d = HERE / "frames" / f"ui_{scene_id}"; d.mkdir(parents=True, exist_ok=True)
    n = int(seconds * FPS)
    for i in range(n):
        page.screenshot(path=str(d / f"{i:04d}.png"))
        time.sleep(1.0 / FPS)

def main():
    serve_ui()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": W, "height": H})
        # point the studio's API base at the engine and inject connectome env
        page.add_init_script(f"window.ENACTOME_API='http://127.0.0.1:{PORT}';")
        page.goto(f"http://127.0.0.1:{UI_PORT}/index.html")
        page.wait_for_timeout(1500)

        for sc in SB["scenes"]:
            if sc["kind"] != "ui":
                continue
            a = sc["action"]
            if a == "load":
                page.click("#btnLoad"); page.wait_for_timeout(2500)
            elif a == "circuit_olfactory":
                page.click('.presetlist li[data-preset="olfactory"]'); page.wait_for_timeout(800)
            elif a == "run_all":
                page.click('.tab[data-tab="experiments"]'); page.wait_for_timeout(800)
                page.click("#btnRunAll")
            elif a == "show_lh":
                page.click('.tab[data-tab="experiments"]')
                page.fill("#expFilter", "lh"); page.wait_for_timeout(400)
            elif a == "disease":
                page.click('.tab[data-tab="data"]'); page.wait_for_timeout(400)
                page.click("#btnDisease")
            elif a == "perturb":
                page.click('.tab[data-otab="perturb"]'); page.wait_for_timeout(600)
                page.click("#btnPerturb")
            grab(page, sc["id"], sc["dur"])
        browser.close()
    print("UI frames captured under frames/ui_*/")

if __name__ == "__main__":
    main()
