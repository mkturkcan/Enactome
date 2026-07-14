# Headless frontend test

Verifies the renderer loads and drives the engine without a browser (jsdom).
Chrome/Electron is not required.

    # 1. start the engine (from the package root, on the port the test expects)
    cd ../.. && python -m uvicorn server.app:app --port 8799
    # 2. in another shell
    cd electron/test && npm install jsdom@22 node-fetch@2 && node test_frontend.js

Passing output shows brand=ENACTOME, all tabs present, 37 experiments listed,
an experiment run returning pass, the disease screen returning results, and
js_errors: [].
