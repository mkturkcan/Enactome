// Enactome Electron main process.
// Spawns the Python analysis server as a child process, then loads the UI.
// The renderer talks to the engine over http://127.0.0.1:8765 — the same API an LLM uses.
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = 8765;
let serverProc = null;

function startEngine() {
  // Launch the bundled Python server. In a packaged app this points at the
  // PyInstaller/conda-pack'd interpreter; in dev it uses the repo's python.
  const repoRoot = path.resolve(__dirname, '..');
  serverProc = spawn('python', ['-m', 'uvicorn', 'server.app:app', '--port', String(PORT)], {
    cwd: repoRoot,
    env: { ...process.env, PYTHONPATH: repoRoot },
    stdio: 'inherit',
  });
  serverProc.on('exit', (code) => console.log(`[engine] exited ${code}`));
}

function waitForEngine(cb, tries = 40) {
  const req = http.get(`http://127.0.0.1:${PORT}/health`, (res) => {
    res.resume(); cb();
  });
  req.on('error', () => {
    if (tries > 0) setTimeout(() => waitForEngine(cb, tries - 1), 250);
    else cb(new Error('engine did not start'));
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1480, height: 900, minWidth: 1180, minHeight: 720,
    backgroundColor: '#0c0f16',
    webPreferences: {
      contextIsolation: true, nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });
  win.loadFile(path.join(__dirname, 'src', 'index.html'));
  if (process.argv.includes('--dev')) win.webContents.openDevTools();
}

app.whenReady().then(() => {
  startEngine();
  waitForEngine((err) => {
    if (err) console.error(err);
    createWindow();
  });
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (serverProc) serverProc.kill();
  if (process.platform !== 'darwin') app.quit();
});
