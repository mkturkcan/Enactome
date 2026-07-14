// Expose engine URL and connectome paths (from env) to the renderer, without
// enabling full node integration. The renderer reads window.ENACTOME_* .
const { contextBridge } = require('electron');
contextBridge.exposeInMainWorld('ENACTOME_API', 'http://127.0.0.1:8765');
contextBridge.exposeInMainWorld('ENACTOME_NODES', process.env.ENACTOME_NODES || '');
contextBridge.exposeInMainWorld('ENACTOME_EDGES', process.env.ENACTOME_EDGES || '');
