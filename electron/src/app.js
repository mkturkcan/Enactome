// Enactome renderer — talks to the local engine over HTTP (the same API an LLM calls).
const API = 'http://127.0.0.1:8765';
const $ = (id) => document.getElementById(id);

async function api(path, method = 'GET', body = null) {
  const opt = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(API + path, opt);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

async function refreshStatus() {
  try {
    const h = await api('/health');
    $('status').textContent = `engine v${h.version} · ${h.connectome_loaded ? 'connectome loaded' : 'no connectome'}`;
  } catch { $('status').textContent = 'engine offline'; }
}

const tabs = {
  load: () => `
    <h2>Load connectome</h2>
    <p class="desc">Point the engine at BANC-format node and edge CSVs (gzip ok). Cached after first load.</p>
    <div class="card">
      <label>Nodes CSV path</label><input id="np" placeholder="/path/to/neurons.csv.gz">
      <label>Edges CSV path</label><input id="ep" placeholder="/path/to/connections.csv.gz">
      <button class="run" onclick="doLoad()">Load</button>
      <div id="loadout"></div>
    </div>`,
  census: () => `
    <h2>Census</h2>
    <p class="desc">Per-cell-class neuron counts, neurotransmitter composition, and dominant cell types.</p>
    <div class="card"><button class="run" onclick="doCensus()">Run census</button><div id="censusout"></div></div>`,
  trace: () => `
    <h2>Trace pathway</h2>
    <p class="desc">Ordered multi-layer pathway through cell classes — returns per-layer size and inter-layer synapse counts.</p>
    <div class="card">
      <label>Layer classes (comma-separated)</label>
      <input id="layers" value="olfactory_receptor_neuron,antennal_lobe_projection_neuron,lateral_horn_output_neuron">
      <button class="run" onclick="doTrace()">Trace</button><div id="traceout"></div>
    </div>`,
  enrich: () => `
    <h2>Disease enrichment</h2>
    <p class="desc">Size-controlled, null-tested enrichment of human-orthologous disease genes per circuit element. Load a prepared gene→disease bundle (JSON).</p>
    <div class="card">
      <label>Gene-disease bundle JSON path</label><input id="bundle" placeholder="/path/to/atlas_bundle.json">
      <button class="run" onclick="doEnrich()">Compute enrichment</button><div id="enrichout"></div>
    </div>`,
  mb: () => `
    <h2>MB behavior &amp; perturbation</h2>
    <p class="desc">Build the mushroom-body learning + locomotion model (replicates Aso 2014), then predict the behavioral effect of activating, silencing, or ablating any MBON/DAN cell type — a testable optogenetic/ablation hypothesis with the genetic driver to use.</p>
    <div class="card">
      <button class="run" onclick="doMBBuild()">Build MB model</button><div id="mbbuildout"></div>
    </div>
    <div class="card">
      <label>Target cell type (e.g. MBON01, PPL1, PAM)</label><input id="mbtarget" value="PPL1">
      <label>Perturbation</label>
      <select id="mbmode"><option value="activate">activate (optogenetic)</option><option value="silence">silence</option><option value="ablate">ablate</option></select>
      <button class="run" onclick="doMBPerturb()">Predict behavior</button><div id="mbperturbout"></div>
    </div>`,
  arena: () => `
    <h2>4-quadrant arena — canonical demo</h2>
    <p class="desc">The whole-brain demo. Optogenetically activate an MBON type while flies are in the lit quadrants; the mushroom-body valence signal gates turning, the central-complex ring attractor supplies heading, and the fly population redistributes. Returns the preference index (PI) measured in real MB optogenetics assays.</p>
    <div class="card">
      <label>MBON type to activate in lit quadrants (e.g. MBON-GLUT, MBON-GABA, MBON01)</label>
      <input id="arenatarget" value="MBON-GLUT">
      <button class="run" onclick="doArena()">Run arena (build MB model first)</button>
      <div id="arenaout"></div>
    </div>`,
  flybrain: () => `
    <h2>Canonical fly brain — innate + learned valence</h2>
    <p class="desc">The fundamental architecture: the lateral horn reads <b>innate</b> (hardwired) valence, the mushroom body writes <b>learned</b> valence via dopaminergic teaching. Query any odor, form a memory, or lesion either channel. Runs from the shipped connectome bundle — no data load needed. See NEURON_MODELS.md for how each layer computes.</p>
    <div class="card">
      <button class="run" onclick="doDissoc()">Run double-dissociation test cases</button>
      <div id="dissocout"></div>
    </div>
    <div class="card">
      <label>Glomerulus (e.g. DM1, DA2)</label><input id="fbglom" value="DM1">
      <label><input type="checkbox" id="fbtrain"> form aversive memory (pair with punishment DAN)</label>
      <label><input type="checkbox" id="fblh"> lesion lateral horn</label>
      <label><input type="checkbox" id="fbmb"> lesion mushroom body</label>
      <button class="run" onclick="doFBValence()">Query valence</button>
      <div id="fbvalout"></div>
    </div>`,
};

function show(tab) {
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  $('main').innerHTML = tabs[tab]();
}
document.querySelectorAll('nav button').forEach(b => b.onclick = () => show(b.dataset.tab));

async function doLoad() {
  const out = $('loadout'); out.innerHTML = 'loading…';
  try {
    const j = await api('/load_connectome', 'POST', { nodes_path: $('np').value, edges_path: $('ep').value });
    out.innerHTML = `<pre>${j.n_neurons.toLocaleString()} neurons · ${j.n_edges.toLocaleString()} edges · ${j.classes.length} classes</pre>`;
    refreshStatus();
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doCensus() {
  const out = $('censusout'); out.innerHTML = 'running…';
  try {
    const rows = await api('/census');
    out.innerHTML = '<table><tr><th>element</th><th>n</th><th>top NT</th></tr>' +
      rows.slice(0, 20).map(r => `<tr><td>${r.element}</td><td>${r.n_neurons}</td>
        <td>${Object.entries(r.nt_composition).slice(0,3).map(([k,v])=>`${k}:${v}`).join(', ')}</td></tr>`).join('') + '</table>';
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doTrace() {
  const out = $('traceout'); out.innerHTML = 'tracing…';
  try {
    const j = await api('/trace_pathway', 'POST', { layer_classes: $('layers').value.split(',').map(s => s.trim()) });
    let h = '<table><tr><th>layer</th><th>neurons</th></tr>' +
      Object.entries(j.layers).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('') + '</table>';
    h += '<table><tr><th>connection</th><th>edges</th><th>synapses</th></tr>' +
      Object.entries(j.edges).map(([k,v]) => `<tr><td>${k}</td><td>${v.n_edges}</td><td>${v.n_syn}</td></tr>`).join('') + '</table>';
    out.innerHTML = h;
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doEnrich() {
  const out = $('enrichout'); out.innerHTML = 'computing null (this takes a moment)…';
  try {
    const bundle = await (await fetch('file://' + $('bundle').value)).json();
    const rows = await api('/enrichment', 'POST', bundle);
    const keys = Object.keys(rows[0]).filter(k => k.startsWith('z_'));
    out.innerHTML = '<table><tr><th>element</th><th>n</th>' + keys.map(k => `<th>${k.slice(2,10)}</th>`).join('') + '</tr>' +
      rows.map(r => `<tr><td>${r.circuit_element}</td><td>${r.n_genes}</td>` +
        keys.map(k => { const z=r[k]; const cls=(Math.abs(z)>=1.96?'z-sig ':'')+(z>0?'z-pos':'z-neg');
          return `<td class="${cls}">${z}</td>`; }).join('') + '</tr>').join('') + '</table>';
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doMBBuild() {
  const out = $('mbbuildout'); out.innerHTML = 'building model…';
  try {
    const j = await api('/mb/build', 'POST', {});
    out.innerHTML = `<pre>${j.n_kc} Kenyon cells → ${j.n_mbon} MBONs, ${j.n_dan} DANs
valence: ${j.valence_counts.approach} approach · ${j.valence_counts.avoid} avoid · ${j.valence_counts.neutral} neutral</pre>`;
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doMBPerturb() {
  const out = $('mbperturbout'); out.innerHTML = 'predicting…';
  try {
    const j = await api('/mb/perturb', 'POST',
      { target_cell_type: $('mbtarget').value, mode: $('mbmode').value });
    const col = j.predicted_behavior === 'approach' ? '#2ca02c' : (j.predicted_behavior === 'avoidance' ? '#d62728' : '#8b98ad');
    out.innerHTML = `<div style="margin-top:10px">
      <div style="font-size:15px;font-weight:700;color:${col}">${j.predicted_behavior.toUpperCase()} &nbsp;<span style="color:var(--muted);font-weight:400">(Δ drive ${j.delta})</span></div>
      <p style="font-size:12.5px;margin:8px 0">${j.hypothesis}</p>
      <pre>genetic handle: ${j.genetic_handle}
drivers: ${(j.drivers||[]).join(', ')}
baseline drive: ${j.baseline_drive} → perturbed: ${j.perturbed_drive}</pre></div>`;
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doArena() {
  const out = $('arenaout'); out.innerHTML = 'simulating 300 flies…';
  try {
    const j = await api('/arena', 'POST', { target_cell_type: $('arenatarget').value });
    const col = j.PI > 0.05 ? '#2ca02c' : (j.PI < -0.05 ? '#d62728' : '#8b98ad');
    out.innerHTML = `<div style="margin-top:10px">
      <div style="font-size:16px;font-weight:700;color:${col}">PI = ${j.PI.toFixed(3)}
        <span style="color:var(--muted);font-weight:400">— ${j.interpretation}</span></div>
      <pre>target: ${j.target}   valence in light: ${j.valence_in_light.toFixed(1)}
occupancy trace (fraction in lit quadrants over time):
${j.occupancy_trace.map(x => x.toFixed(2)).join(' ')}</pre></div>`;
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doDissoc() {
  const out = $('dissocout'); out.innerHTML = 'running test cases…';
  try {
    const j = await api('/flybrain/dissociation');
    out.innerHTML = `<table>
      <tr><th>test case</th><th>innate (LH)</th><th>learned (MB)</th></tr>
      <tr><td>1. naive, appetitive odor</td><td>${j.test1_innate_tracks_glomerulus.appetitive_odor_innate}</td><td>—</td></tr>
      <tr><td>1. naive, aversive odor</td><td>${j.test1_innate_tracks_glomerulus.aversive_odor_innate}</td><td>—</td></tr>
      <tr><td>2. after aversive training</td><td>${j.test2_mb_writes_memory.innate_unchanged} (unchanged)</td><td>${j.test2_mb_writes_memory.learned_before} → ${j.test2_mb_writes_memory.learned_after}</td></tr>
      <tr><td>3. LH lesion</td><td style="color:#d62728">${j.test3_LH_lesion.innate} (abolished)</td><td style="color:#2ca02c">${j.test3_LH_lesion.learned} (spared)</td></tr>
      <tr><td>4. MB lesion</td><td style="color:#2ca02c">${j.test4_MB_lesion.innate} (spared)</td><td style="color:#d62728">${j.test4_MB_lesion.learned} (abolished)</td></tr>
      </table><p style="font-size:12px;color:var(--muted);margin-top:8px">${j.conclusion}</p>`;
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

async function doFBValence() {
  const out = $('fbvalout'); out.innerHTML = 'computing…';
  try {
    const j = await api('/flybrain/valence', 'POST', {
      glomerulus: $('fbglom').value, train_punishment: $('fbtrain').checked,
      lh_lesion: $('fblh').checked, mb_lesion: $('fbmb').checked });
    out.innerHTML = `<pre>glomerulus ${j.glomerulus} (innate glom valence ${j.innate_glom_valence})
innate  (LH): ${j.innate}
learned (MB): ${j.learned}
combined:     ${j.combined}
${j.trained_punishment ? '[aversive memory formed] ' : ''}${j.lh_lesion ? '[LH lesioned] ' : ''}${j.mb_lesion ? '[MB lesioned]' : ''}</pre>`;
  } catch (e) { out.innerHTML = `<pre style="color:#ff8a8a">${e.message}</pre>`; }
}

show('load');
refreshStatus();
setInterval(refreshStatus, 5000);
