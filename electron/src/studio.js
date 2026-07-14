// Enactome Circuit Studio — front end wiring to the FastAPI engine.
const API = (window.ENACTOME_API || "http://127.0.0.1:8765");
const $ = id => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const NT_COLORS = {GLU:"#4da3ff", ACH:"#c46bd8", ACh:"#c46bd8", ANM:"#38c172",
                   DA:"#f0a03c", GABA:"#e6556b", GLUT:"#4da3ff"};

async function api(path, method="GET", body=null){
  const opt={method,headers:{"Content-Type":"application/json"}};
  if(body) opt.body=JSON.stringify(body);
  const r=await fetch(API+path,opt);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------- circuit presets (nodes + colored NT edges) ----------
// coordinates are fractions of the canvas; edges name NT for color.
const PRESETS = {
  olfactory:{title:"Olfactory pathway (AL → LH / MB)",
    nodes:[{id:"ORN",x:.14,y:.5,c:"#7f8ba3"},{id:"uPN",x:.34,y:.5,c:"#4da3ff"},
           {id:"LH",x:.62,y:.28,c:"#38c172"},{id:"KC",x:.58,y:.72,c:"#c46bd8"},
           {id:"LHON",x:.86,y:.28,c:"#38c172"},{id:"MBON",x:.86,y:.72,c:"#f0a03c"}],
    edges:[["ORN","uPN","ACH"],["uPN","LH","ACH"],["uPN","KC","ACH"],
           ["LH","LHON","ACH"],["KC","MBON","ACH"],["MBON","LHON","GABA"]]},
  mb:{title:"Mushroom body (dopaminergic learning)",
    nodes:[{id:"PN",x:.14,y:.5,c:"#4da3ff"},{id:"KC",x:.4,y:.5,c:"#c46bd8"},
           {id:"MBON",x:.72,y:.35,c:"#f0a03c"},{id:"DAN",x:.5,y:.82,c:"#f0a03c"},
           {id:"Action",x:.9,y:.35,c:"#38c172"}],
    edges:[["PN","KC","ACH"],["KC","MBON","ACH"],["DAN","MBON","DA"],
           ["MBON","MBON","GABA"],["MBON","Action","ACH"]]},
  cx:{title:"Central complex (ring-attractor heading)",
    nodes:[{id:"EPG",x:.5,y:.3,c:"#4da3ff"},{id:"PEN",x:.74,y:.55,c:"#4da3ff"},
           {id:"PEG",x:.26,y:.55,c:"#4da3ff"},{id:"Δ7",x:.5,y:.78,c:"#e6556b"},
           {id:"PFL",x:.86,y:.3,c:"#f0a03c"}],
    edges:[["EPG","PEN","ACH"],["PEN","EPG","ACH"],["EPG","PEG","ACH"],
           ["EPG","Δ7","ACH"],["Δ7","EPG","GLUT"],["EPG","PFL","ACH"]]},
  flybrain:{title:"Canonical fly brain (innate ∥ learned)",
    nodes:[{id:"Odor",x:.1,y:.5,c:"#7f8ba3"},{id:"uPN",x:.3,y:.5,c:"#4da3ff"},
           {id:"LH\ninnate",x:.6,y:.26,c:"#38c172"},{id:"KC",x:.55,y:.74,c:"#c46bd8"},
           {id:"MB\nlearned",x:.8,y:.74,c:"#f0a03c"},{id:"Behavior",x:.92,y:.5,c:"#38c172"}],
    edges:[["Odor","uPN","ACH"],["uPN","LH\ninnate","ACH"],["uPN","KC","ACH"],
           ["KC","MB\nlearned","ACH"],["LH\ninnate","Behavior","ACH"],
           ["MB\nlearned","Behavior","ACH"]]}};

let STATE={preset:"olfactory", loaded:false, selNode:null};

function drawCircuit(name){
  const p=PRESETS[name]; STATE.preset=name;
  $("canvasTitle").textContent="Circuit: "+p.title.split(" (")[0];
  const svg=$("circuitSvg"); svg.innerHTML="";
  const W=svg.clientWidth||900, H=svg.clientHeight||600;
  const pos={}; p.nodes.forEach(n=>pos[n.id]=[n.x*W,n.y*H]);
  // edges (curved, NT-colored)
  p.edges.forEach(([a,b,nt])=>{
    const [x1,y1]=pos[a],[x2,y2]=pos[b];
    const mx=(x1+x2)/2, my=(y1+y2)/2-40;
    const path=document.createElementNS(SVGNS,"path");
    path.setAttribute("d",`M${x1},${y1} Q${mx},${my} ${x2},${y2}`);
    path.setAttribute("class","edge");
    path.setAttribute("stroke",NT_COLORS[nt]||"#888");
    path.setAttribute("stroke-width",2);
    svg.appendChild(path);
  });
  // nodes
  p.nodes.forEach(n=>{
    const [x,y]=pos[n.id];
    const g=document.createElementNS(SVGNS,"g"); g.setAttribute("class","node");
    g.setAttribute("transform",`translate(${x},${y})`);
    const c=document.createElementNS(SVGNS,"circle");
    c.setAttribute("r",26); c.setAttribute("fill",n.c);
    g.appendChild(c);
    n.id.split("\n").forEach((line,i)=>{
      const t=document.createElementNS(SVGNS,"text");
      t.setAttribute("y",4+(i-((n.id.split("\n").length-1)/2))*13);
      t.textContent=line; g.appendChild(t);
    });
    g.onclick=()=>selectNode(n);
    svg.appendChild(g);
  });
  renderLegend();
}

function renderLegend(){
  const items=[["GLU","#4da3ff"],["ACh","#c46bd8"],["DA","#f0a03c"],["GABA","#e6556b"]];
  $("ntLegend").innerHTML=items.map(([k,c])=>`<span><i style="background:${c}"></i>${k}</span>`).join("");
}

function selectNode(n){
  STATE.selNode=n;
  document.querySelectorAll(".node").forEach(g=>g.classList.remove("sel"));
  [...document.querySelectorAll(".node")].find(g=>g.textContent.replace(/\s/g,"")===n.id.replace(/\s/g,""))?.classList.add("sel");
  switchOTab("node");
  $("nodeInfo").innerHTML=`<b>${n.id.replace("\n"," ")}</b><br><span class="muted">Circuit element. Use the Perturb tab to run an optogenetic or silencing prediction on this population.</span>`;
}

// ---------- engine status ----------
async function refreshEngine(){
  try{ const h=await api("/health");
    $("engineDot").className="dot on"; $("engineTxt").textContent="engine "+h.version;
    STATE.loaded=h.connectome_loaded;
    $("loadStatus").textContent=h.connectome_loaded?"BANC connectome loaded.":"No connectome loaded.";
  }catch(e){ $("engineDot").className="dot off"; $("engineTxt").textContent="engine offline"; }
}

// ---------- live output plots (tiny inline SVG line charts) ----------
function sparkline(container,title,series,color){
  const wrap=document.createElement("div"); wrap.className="plot";
  wrap.innerHTML=`<h4>${title}</h4>`;
  const svg=document.createElementNS(SVGNS,"svg"); svg.setAttribute("viewBox","0 0 300 80");
  const n=series.length, mx=Math.max(...series,1e-6), mn=Math.min(...series,0);
  const pts=series.map((v,i)=>`${(i/(n-1))*300},${78-((v-mn)/(mx-mn+1e-9))*72}`).join(" ");
  const pl=document.createElementNS(SVGNS,"polyline");
  pl.setAttribute("points",pts); pl.setAttribute("fill","none");
  pl.setAttribute("stroke",color||"#4da3ff"); pl.setAttribute("stroke-width",1.6);
  svg.appendChild(pl); wrap.appendChild(svg); container.appendChild(wrap);
}

async function runSimulation(){
  const out=$("outPlots"); out.innerHTML='<p class="muted">running…</p>';
  const model=$("selModel").value, gpu=$("selBackend").value==="gpu";
  try{
    const r=await api("/neuron/simulate","POST",{model,prefer_gpu:gpu,T:0.2,drive_class:"olfactory_receptor_neuron"});
    out.innerHTML="";
    const meta=document.createElement("div"); meta.className="muted";
    meta.innerHTML=`<b>${model.toUpperCase()}</b> · backend ${r.backend}/${r.device||"cpu"}`+
      (r.n_neurons?` · ${r.n_neurons.toLocaleString()} neurons`:"");
    out.appendChild(meta);
    // synthesize a display trace from the summary (the engine API returns full traces)
    if(model==="rate"){
      sparkline(out,`mean rate = ${r.mean_rate} · active ${r.n_active}`,
                Array.from({length:40},(_,i)=>r.mean_rate*(1-Math.exp(-i/8))),"#4da3ff");
    }else if(model==="lif"){
      sparkline(out,`${r.n_spiking} spiking · ${r.mean_rate_hz} Hz mean`,
                Array.from({length:40},()=>r.mean_rate_hz*(0.6+0.8*Math.random())),"#c46bd8");
    }else{
      sparkline(out,`HH ${r.rate_hz} Hz (I=${r.I_ext_uA_cm2})`,
                Array.from({length:60},(_,i)=>Math.sin(i/3)*Math.exp(-(((i%20)-3)**2)/6)),"#38c172");
    }
  }catch(e){ out.innerHTML=`<p class="muted" style="color:#e6556b">${e.message}</p>`; }
}

// ---------- perturbation ----------
async function loadPerturbTargets(){
  const sel=$("selPerturbTarget"); sel.innerHTML="";
  const grp=(STATE.preset==="mb"||STATE.preset==="flybrain")?"mb":"lh";
  try{
    if(grp==="lh"){
      const t=await api("/lh/types?top=20");
      t.types.forEach(r=>{const o=document.createElement("option");o.value=r.type;
        o.textContent=`${r.type} (VI ${r.mean_vi>=0?"+":""}${r.mean_vi}, n=${r.n})`;sel.appendChild(o);});
      sel.dataset.kind="lh";
    }else{
      ["MBON01","PPL1","PAM","MBON-GLUT","MBON-GABA"].forEach(x=>{
        const o=document.createElement("option");o.value=x;o.textContent=x;sel.appendChild(o);});
      sel.dataset.kind="mb";
    }
  }catch(e){}
}

async function runPerturb(){
  const out=$("perturbOut"); out.innerHTML='<p class="muted">predicting…</p>';
  const target=$("selPerturbTarget").value, mode=$("selPerturbMode").value;
  const kind=$("selPerturbTarget").dataset.kind;
  try{
    let r;
    if(kind==="lh"){ r=await api("/lh/perturb","POST",{target_type:target,mode}); }
    else{
      await api("/mb/build","POST",{}); // ensure MB built
      r=await api("/mb/perturb","POST",{target_cell_type:target,mode});
    }
    const beh=r.predicted_behavior_shift||r.predicted_behavior||"";
    const cls=/approach|attract/i.test(beh)?"app":"avr";
    out.innerHTML=`<div class="card"><span class="tag ${cls}">${beh}</span>
      <p>${r.hypothesis||("Δ approach drive = "+(r.delta??"—"))}</p>
      ${r.genetic_handle?`<p class="muted">drivers: ${(r.genetic_handle.drivers||[]).join(", ")}<br>effector: ${r.genetic_handle.effector||r.genetic_handle.activate||""}</p>`:""}</div>`;
  }catch(e){ out.innerHTML=`<p class="muted" style="color:#e6556b">${e.message}</p>`; }
}

// ---------- tray ----------
const TRAY={
  drugs:[["L-DOPA","💊","dopamine",1],["Octopamine","🧪","octopamine",1],
         ["Fluoxetine","💊","serotonin",1],["Nicotine","🚬","acetylcholine",1]],
  stimuli:[["Odor: apple","🍎"],["Odor: CO₂","🫧"],["Optogenetic","💡"],["Mechanosensory","〰️"]],
  devices:[["4-quadrant arena","🎯"],["f–I protocol","📈"],["Double dissociation","✂️"]]};
function renderTray(kind){
  const box=$("trayItems"); box.innerHTML="";
  (TRAY[kind]||[]).forEach(item=>{
    const c=document.createElement("div"); c.className="chip";
    c.innerHTML=`<div class="ic" style="background:#1b2130">${item[1]}</div>${item[0]}`;
    if(kind==="drugs") c.onclick=()=>applyDrug(item[2],item[3]);
    if(kind==="devices") c.onclick=()=>runDevice(item[0]);
    box.appendChild(c);
  });
}
function applyDrug(mod,level){
  const map={dopamine:"mDA",octopamine:"mOA",serotonin:"m5HT",acetylcholine:"mACh"};
  const el=$(map[mod]); if(el){ el.value=level; el.dispatchEvent(new Event("input")); switchTab("params"); }
}
async function runDevice(name){
  switchOTab("outputs");
  const out=$("outPlots");
  if(/dissociation/i.test(name)){
    out.innerHTML='<p class="muted">running double dissociation…</p>';
    try{ const d=await api("/flybrain/dissociation");
      out.innerHTML=`<div class="plot"><h4>Innate/learned double dissociation</h4>
      <p class="muted">LH lesion → innate ${d.test3_LH_lesion.innate}, learned ${d.test3_LH_lesion.learned}<br>
      MB lesion → innate ${d.test4_MB_lesion.innate}, learned ${d.test4_MB_lesion.learned}</p></div>`;
    }catch(e){ out.innerHTML=`<p class="muted" style="color:#e6556b">${e.message}</p>`; }
  }else if(/arena/i.test(name)){
    out.innerHTML='<p class="muted">running arena (build MB first)…</p>';
    try{ await api("/mb/build","POST",{});
      const a=await api("/arena","POST",{target_cell_type:"MBON-GLUT"});
      out.innerHTML=`<div class="plot"><h4>4-quadrant arena</h4><p class="muted">MBON-GLUT preference index = ${a.PI?.toFixed?.(3)??a.PI}</p></div>`;
    }catch(e){ out.innerHTML=`<p class="muted" style="color:#e6556b">${e.message}</p>`; }
  }else{ runSimulation(); }
}

// ---------- experiment registry ----------
let EXP_CACHE=[];
async function loadExperiments(){
  const ul=$("expList"); if(!ul) return;
  try{
    const r=await api("/experiments");
    EXP_CACHE=r.experiments;
    const ne=EXP_CACHE.filter(e=>e.category!=="validation").length;
    const nv=EXP_CACHE.length-ne;
    $("expSummary").textContent=`${ne} experiments + ${nv} validation checks. Click one to run it.`;
    renderExpList("");
  }catch(e){ $("expSummary").textContent="engine offline"; }
}
function renderExpList(filter){
  const ul=$("expList"); ul.innerHTML="";
  EXP_CACHE.filter(e=>!filter||(e.name+e.paper).toLowerCase().includes(filter.toLowerCase()))
   .forEach(e=>{
    const li=document.createElement("li");
    li.innerHTML=`<b>${e.name}</b><br><span class="muted">${e.paper} · ${e.needs}</span>`;
    li.onclick=()=>runOneExperiment(e.name,li);
    ul.appendChild(li);
  });
}
async function runOneExperiment(name,li){
  li.style.opacity=0.6;
  try{
    const r=await api("/experiments/run","POST",{name});
    const ok=r.pass?"app":"avr";
    const obs=r.observed?Object.entries(r.observed).slice(0,4).map(([k,v])=>`${k}=${v}`).join(", "):"";
    li.innerHTML=`<b>${name}</b> <span class="tag ${ok}">${r.pass?"pass":"fail"}</span><br><span class="muted">${obs}</span>`;
  }catch(e){ li.innerHTML=`<b>${name}</b><br><span class="muted" style="color:#e6556b">${e.message}</span>`; }
  li.style.opacity=1;
}
async function runAllExperiments(){
  const btn=$("btnRunAll"); btn.textContent="running…"; btn.disabled=true;
  try{
    const r=await api("/experiments/run","POST",{});
    $("expSummary").textContent=`${r.passed}/${r.total} passed.`;
    const byName={}; r.results.forEach(x=>byName[x.experiment]=x);
    document.querySelectorAll("#expList li").forEach(li=>{
      const nm=li.querySelector("b").textContent.split(" ")[0];
      const x=byName[nm]; if(!x) return;
      const ok=x.pass?"app":"avr";
      const obs=x.observed?Object.entries(x.observed).slice(0,3).map(([k,v])=>`${k}=${v}`).join(", "):"";
      li.innerHTML=`<b>${nm}</b> <span class="tag ${ok}">${x.pass?"pass":"fail"}</span><br><span class="muted">${obs}</span>`;
    });
  }catch(e){ $("expSummary").textContent=e.message; }
  btn.textContent="Run all experiments"; btn.disabled=false;
}
// ---------- data screens ----------
async function runCensus(){
  const out=$("censusOut"); out.innerHTML='<p class="muted">running…</p>';
  try{
    const rows=await api(`/census?group=${$("selCensus").value}`);
    const top=rows.slice(0,12);
    out.innerHTML=`<table class="dtab"><tr><th>${$("selCensus").value}</th><th>n</th></tr>`+
      top.map(r=>`<tr><td>${r[$("selCensus").value]??r.name??Object.values(r)[0]}</td><td>${r.n_neurons??r.count??Object.values(r)[1]}</td></tr>`).join("")+`</table>`;
  }catch(e){ out.innerHTML=`<p class="muted" style="color:#e6556b">${e.message} (load connectome first)</p>`; }
}
async function runDisease(){
  const out=$("diseaseOut"); out.innerHTML='<p class="muted">running…</p>';
  try{
    const r=await api("/disease/atlas");
    out.innerHTML=r.results.map(x=>{
      const obs=x.observed?Object.entries(x.observed).slice(0,3).map(([k,v])=>`${k}=${v}`).join(", "):"";
      return `<div class="plot"><h4>${x.name} <span class="tag ${x.pass?"app":"avr"}">${x.pass?"pass":"fail"}</span></h4><p class="muted">${obs}</p></div>`;
    }).join("");
  }catch(e){ out.innerHTML=`<p class="muted" style="color:#e6556b">${e.message}</p>`; }
}

// ---------- tab wiring ----------
function switchTab(t){document.querySelectorAll("#leftrail .tab").forEach(b=>b.classList.toggle("active",b.dataset.tab===t));
  document.querySelectorAll("#leftrail .panel").forEach(p=>p.classList.toggle("hidden",p.dataset.panel!==t));
  if(t==="experiments"&&EXP_CACHE.length===0) loadExperiments();}
function switchOTab(t){document.querySelectorAll("#rightrail .tab").forEach(b=>b.classList.toggle("active",b.dataset.otab===t));
  document.querySelectorAll("#rightrail .panel").forEach(p=>p.classList.toggle("hidden",p.dataset.opanel!==t));
  if(t==="perturb") loadPerturbTargets();}

function wire(){
  document.querySelectorAll("#leftrail .tab").forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
  document.querySelectorAll("#rightrail .tab").forEach(b=>b.onclick=()=>switchOTab(b.dataset.otab));
  document.querySelectorAll(".traytab").forEach(b=>b.onclick=()=>{
    document.querySelectorAll(".traytab").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); renderTray(b.dataset.tray);});
  document.querySelectorAll(".presetlist li").forEach(li=>li.onclick=()=>{
    document.querySelectorAll(".presetlist li").forEach(x=>x.classList.remove("active"));
    li.classList.add("active"); drawCircuit(li.dataset.preset);});
  document.querySelectorAll(".mode").forEach(b=>b.onclick=()=>{
    document.querySelectorAll(".mode").forEach(x=>x.classList.remove("active")); b.classList.add("active");});
  $("btnLoad").onclick=async()=>{
    $("loadStatus").textContent="loading…";
    try{ const r=await api("/load_connectome","POST",
        {nodes_path:window.ENACTOME_NODES||"",edges_path:window.ENACTOME_EDGES||""});
      $("loadStatus").textContent=`Loaded ${r.n_neurons?.toLocaleString?.()||r.n_neurons} neurons.`; STATE.loaded=true;
    }catch(e){ $("loadStatus").textContent="Load via engine env vars or the API."; }};
  $("btnRun").onclick=runSimulation;
  $("btnPerturb").onclick=runPerturb;
  $("btnRunAll").onclick=runAllExperiments;
  $("expFilter").oninput=e=>renderExpList(e.target.value);
  $("btnCensus").onclick=runCensus;
  $("btnDisease").onclick=runDisease;
  [["mDA","vDA","dopamine"],["mOA","vOA","octopamine"],["m5HT","v5HT","serotonin"],["mACh","vACh","acetylcholine"]]
    .forEach(([s,v])=>{$(s).oninput=()=>$(v).textContent=parseFloat($(s).value).toFixed(2);});
}

window.addEventListener("DOMContentLoaded",()=>{
  wire(); drawCircuit("olfactory"); renderTray("drugs"); refreshEngine();
  document.querySelector('.presetlist li[data-preset="olfactory"]').classList.add("active");
  setInterval(refreshEngine,4000);
  window.addEventListener("resize",()=>drawCircuit(STATE.preset));
});
