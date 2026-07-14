const fs=require("fs"),path=require("path");
const {JSDOM}=require("jsdom");
const fetch=require("node-fetch");
const SRC=require("path").join(__dirname,"../src");
const API="http://127.0.0.1:8799";

(async()=>{
  const html=fs.readFileSync(path.join(SRC,"index.html"),"utf8");
  const errors=[];
  const dom=new JSDOM(html,{runScripts:"outside-only",pretendToBeVisual:true,url:"http://localhost/"});
  const {window}=dom;
  // wire globals the script expects
  window.fetch=(u,o)=>fetch(u.startsWith("http")?u:API+u.replace(API,""),o);
  global.fetch=window.fetch; window.SVGNS="http://www.w3.org/2000/svg";
  window.ENACTOME_NODES=""; window.ENACTOME_EDGES="";
  window.onerror=(m)=>errors.push(m);
  // inject studio.js into the window context
  const js=fs.readFileSync(path.join(SRC,"studio.js"),"utf8");
  const script=window.document.createElement("script");
  try{ window.eval(js); }catch(e){ errors.push("eval studio.js: "+e.message); }
  // fire DOMContentLoaded
  window.document.dispatchEvent(new window.Event("DOMContentLoaded"));
  await new Promise(r=>setTimeout(r,500));

  // structural checks
  const brand=window.document.querySelector(".brand").textContent;
  const tabs=[...window.document.querySelectorAll("#leftrail .tab")].map(t=>t.dataset.tab);
  const hasExp=window.document.querySelector('[data-panel="experiments"]')!==null;
  const hasData=window.document.querySelector('[data-panel="data"]')!==null;

  // functional: load experiments from live server
  let expCount=0, expText="";
  try{
    const r=await fetch(API+"/experiments"); const j=await r.json();
    expCount=j.experiments.length;
  }catch(e){ errors.push("fetch /experiments: "+e.message); }

  // run one bundle experiment through the server (what a click does)
  let runOK=false, runObs="";
  try{
    const r=await fetch(API+"/experiments/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:"disease_parkinson_bridge"})});
    const j=await r.json(); runOK=j.pass; runObs=JSON.stringify(j.observed);
  }catch(e){ errors.push("run exp: "+e.message); }

  // disease atlas screen
  let diseaseN=0;
  try{ const r=await fetch(API+"/disease/atlas"); diseaseN=(await r.json()).results.length; }catch(e){ errors.push("disease: "+e.message); }

  console.log(JSON.stringify({
    brand, tabs, hasExp, hasData,
    experiments_listed:expCount,
    run_disease_parkinson_pass:runOK, run_obs:runObs,
    disease_screen_results:diseaseN,
    js_errors:errors
  },null,1));
})();
