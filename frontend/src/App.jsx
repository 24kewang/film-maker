import { useState, useEffect, useRef, useCallback } from "react";

// ─── Config ───────────────────────────────────────────────────────────────────
// Change this if your FastAPI server runs elsewhere
const API = "http://localhost:8000";
const WS_BASE = API.replace(/^http/, "ws");

// ─── CSS ─────────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #060608; }
  :root {
    --bg: #060608; --surface: #0c0c10; --surface2: #12121a;
    --border: #1a1a28; --border2: #252538;
    --accent: #b8ff35; --accent2: #35c8ff; --accent3: #ff9035;
    --text: #e4e4f0; --muted: #56566e; --danger: #ff4444;
    --font: 'Syne', sans-serif; --mono: 'JetBrains Mono', monospace;
  }
  .app { min-height: 100vh; background: var(--bg); color: var(--text); font-family: var(--font); }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 40px; border-bottom: 1px solid var(--border); background: var(--surface);
    position: sticky; top: 0; z-index: 200;
  }
  .logo { font-size: 15px; font-weight: 800; letter-spacing: 2px; }
  .logo span { color: var(--accent); }
  .badge { background: var(--accent); color: #000; font-size: 9px; font-weight: 800; padding: 3px 8px; border-radius: 20px; letter-spacing: 1.5px; }
  .steps-nav { display: flex; overflow-x: auto; padding: 0 40px; background: var(--surface); border-bottom: 1px solid var(--border); }
  .step { display: flex; align-items: center; gap: 8px; padding: 13px 18px; white-space: nowrap; cursor: default; font-size: 11px; font-weight: 600; letter-spacing: 0.8px; color: var(--muted); border-bottom: 2px solid transparent; transition: all 0.2s; }
  .step.active { color: var(--accent); border-color: var(--accent); }
  .step.done { color: var(--text); }
  .step-n { width: 18px; height: 18px; border-radius: 50%; font-size: 9px; font-weight: 800; display: flex; align-items: center; justify-content: center; background: var(--border); font-family: var(--mono); }
  .step.active .step-n { background: var(--accent); color: #000; }
  .step.done .step-n { background: rgba(184,255,53,0.15); color: var(--accent); }
  .content { padding: 40px; max-width: 1280px; margin: 0 auto; }

  /* Idea stage */
  .idea-stage { display: flex; flex-direction: column; align-items: center; gap: 28px; padding: 64px 0; }
  .idea-h { font-size: 52px; font-weight: 800; text-align: center; line-height: 1.05; letter-spacing: -2.5px; }
  .idea-h em { color: var(--accent); font-style: normal; }
  .idea-sub { color: var(--muted); font-size: 13px; font-family: var(--mono); }
  .idea-input { width: 100%; max-width: 640px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 22px; font-size: 15px; font-family: var(--font); color: var(--text); resize: none; min-height: 110px; outline: none; transition: border 0.2s; }
  .idea-input:focus { border-color: var(--accent); }
  .idea-input::placeholder { color: var(--muted); }
  .idea-btn { width: 100%; max-width: 640px; background: var(--accent); color: #000; border: none; cursor: pointer; padding: 15px; border-radius: 9px; font-size: 14px; font-weight: 800; font-family: var(--font); letter-spacing: 0.5px; transition: opacity 0.2s; }
  .idea-btn:hover { opacity: 0.88; }
  .idea-btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  .chip { background: var(--surface); border: 1px solid var(--border); padding: 6px 14px; border-radius: 20px; font-size: 11px; font-family: var(--mono); color: var(--muted); cursor: pointer; transition: all 0.2s; }
  .chip:hover { border-color: var(--accent); color: var(--accent); }

  /* Log panel */
  .log-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; width: 100%; max-width: 640px; font-family: var(--mono); font-size: 11px; max-height: 320px; overflow-y: auto; }
  .log-hd { font-size: 9px; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 10px; }
  .log-row { display: flex; gap: 10px; padding: 3px 0; line-height: 1.5; }
  .log-ts { color: var(--muted); min-width: 60px; flex-shrink: 0; }
  .log-msg { color: var(--text); }
  .log-msg.ok  { color: var(--accent); }
  .log-msg.err { color: var(--danger); }
  .log-msg.dim { color: var(--muted); }

  /* Spinner */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spin { display: inline-block; width: 9px; height: 9px; border: 1.5px solid var(--border2); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 6px; vertical-align: middle; }

  /* Section header */
  .sh { margin-bottom: 28px; }
  .sh h2 { font-size: 26px; font-weight: 800; letter-spacing: -1px; display: flex; align-items: center; gap: 10px; }
  .sh p { font-size: 12px; color: var(--muted); font-family: var(--mono); margin-top: 6px; max-width: 780px; line-height: 1.75; }
  .notice { display: inline-flex; align-items: center; gap: 6px; background: rgba(255,144,53,0.08); border: 1px solid rgba(255,144,53,0.2); padding: 4px 10px; border-radius: 6px; font-size: 10px; font-family: var(--mono); color: var(--accent3); margin-top: 8px; }

  /* Variant picker */
  .ref-section { display: flex; flex-direction: column; gap: 36px; }
  .ref-block { }
  .ref-label { font-size: 12px; font-weight: 700; letter-spacing: 0.6px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .ref-tag { font-family: var(--mono); font-size: 9px; color: var(--muted); background: var(--border); padding: 2px 7px; border-radius: 4px; letter-spacing: 0.5px; }
  .ref-prompt { font-family: var(--mono); font-size: 10px; color: var(--muted); margin-bottom: 10px; max-width: 680px; line-height: 1.6; }
  .variant-row { display: flex; gap: 10px; }
  .vcard { flex: 1; max-width: 320px; cursor: pointer; border-radius: 8px; border: 2px solid var(--border); overflow: hidden; position: relative; transition: border-color 0.15s, transform 0.1s; background: var(--surface2); }
  .vcard.sel { border-color: var(--accent); }
  .vcard:hover:not(.sel) { border-color: var(--border2); transform: translateY(-2px); }
  .vcard img { width: 100%; aspect-ratio: 16/9; display: block; object-fit: cover; }
  .vcard-placeholder { width: 100%; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; background: var(--border); color: var(--muted); font-size: 10px; font-family: var(--mono); }
  .vcheck { position: absolute; top: 7px; right: 7px; width: 20px; height: 20px; border-radius: 50%; background: var(--accent); color: #000; font-size: 10px; font-weight: 800; display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity 0.15s; }
  .vcard.sel .vcheck { opacity: 1; }
  .vseed { font-family: var(--mono); font-size: 9px; color: var(--muted); padding: 5px 8px; }

  /* Keyframe timeline */
  .kf-timeline { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  .kf-row { display: flex; align-items: stretch; border-bottom: 1px solid var(--border); }
  .kf-row:last-child { border-bottom: none; }
  .kf-id-col { width: 80px; min-width: 80px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 14px 0; border-right: 1px solid var(--border); background: var(--surface); }
  .kf-dot { width: 32px; height: 32px; border-radius: 50%; margin-bottom: 5px; display: flex; align-items: center; justify-content: center; font-family: var(--mono); font-size: 11px; font-weight: 600; border: 2px solid var(--border2); color: var(--muted); }
  .kf-dot.opening    { border-color: var(--accent);  color: var(--accent);  background: rgba(184,255,53,0.06); }
  .kf-dot.transition { border-color: var(--accent2); color: var(--accent2); background: rgba(53,200,255,0.06); }
  .kf-dot.closing    { border-color: var(--accent3); color: var(--accent3); background: rgba(255,144,53,0.06); }
  .kf-role { font-size: 8px; letter-spacing: 0.8px; font-family: var(--mono); color: var(--muted); text-transform: uppercase; }
  .kf-variants-col { flex: 1; padding: 14px 18px; min-width: 0; }
  .kf-prompt-col { width: 220px; min-width: 220px; padding: 14px 16px; border-left: 1px solid var(--border); display: flex; flex-direction: column; justify-content: center; }
  .kf-prompt-lbl { font-size: 9px; letter-spacing: 1px; color: var(--muted); font-family: var(--mono); margin-bottom: 5px; }
  .kf-prompt-txt { font-size: 10px; font-family: var(--mono); color: #6a6a88; line-height: 1.65; }
  .seg-connector { display: flex; align-items: center; gap: 10px; padding: 7px 20px 7px 96px; background: rgba(53,200,255,0.025); border-bottom: 1px solid var(--border); }
  .seg-bar { flex: 1; height: 1px; background: linear-gradient(90deg, transparent, var(--accent2), transparent); opacity: 0.4; }
  .seg-lbl { font-size: 9px; font-family: var(--mono); color: var(--accent2); white-space: nowrap; opacity: 0.6; }

  /* Action bar */
  .action-bar { position: sticky; bottom: 0; background: rgba(6,6,8,0.95); backdrop-filter: blur(8px); border-top: 1px solid var(--border); padding: 16px 40px; display: flex; align-items: center; justify-content: space-between; z-index: 100; }
  .action-msg { font-size: 11px; font-family: var(--mono); color: var(--muted); }
  .btn { background: var(--accent); color: #000; border: none; cursor: pointer; padding: 12px 28px; border-radius: 7px; font-size: 13px; font-weight: 800; font-family: var(--font); transition: opacity 0.2s; display: flex; align-items: center; gap: 6px; }
  .btn:hover { opacity: 0.88; }
  .btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .btn.ghost { background: transparent; color: var(--text); border: 1px solid var(--border); }

  /* Video generation */
  .video-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
  .vgen-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; transition: border-color 0.4s; }
  .vgen-card.done { border-color: rgba(184,255,53,0.3); }
  .vgen-card.generating { border-color: rgba(53,200,255,0.3); }
  .vgen-thumb { width: 100%; aspect-ratio: 16/9; background: var(--border); display: flex; align-items: center; justify-content: center; font-size: 22px; position: relative; overflow: hidden; }
  .vgen-thumb img { width: 100%; height: 100%; object-fit: cover; }
  @keyframes pulse { 0%,100% { opacity:0.3; } 50% { opacity:0.7; } }
  .vgen-thumb.pulsing { animation: pulse 1.5s ease-in-out infinite; }
  .vgen-kf-pair { display: flex; align-items: center; gap: 4px; padding: 8px 10px 4px; }
  .vgen-kf-img { flex: 1; aspect-ratio: 16/9; border-radius: 4px; overflow: hidden; background: var(--border2); }
  .vgen-kf-img img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .kf-arr { color: var(--muted); font-size: 9px; flex-shrink: 0; }
  .vgen-meta { padding: 8px 10px 12px; }
  .vgen-title { font-size: 12px; font-weight: 700; }
  .vgen-status { font-size: 10px; font-family: var(--mono); color: var(--muted); margin-top: 3px; }
  .vgen-status.generating { color: var(--accent2); }
  .vgen-status.done-c { color: var(--accent); }

  /* Progress bar */
  .prog-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 24px; margin-bottom: 28px; }
  .prog-hdr { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px; }
  .prog-lbl { font-size: 10px; font-family: var(--mono); color: var(--muted); letter-spacing: 1px; }
  .prog-count { font-size: 26px; font-weight: 800; font-family: var(--mono); color: var(--accent); }
  .prog-bar { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .prog-fill { height: 100%; background: var(--accent); transition: width 0.6s ease; border-radius: 2px; }

  /* Final */
  .final-stage { display: flex; flex-direction: column; align-items: center; gap: 28px; padding: 64px 0; }
  .film-card { width: 280px; border-radius: 14px; overflow: hidden; border: 1px solid var(--border); position: relative; }
  .film-poster { width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; }
  .film-poster-placeholder { width: 100%; aspect-ratio: 2/3; background: linear-gradient(160deg, #0d1b2a, #1b3a5c, #0d3b1a); display: flex; align-items: center; justify-content: center; font-size: 48px; }
  .film-overlay { position: absolute; bottom: 0; left: 0; right: 0; padding: 36px 20px 18px; background: linear-gradient(transparent, rgba(0,0,0,0.95)); }
  .film-title { font-size: 16px; font-weight: 800; letter-spacing: -0.5px; }
  .film-meta { font-size: 10px; font-family: var(--mono); color: var(--muted); margin-top: 4px; }
  .dl-btn { background: var(--accent); color: #000; border: none; cursor: pointer; padding: 14px 40px; border-radius: 9px; font-size: 14px; font-weight: 800; font-family: var(--font); transition: opacity 0.2s; }
  .dl-btn:hover { opacity: 0.88; }

  /* Error */
  .err-box { background: rgba(255,68,68,0.07); border: 1px solid rgba(255,68,68,0.25); border-radius: 10px; padding: 18px 22px; max-width: 640px; width: 100%; }
  .err-title { font-size: 13px; font-weight: 700; color: var(--danger); margin-bottom: 6px; }
  .err-msg { font-size: 11px; font-family: var(--mono); color: #cc8888; line-height: 1.6; }

  /* Responsive collapse of prompt column */
  @media (max-width: 900px) {
    .kf-prompt-col { display: none; }
    .content { padding: 20px; }
    .action-bar { padding: 14px 20px; }
    .steps-nav { padding: 0 20px; }
    .header { padding: 14px 20px; }
    .idea-h { font-size: 36px; }
  }
`;

const STEPS = [
  { id: "idea",      label: "Film Idea"  },
  { id: "loading",   label: "Generating" },
  { id: "refs",      label: "References" },
  { id: "keyframes", label: "Keyframes"  },
  { id: "video",     label: "Clips"      },
  { id: "final",     label: "Final Film" },
];

const STEP_IDS = STEPS.map(s => s.id);

const EXAMPLES = [
  "A lighthouse keeper discovers the beam attracts ghosts, not ships — until his missing daughter arrives",
  "Two rival chefs are the last survivors of a food apocalypse",
  "An AI that guards a museum by night falls in love with a painting",
];

// ─── WS event → human label ──────────────────────────────────────────────────
const EVENT_LABELS = {
  "script:start":          { msg: "Generating 10-fragment script via Claude Sonnet…" },
  "script:done":           { msg: "Script ready ✓  — 10 × 8-second segments", type: "ok" },
  "refs:start":            { msg: "Generating character & environment refs on Modal / FLUX.1-dev…" },
  "refs:done":             { msg: "Reference images ready ✓  — 3 variants each", type: "ok" },
  "keyframes:start":       { msg: "Generating 11 boundary keyframe descriptions via Claude…" },
  "keyframes:done":        { msg: "Keyframe descriptions ready ✓  (kf_0 … kf_10)", type: "ok" },
  "keyframe_images:start": { msg: "Generating 11 × 3 = 33 keyframe images on Modal (FLUX + IP-Adapter conditioning)…" },
  "keyframe_images:done":  { msg: "All keyframe images ready ✓  — select your preferred variants below", type: "ok" },
  "video:start":           { msg: "Dispatching 10 parallel LTX-Video jobs on Modal…" },
  "video:done":            { msg: "All 10 clips generated ✓", type: "ok" },
  "assembly:start":        { msg: "Assembling final film with ffmpeg…" },
  "assembly:done":         { msg: "Film assembled ✓", type: "ok" },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
const b64Src = (b64) => `data:image/png;base64,${b64}`;
const ts = () => new Date().toLocaleTimeString("en-GB", { hour12: false });

function StepsNav({ current }) {
  const ci = STEP_IDS.indexOf(current);
  return (
    <nav className="steps-nav">
      {STEPS.map((s, i) => (
        <div key={s.id} className={`step ${s.id === current ? "active" : ""} ${i < ci ? "done" : ""}`}>
          <div className="step-n">{i < ci ? "✓" : i + 1}</div>
          {s.label}
        </div>
      ))}
    </nav>
  );
}

// Three-variant image picker — works for both real base64 and placeholder
function VariantPicker({ keyId, variants = [], chosen = 0, onSelect }) {
  return (
    <div className="variant-row">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className={`vcard ${chosen === i ? "sel" : ""}`}
          onClick={() => onSelect(keyId, i)}
        >
          {variants[i]
            ? <img src={b64Src(variants[i])} alt={`variant ${i + 1}`} />
            : <div className="vcard-placeholder">variant {i + 1}</div>
          }
          <div className="vcheck">✓</div>
          <div className="vseed">variant {i + 1}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function App() {
  const [stage, setStage]           = useState("idea");
  const [idea, setIdea]             = useState("");
  const [sessionId, setSessionId]   = useState(null);
  const [pipelineData, setPipeline] = useState(null);  // full /state response
  const [sel, setSel]               = useState({});    // { "char:X": 0, "env": 1, "kf_0": 2, … }
  const [logs, setLogs]             = useState([]);
  const [clipsReady, setClipsReady] = useState([]);    // fragment_ids that have finished
  const [busy, setBusy]             = useState(false); // action-bar button lock
  const [error, setError]           = useState(null);

  const wsRef      = useRef(null);
  const pollRef    = useRef(null);
  const logRef     = useRef(null);
  const sessionRef = useRef(null);    // always-current sessionId for WS callbacks

  useEffect(() => { sessionRef.current = sessionId; }, [sessionId]);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [logs]);

  // Cleanup on unmount
  useEffect(() => () => {
    wsRef.current?.close();
    clearInterval(pollRef.current);
  }, []);

  const addLog = useCallback((msg, type = "") => {
    setLogs(l => [...l, { ts: ts(), msg, type }]);
  }, []);

  const pick = (key, idx) => setSel(s => ({ ...s, [key]: idx }));

  // ── POST /pipeline/start ──────────────────────────────────────────────────
  const startPipeline = async () => {
    if (!idea.trim()) return;
    setError(null);
    setLogs([]);
    setSel({});
    setClipsReady([]);
    setPipeline(null);
    setStage("loading");
    setBusy(true);

    let sid;
    try {
      const res = await fetch(`${API}/pipeline/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ idea: idea.trim() }),
      });
      if (!res.ok) throw new Error(`Server ${res.status}: ${await res.text()}`);
      const data = await res.json();
      sid = data.session_id;
    } catch (e) {
      setError(`Failed to start pipeline: ${e.message}`);
      setStage("idea");
      setBusy(false);
      return;
    }

    setSessionId(sid);
    addLog(`Pipeline started — session ${sid}`, "dim");
    openWebSocket(sid);
    setBusy(false);
  };

  // ── WebSocket ─────────────────────────────────────────────────────────────
  const openWebSocket = (sid) => {
    const ws = new WebSocket(`${WS_BASE}/pipeline/${sid}/ws`);
    wsRef.current = ws;

    // Keep-alive ping every 20s
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send("ping");
    }, 20_000);

    ws.onmessage = async (e) => {
      let parsed;
      try { parsed = JSON.parse(e.data); }
      catch { return; }

      const { event, data } = parsed;

      if (EVENT_LABELS[event]) {
        const { msg, type } = EVENT_LABELS[event];
        addLog(msg, type || "");
      }

      if (event === "ready_for_selection") {
        // Steps 1-4 done — fetch full state then show ref picker
        try {
          const res = await fetch(`${API}/pipeline/${sid}/state`);
          const state = await res.json();
          setPipeline(state);
          // Pre-populate sel defaults from server-side chosen_index
          const defaults = {};
          for (const [name] of Object.entries(state.character_refs || {}))
            defaults[`char:${name}`] = 0;
          if (state.environment_ref) defaults["env"] = 0;
          for (const key of Object.keys(state.keyframe_images || {}))
            defaults[key] = 0;
          setSel(defaults);
          setStage("refs");
        } catch (err) {
          setError(`Failed to load pipeline state: ${err.message}`);
        }
      }

      if (event === "film:ready") {
        clearInterval(pollRef.current);
        setStage("final");
      }

      if (event === "error") {
        setError(data?.message || "An unknown pipeline error occurred.");
        clearInterval(pollRef.current);
        setBusy(false);
      }
    };

    ws.onerror = () => addLog("WebSocket error — check that server.py is running", "err");
    ws.onclose = () => { clearInterval(ping); addLog("Connection closed", "dim"); };
  };

  // ── POST /pipeline/{sid}/select ───────────────────────────────────────────
  const postSelections = async (keys) => {
    const subset = {};
    for (const k of keys) if (sel[k] !== undefined) subset[k] = sel[k];
    if (Object.keys(subset).length === 0) return;
    await fetch(`${API}/pipeline/${sessionId}/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selections: subset }),
    });
  };

  // ── Confirm ref choices → show keyframe picker ────────────────────────────
  const confirmRefs = async () => {
    setBusy(true);
    try {
      const refKeys = [
        ...Object.keys(pipelineData?.character_refs || {}).map(n => `char:${n}`),
        "env",
      ];
      await postSelections(refKeys);
    } catch (e) {
      addLog(`Selection save failed: ${e.message}`, "err");
    }
    setBusy(false);
    setStage("keyframes");
  };

  // ── Confirm keyframe choices → POST /continue ─────────────────────────────
  const startVideoGen = async () => {
    setBusy(true);
    try {
      // Save kf selections
      const kfKeys = Object.keys(pipelineData?.keyframe_images || {});
      await postSelections(kfKeys);

      // Kick off video + assembly
      const res = await fetch(`${API}/pipeline/${sessionId}/continue`, { method: "POST" });
      if (!res.ok) throw new Error(`Server ${res.status}`);

      setStage("video");

      // Poll /state every 4s to update clip progress (WS only gives start/done, not per-clip)
      pollRef.current = setInterval(async () => {
        try {
          const r = await fetch(`${API}/pipeline/${sessionId}/state`);
          const s = await r.json();
          setClipsReady(s.video_clips_ready || []);
          if (s.has_final_film) clearInterval(pollRef.current);
        } catch { /* ignore transient poll errors */ }
      }, 4_000);
    } catch (e) {
      setError(`Failed to start video generation: ${e.message}`);
      setBusy(false);
    }
    setBusy(false);
  };

  // ── Download final film ───────────────────────────────────────────────────
  const downloadFilm = () => window.open(`${API}/film/${sessionId}`, "_blank");

  // ── Reset ─────────────────────────────────────────────────────────────────
  const reset = () => {
    wsRef.current?.close();
    clearInterval(pollRef.current);
    setStage("idea"); setIdea(""); setSessionId(null); setPipeline(null);
    setSel({}); setLogs([]); setClipsReady([]); setError(null); setBusy(false);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────

  const header = (
    <header className="header">
      <div className="logo">FRAME<span>CRAFT</span></div>
      <div className="badge">AI FILM PIPELINE</div>
    </header>
  );

  // ── IDEA STAGE ──────────────────────────────────────────────────────────
  if (stage === "idea") return (
    <div className="app">
      <style>{css}</style>
      {header}
      <StepsNav current="idea" />
      <div className="content">
        <div className="idea-stage">
          <h1 className="idea-h">Turn any idea into<br /><em>a short film.</em></h1>
          <p className="idea-sub">Claude → FLUX on Modal → LTX-Video on Modal → ffmpeg</p>
          <textarea
            className="idea-input"
            placeholder="Describe your film idea…"
            value={idea}
            onChange={e => setIdea(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) startPipeline(); }}
          />
          <button className="idea-btn" disabled={!idea.trim() || busy} onClick={startPipeline}>
            Generate Film →
          </button>
          {error && (
            <div className="err-box">
              <div className="err-title">Error</div>
              <div className="err-msg">{error}</div>
            </div>
          )}
          <div className="chips">
            {EXAMPLES.map(ex => (
              <div key={ex} className="chip" onClick={() => setIdea(ex)}>{ex}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  // ── LOADING STAGE ────────────────────────────────────────────────────────
  if (stage === "loading") return (
    <div className="app">
      <style>{css}</style>
      {header}
      <StepsNav current="loading" />
      <div className="content">
        <div className="idea-stage">
          <h2 style={{ fontSize: 22, fontWeight: 800 }}>
            <span className="spin" />Running pipeline…
          </h2>
          {sessionId && (
            <p style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>
              session: {sessionId}
            </p>
          )}
          <div className="log-panel" ref={logRef}>
            <div className="log-hd">PIPELINE LOG</div>
            {logs.map((l, i) => (
              <div key={i} className="log-row">
                <span className="log-ts">{l.ts}</span>
                <span className={`log-msg ${l.type}`}>{l.msg}</span>
              </div>
            ))}
            <div className="log-row">
              <span className="log-ts" />
              <span className="log-msg dim"><span className="spin" />waiting…</span>
            </div>
          </div>
          {error && (
            <div className="err-box">
              <div className="err-title">Pipeline error</div>
              <div className="err-msg">{error}</div>
            </div>
          )}
          {error && (
            <button className="btn ghost" onClick={reset}>← Start over</button>
          )}
        </div>
      </div>
    </div>
  );

  // ── REFS STAGE ───────────────────────────────────────────────────────────
  if (stage === "refs") {
    const charRefs = pipelineData?.character_refs || {};
    const envRef   = pipelineData?.environment_ref;

    return (
      <div className="app">
        <style>{css}</style>
        {header}
        <StepsNav current="refs" />
        <div className="content">
          <div className="sh">
            <h2>Choose reference images</h2>
            <p>
              Character references are used as IP-Adapter conditioning for all 11 keyframe
              generations — this is what keeps character appearance consistent across the film.
              Pick the variant that best captures the look you want.
            </p>
          </div>

          <div className="ref-section">
            {/* Character refs */}
            {Object.entries(charRefs).map(([name, gi]) => (
              <div key={name} className="ref-block">
                <div className="ref-label">
                  {name}
                  <span className="ref-tag">CHARACTER · IP-ADAPTER CONDITIONING</span>
                </div>
                <div className="ref-prompt">{gi.prompt}</div>
                <VariantPicker
                  keyId={`char:${name}`}
                  variants={gi.variants}
                  chosen={sel[`char:${name}`] ?? 0}
                  onSelect={pick}
                />
              </div>
            ))}

            {/* Environment ref */}
            {envRef && (
              <div className="ref-block">
                <div className="ref-label">
                  {pipelineData.environment_name}
                  <span className="ref-tag">ENVIRONMENT</span>
                  <span className="notice">
                    ⚠ The environment description shapes keyframe prompts.
                    This image is for reference — the selected variant is not
                    currently fed back into image generation.
                  </span>
                </div>
                <div className="ref-prompt">{envRef.prompt}</div>
                <VariantPicker
                  keyId="env"
                  variants={envRef.variants}
                  chosen={sel["env"] ?? 0}
                  onSelect={pick}
                />
              </div>
            )}
          </div>

          <div style={{ height: 90 }} />
        </div>

        <div className="action-bar">
          <div className="action-msg">
            {Object.keys(charRefs).length} character(s) · 1 environment · select 1 variant each
          </div>
          <button className="btn" disabled={busy} onClick={confirmRefs}>
            {busy ? <><span className="spin" />Saving…</> : "Confirm references →"}
          </button>
        </div>
      </div>
    );
  }

  // ── KEYFRAMES STAGE ──────────────────────────────────────────────────────
  if (stage === "keyframes") {
    const kfImages  = pipelineData?.keyframe_images || {};
    const kfDescs   = pipelineData?.keyframe_descriptions || [];
    const fragments = pipelineData?.fragments || [];

    return (
      <div className="app">
        <style>{css}</style>
        {header}
        <StepsNav current="keyframes" />
        <div className="content">
          <div className="sh">
            <h2>Choose keyframe variants</h2>
            <p>
              11 boundary keyframes for 10 segments. Interior keyframes (kf_1–kf_9) are shared —
              each serves as the last frame of the preceding segment and the first frame of the next.
              kf_0 opens the film; kf_10 closes it. LTX-Video interpolates motion between each pair.
            </p>
          </div>

          <div className="kf-timeline">
            {kfDescs.map((kf, idx) => {
              const key = `kf_${kf.keyframe_id}`;
              const gi  = kfImages[key];
              const seg = fragments[kf.keyframe_id];  // segment starting at this kf

              return (
                <div key={kf.keyframe_id}>
                  <div className="kf-row">
                    <div className="kf-id-col">
                      <div className={`kf-dot ${kf.role}`}>{kf.keyframe_id}</div>
                      <div className="kf-role">{kf.role}</div>
                    </div>

                    <div className="kf-variants-col">
                      <VariantPicker
                        keyId={key}
                        variants={gi?.variants || []}
                        chosen={sel[key] ?? 0}
                        onSelect={pick}
                      />
                    </div>

                    <div className="kf-prompt-col">
                      <div className="kf-prompt-lbl">PROMPT EXCERPT</div>
                      <div className="kf-prompt-txt">
                        {(kf.prompt || "").slice(0, 120)}{(kf.prompt || "").length > 120 ? "…" : ""}
                      </div>
                    </div>
                  </div>

                  {idx < kfDescs.length - 1 && (
                    <div className="seg-connector">
                      <div className="seg-bar" />
                      <div className="seg-lbl">
                        SEGMENT {idx} — {seg?.title || `Scene ${idx}`} · 8s ·
                        kf_{idx} → kf_{idx + 1}
                      </div>
                      <div className="seg-bar" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ height: 90 }} />
        </div>

        <div className="action-bar">
          <div className="action-msg">
            11 keyframes · 10 LTX-Video jobs will launch in parallel
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn ghost" onClick={() => setStage("refs")}>← Back</button>
            <button className="btn" disabled={busy} onClick={startVideoGen}>
              {busy ? <><span className="spin" />Starting…</> : "Generate film clips →"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── VIDEO STAGE ──────────────────────────────────────────────────────────
  if (stage === "video") {
    const fragments    = pipelineData?.fragments || [];
    const kfImages     = pipelineData?.keyframe_images || {};
    const doneCount    = clipsReady.length;
    const doneSet      = new Set(clipsReady);

    return (
      <div className="app">
        <style>{css}</style>
        {header}
        <StepsNav current="video" />
        <div className="content">
          <div className="sh">
            <h2>
              {doneCount < 10
                ? <><span className="spin" />Generating clips on Modal…</>
                : "All clips ready — assembling film…"}
            </h2>
            <p>
              10 LTX-Video jobs dispatched simultaneously — each runs on its own A100-40GB
              GPU container (concurrency_limit=10). First clip may take 3–5 min for cold start;
              subsequent clips reuse warm containers.
            </p>
          </div>

          <div className="prog-wrap">
            <div className="prog-hdr">
              <span className="prog-lbl">CLIPS COMPLETE</span>
              <span className="prog-count">{doneCount} / 10</span>
            </div>
            <div className="prog-bar">
              <div className="prog-fill" style={{ width: `${doneCount * 10}%` }} />
            </div>
          </div>

          <div className="video-grid">
            {fragments.map(f => {
              const isDone  = doneSet.has(f.fragment_id);
              const isGoing = !isDone && doneCount > 0;
              const status  = isDone ? "done" : isGoing ? "generating" : "queued";

              const firstKey = `kf_${f.fragment_id}`;
              const lastKey  = `kf_${f.fragment_id + 1}`;
              const firstGi  = kfImages[firstKey];
              const lastGi   = kfImages[lastKey];
              const firstB64 = firstGi?.variants?.[sel[firstKey] ?? 0];
              const lastB64  = lastGi?.variants?.[sel[lastKey] ?? 0];

              return (
                <div key={f.fragment_id} className={`vgen-card ${status}`}>
                  <div className={`vgen-thumb${status === "generating" ? " pulsing" : ""}`}>
                    {isDone ? "✅" : status === "generating" ? <span className="spin" /> : "🎬"}
                  </div>

                  <div className="vgen-kf-pair">
                    <div className="vgen-kf-img">
                      {firstB64
                        ? <img src={b64Src(firstB64)} alt={firstKey} />
                        : <div style={{ width: "100%", height: "100%", background: "var(--border2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "var(--muted)", fontFamily: "var(--mono)" }}>{firstKey}</div>
                      }
                    </div>
                    <div className="kf-arr">→</div>
                    <div className="vgen-kf-img">
                      {lastB64
                        ? <img src={b64Src(lastB64)} alt={lastKey} />
                        : <div style={{ width: "100%", height: "100%", background: "var(--border2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "var(--muted)", fontFamily: "var(--mono)" }}>{lastKey}</div>
                      }
                    </div>
                  </div>

                  <div className="vgen-meta">
                    <div className="vgen-title">{f.title}</div>
                    <div className={`vgen-status${isDone ? " done-c" : status === "generating" ? " generating" : ""}`}>
                      {status === "generating" && <span className="spin" />}
                      {{ queued: "Queued", generating: "Generating…", done: "Ready ✓" }[status]}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {error && (
            <div className="err-box" style={{ marginTop: 28 }}>
              <div className="err-title">Error during generation</div>
              <div className="err-msg">{error}</div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── FINAL STAGE ──────────────────────────────────────────────────────────
  if (stage === "final") {
    const fragments = pipelineData?.fragments || [];

    return (
      <div className="app">
        <style>{css}</style>
        {header}
        <StepsNav current="final" />
        <div className="content">
          <div className="final-stage">
            <h2 style={{ fontSize: 30, fontWeight: 800, textAlign: "center" }}>
              Your film is ready ✓
            </h2>
            <div className="film-card">
              <div className="film-poster-placeholder">🎬</div>
              <div className="film-overlay">
                <div className="film-title">{idea || "Untitled Film"}</div>
                <div className="film-meta">
                  1 min 20 sec · 10 clips · 11 keyframes · session {sessionId}
                </div>
              </div>
            </div>
            <button className="dl-btn" onClick={downloadFilm}>↓ Download MP4</button>
            <button className="btn ghost" onClick={reset}>Make another film</button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}