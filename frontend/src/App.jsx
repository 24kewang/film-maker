import { useState, useEffect, useRef, useCallback } from "react";

// ─── Mock data ────────────────────────────────────────────────────────────────
const MOCK_FRAGMENTS = [
  "Opening", "Inciting Incident", "Rising Action", "Confrontation",
  "Revelation", "Turning Point", "Crisis", "Climax", "Resolution", "Epilogue"
].map((title, i) => ({
  fragment_id: i, title,
  action: "Character moves through the rain-soaked neon alley.",
  environment: "Neo-Tokyo back alley, 2087",
}));

// 11 boundary keyframes for 10 segments
const MOCK_KF_DESCRIPTIONS = Array.from({ length: 11 }, (_, i) => ({
  keyframe_id: i,
  role: i === 0 ? "opening" : i === 10 ? "closing" : "transition",
  prompt: i === 0
    ? "Wide establishing shot. Empty neon alley, rain falling, no characters yet. Cyan and amber palette."
    : i === 10
    ? "Close-up on protagonist's face, tear blending with rain, warm amber backlight. Resolution."
    : `Transition frame between segments ${i-1} and ${i}. Character in motion, mid-action, bridging scenes.`,
}));

const mockGrad = (seed) => {
  const grads = [
    ["#0d1b2a","#1b3a5c"],["#1a0533","#0d3b6e"],["#0d3b1a","#1a3b0d"],
    ["#3b0d0d","#1a0533"],["#1a2e00","#003b3b"],["#2e1a00","#3b0d2e"],
    ["#001a3b","#0d2e1a"],["#2e0d2e","#0d1b3b"],["#1a1a00","#003b1a"],
    ["#3b1a00","#1a003b"],["#001a1a","#1a3b00"],
  ];
  const g = grads[seed % grads.length];
  return `linear-gradient(135deg, ${g[0]}, ${g[1]})`;
};

// ─── CSS ─────────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #060608; }

  :root {
    --bg: #060608;
    --surface: #0c0c10;
    --surface2: #12121a;
    --border: #1a1a28;
    --border2: #252538;
    --accent: #b8ff35;
    --accent2: #35c8ff;
    --accent3: #ff9035;
    --text: #e4e4f0;
    --muted: #56566e;
    --danger: #ff4444;
    --font: 'Syne', sans-serif;
    --mono: 'JetBrains Mono', monospace;
  }

  .app { min-height: 100vh; background: var(--bg); color: var(--text); font-family: var(--font); }

  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 40px; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .logo { font-size: 15px; font-weight: 800; letter-spacing: 2px; }
  .logo span { color: var(--accent); }
  .badge {
    background: var(--accent); color: #000; font-size: 9px; font-weight: 800;
    padding: 3px 8px; border-radius: 20px; letter-spacing: 1.5px;
  }

  .steps-nav {
    display: flex; overflow-x: auto;
    padding: 0 40px; background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .step {
    display: flex; align-items: center; gap: 8px;
    padding: 13px 18px; white-space: nowrap; cursor: default;
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
    color: var(--muted); border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }
  .step.active { color: var(--accent); border-color: var(--accent); }
  .step.done { color: var(--text); }
  .step-n {
    width: 18px; height: 18px; border-radius: 50%; font-size: 9px; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    background: var(--border); font-family: var(--mono);
  }
  .step.active .step-n { background: var(--accent); color: #000; }
  .step.done .step-n { background: rgba(184,255,53,0.15); color: var(--accent); }

  .content { padding: 40px; max-width: 1280px; margin: 0 auto; }

  .idea-stage { display: flex; flex-direction: column; align-items: center; gap: 28px; padding: 64px 0; }
  .idea-h { font-size: 52px; font-weight: 800; text-align: center; line-height: 1.05; letter-spacing: -2.5px; }
  .idea-h em { color: var(--accent); font-style: normal; }
  .idea-sub { color: var(--muted); font-size: 13px; font-family: var(--mono); }
  .idea-input {
    width: 100%; max-width: 640px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 22px; font-size: 15px; font-family: var(--font);
    color: var(--text); resize: none; min-height: 110px; outline: none; transition: border 0.2s;
  }
  .idea-input:focus { border-color: var(--accent); }
  .idea-input::placeholder { color: var(--muted); }
  .idea-btn {
    width: 100%; max-width: 640px; background: var(--accent); color: #000; border: none;
    cursor: pointer; padding: 15px; border-radius: 9px; font-size: 14px; font-weight: 800;
    font-family: var(--font); letter-spacing: 0.5px; transition: opacity 0.2s;
  }
  .idea-btn:hover { opacity: 0.88; }
  .idea-btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  .chip {
    background: var(--surface); border: 1px solid var(--border);
    padding: 6px 14px; border-radius: 20px; font-size: 11px; font-family: var(--mono);
    color: var(--muted); cursor: pointer; transition: all 0.2s;
  }
  .chip:hover { border-color: var(--accent); color: var(--accent); }

  .log-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 22px; width: 100%; max-width: 640px;
    font-family: var(--mono); font-size: 11px; max-height: 280px; overflow-y: auto;
  }
  .log-hd { font-size: 9px; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 10px; }
  .log-row { display: flex; gap: 10px; padding: 3px 0; }
  .log-ts { color: var(--muted); min-width: 52px; }
  .log-msg { color: var(--text); }
  .log-msg.done { color: var(--accent); }
  .log-msg.err  { color: var(--danger); }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spin {
    display: inline-block; width: 9px; height: 9px;
    border: 1.5px solid var(--border2); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 5px;
  }

  .sh { margin-bottom: 28px; }
  .sh h2 { font-size: 26px; font-weight: 800; letter-spacing: -1px; }
  .sh p { font-size: 12px; color: var(--muted); font-family: var(--mono); margin-top: 5px; max-width: 680px; line-height: 1.7; }

  .ref-grid { display: flex; flex-direction: column; gap: 32px; }
  .ref-label {
    font-size: 12px; font-weight: 700; letter-spacing: 0.8px; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
  }
  .ref-tag {
    font-family: var(--mono); font-size: 9px; color: var(--muted);
    background: var(--border); padding: 2px 7px; border-radius: 4px; letter-spacing: 0.5px;
  }
  .variant-row { display: flex; gap: 10px; }
  .vcard {
    flex: 1; max-width: 300px; cursor: pointer; border-radius: 8px;
    border: 2px solid transparent; overflow: hidden; position: relative;
    transition: border-color 0.15s, transform 0.1s;
  }
  .vcard.sel { border-color: var(--accent); }
  .vcard:hover:not(.sel) { border-color: var(--border2); transform: translateY(-2px); }
  .vimg { width: 100%; aspect-ratio: 16/9; display: block; }
  .vcheck {
    position: absolute; top: 7px; right: 7px;
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--accent); color: #000; font-size: 10px; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    opacity: 0; transition: opacity 0.15s;
  }
  .vcard.sel .vcheck { opacity: 1; }
  .vseed { font-family: var(--mono); font-size: 9px; color: var(--muted); padding: 5px 7px; }

  /* ── Keyframe timeline ── */
  .kf-timeline {
    border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
  }
  .kf-row {
    display: flex; align-items: stretch;
    border-bottom: 1px solid var(--border);
  }
  .kf-row:last-child { border-bottom: none; }
  .kf-id-col {
    width: 76px; min-width: 76px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 14px 0;
    border-right: 1px solid var(--border); background: var(--surface);
  }
  .kf-dot {
    width: 30px; height: 30px; border-radius: 50%; margin-bottom: 5px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 11px; font-weight: 600;
    border: 2px solid var(--border2); color: var(--muted);
  }
  .kf-dot.opening    { border-color: var(--accent);  color: var(--accent);  background: rgba(184,255,53,0.06); }
  .kf-dot.transition { border-color: var(--accent2); color: var(--accent2); background: rgba(53,200,255,0.06); }
  .kf-dot.closing    { border-color: var(--accent3); color: var(--accent3); background: rgba(255,144,53,0.06); }
  .kf-role { font-size: 8px; letter-spacing: 0.8px; font-family: var(--mono); color: var(--muted); text-transform: uppercase; }
  .kf-variants-col { flex: 1; padding: 14px 18px; }
  .kf-prompt-col {
    width: 240px; min-width: 240px; padding: 14px 16px;
    border-left: 1px solid var(--border);
    display: flex; flex-direction: column; justify-content: center;
  }
  .kf-prompt-lbl { font-size: 9px; letter-spacing: 1px; color: var(--muted); font-family: var(--mono); margin-bottom: 5px; }
  .kf-prompt-txt { font-size: 11px; font-family: var(--mono); color: #7a7a99; line-height: 1.6; }

  /* Segment connector between keyframe rows */
  .seg-connector {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 76px 8px 20px;
    background: rgba(53,200,255,0.03);
    border-bottom: 1px solid var(--border);
  }
  .seg-bar { flex: 1; height: 1px; background: linear-gradient(90deg, transparent, var(--accent2), transparent); }
  .seg-lbl { font-size: 9px; font-family: var(--mono); color: var(--accent2); white-space: nowrap; opacity: 0.7; }

  .action-bar {
    position: sticky; bottom: 0;
    background: rgba(6,6,8,0.95); backdrop-filter: blur(8px);
    border-top: 1px solid var(--border);
    padding: 16px 40px; display: flex; align-items: center; justify-content: space-between;
    z-index: 100;
  }
  .action-msg { font-size: 11px; font-family: var(--mono); color: var(--muted); }
  .btn {
    background: var(--accent); color: #000; border: none; cursor: pointer;
    padding: 12px 28px; border-radius: 7px; font-size: 13px; font-weight: 800;
    font-family: var(--font); transition: opacity 0.2s;
    display: flex; align-items: center; gap: 6px;
  }
  .btn:hover { opacity: 0.88; }
  .btn.ghost { background: transparent; color: var(--text); border: 1px solid var(--border); }

  .video-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
  .vgen-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; transition: border-color 0.3s;
  }
  .vgen-card.done { border-color: rgba(184,255,53,0.3); }
  .vgen-thumb {
    width: 100%; aspect-ratio: 16/9; background: var(--border);
    display: flex; align-items: center; justify-content: center; font-size: 24px; opacity: 0.3;
  }
  .vgen-kf-pair { display: flex; gap: 4px; padding: 8px 12px 0; }
  .vgen-kf-thumb {
    flex: 1; aspect-ratio: 16/9; border-radius: 4px; background: var(--border2);
    font-size: 8px; display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-family: var(--mono);
  }
  .kf-arr { color: var(--muted); font-size: 9px; display: flex; align-items: center; }
  .vgen-meta { padding: 10px 12px 12px; }
  .vgen-title { font-size: 12px; font-weight: 700; }
  .vgen-status { font-size: 10px; font-family: var(--mono); color: var(--muted); margin-top: 3px; }
  .vgen-status.generating { color: var(--accent2); }
  .vgen-status.done { color: var(--accent); }

  .prog-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px; margin-bottom: 28px;
  }
  .prog-hdr { display: flex; justify-content: space-between; margin-bottom: 10px; }
  .prog-lbl { font-size: 10px; font-family: var(--mono); color: var(--muted); letter-spacing: 1px; }
  .prog-count { font-size: 22px; font-weight: 800; font-family: var(--mono); color: var(--accent); }
  .prog-bar { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .prog-fill { height: 100%; background: var(--accent); transition: width 0.5s ease; }

  .final-stage { display: flex; flex-direction: column; align-items: center; gap: 28px; padding: 64px 0; }
  .film-card { width: 300px; border-radius: 14px; overflow: hidden; border: 1px solid var(--border); position: relative; }
  .film-img { width: 100%; aspect-ratio: 2/3; }
  .film-overlay {
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 36px 20px 18px;
    background: linear-gradient(transparent, rgba(0,0,0,0.97));
  }
  .film-title { font-size: 18px; font-weight: 800; letter-spacing: -0.5px; }
  .film-meta { font-size: 10px; font-family: var(--mono); color: var(--muted); margin-top: 4px; }
  .dl-btn {
    background: var(--accent); color: #000; border: none; cursor: pointer;
    padding: 14px 36px; border-radius: 9px; font-size: 14px; font-weight: 800;
    font-family: var(--font); transition: opacity 0.2s;
  }
  .dl-btn:hover { opacity: 0.88; }
`;

const STEPS = [
  { id: "idea", label: "Film Idea" },
  { id: "script", label: "Script" },
  { id: "refs", label: "References" },
  { id: "keyframes", label: "Keyframes" },
  { id: "video", label: "Generate" },
  { id: "final", label: "Final Film" },
];

const EXAMPLES = [
  "A time-traveller returns home to find their childhood erased",
  "Two rival chefs survive a food apocalypse together",
  "An AI painter falls in love with the museum it guards",
];

function StepsNav({ current }) {
  const ci = STEPS.findIndex(s => s.id === current);
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

function GradImg({ seed, style = {} }) {
  return <div style={{ background: mockGrad(seed), width: "100%", height: "100%", ...style }} />;
}

function VariantPicker({ keyId, selected = 0, onSelect }) {
  return (
    <div className="variant-row">
      {[0, 1, 2].map(i => (
        <div key={i} className={`vcard ${selected === i ? "sel" : ""}`} onClick={() => onSelect(keyId, i)}>
          <div className="vimg"><GradImg seed={Math.abs((keyId.charCodeAt(0) * 7 + i * 31) % 11)} /></div>
          <div className="vcheck">✓</div>
          <div className="vseed">variant {i + 1}</div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [stage, setStage] = useState("idea");
  const [idea, setIdea] = useState("");
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([]);
  const [sel, setSel] = useState({});
  const [videoStatus, setVideoStatus] = useState({});
  const logRef = useRef(null);

  const log = useCallback((msg, type = "") => {
    const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
    setLogs(l => [...l, { ts, msg, type }]);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const pick = (key, idx) => setSel(s => ({ ...s, [key]: idx }));

  const runPipeline = async () => {
    if (!idea.trim()) return;
    setLoading(true);
    setLogs([]);
    setStage("generating");
    const steps = [
      [500,  "Generating 10-fragment script via Claude Sonnet…"],
      [1400, "Script ready ✓  (10 × 8-second segments)", "done"],
      [400,  "Requesting character reference images from Modal / FLUX.1-dev…"],
      [2200, "Character refs ready ✓  (3 variants each)", "done"],
      [350,  "Generating environment reference…"],
      [1400, "Environment ref ready ✓", "done"],
      [450,  "Generating 11 boundary keyframe descriptions via Claude…"],
      [1800, "Keyframe descriptions ready ✓  (kf_0 … kf_10)", "done"],
      [400,  "Generating 11 × 3 = 33 keyframe images on Modal (FLUX + IP-Adapter)…"],
      [3400, "All keyframe variants ready ✓  — select your preferred images to continue", "done"],
    ];
    for (const [delay, msg, type] of steps) {
      await new Promise(r => setTimeout(r, delay));
      log(msg, type);
    }
    setLoading(false);
    setStage("refs");
  };

  const runVideoGen = async () => {
    setStage("video");
    const init = {};
    for (let i = 0; i < 10; i++) init[i] = "queued";
    setVideoStatus({ ...init });

    for (let i = 0; i < 10; i++) {
      const startDelay = 600 + Math.random() * 2400;
      const genTime    = 5000 + Math.random() * 5000;
      setTimeout(() => {
        setVideoStatus(s => ({ ...s, [i]: "generating" }));
        setTimeout(() => setVideoStatus(s => ({ ...s, [i]: "done" })), genTime);
      }, startDelay);
    }
    setTimeout(() => setStage("final"), 14000);
  };

  // ─── Stages ────────────────────────────────────────────────────────────────

  if (stage === "idea") return (
    <div className="app">
      <style>{css}</style>
      <header className="header">
        <div className="logo">FRAME<span>CRAFT</span></div>
        <div className="badge">HACKATHON DEMO</div>
      </header>
      <StepsNav current="idea" />
      <div className="content">
        <div className="idea-stage">
          <h1 className="idea-h">Turn any idea into<br /><em>a short film.</em></h1>
          <p className="idea-sub">Script → Keyframes → LTX-Video on Modal → Final cut</p>
          <textarea className="idea-input" placeholder="Describe your film idea…"
            value={idea} onChange={e => setIdea(e.target.value)} />
          <button className="idea-btn" disabled={!idea.trim()} onClick={runPipeline}>
            Generate Film →
          </button>
          <div className="chips">
            {EXAMPLES.map(e => <div key={e} className="chip" onClick={() => setIdea(e)}>{e}</div>)}
          </div>
        </div>
      </div>
    </div>
  );

  if (stage === "generating") return (
    <div className="app">
      <style>{css}</style>
      <header className="header">
        <div className="logo">FRAME<span>CRAFT</span></div>
        <div className="badge">HACKATHON DEMO</div>
      </header>
      <StepsNav current="script" />
      <div className="content">
        <div className="idea-stage">
          <h2 style={{ fontSize: 22, fontWeight: 800 }}>
            {loading ? <><span className="spin" />Running pipeline…</> : "Pipeline ready ✓"}
          </h2>
          <div className="log-panel" ref={logRef}>
            <div className="log-hd">PIPELINE LOG</div>
            {logs.map((l, i) => (
              <div key={i} className="log-row">
                <span className="log-ts">{l.ts}</span>
                <span className={`log-msg ${l.type}`}>{l.msg}</span>
              </div>
            ))}
            {loading && <div className="log-row"><span className="log-ts" /><span className="log-msg"><span className="spin" /></span></div>}
          </div>
        </div>
      </div>
    </div>
  );

  if (stage === "refs") return (
    <div className="app">
      <style>{css}</style>
      <header className="header">
        <div className="logo">FRAME<span>CRAFT</span></div>
        <div className="badge">HACKATHON DEMO</div>
      </header>
      <StepsNav current="refs" />
      <div className="content">
        <div className="sh">
          <h2>Choose reference images</h2>
          <p>Your selections are used as IP-Adapter conditioning for all 11 keyframe generations — ensuring consistent character appearance across the entire film.</p>
        </div>
        <div className="ref-grid">
          {[["Alex (protagonist)", "CHARACTER"], ["Dr. Reyes (antagonist)", "CHARACTER"]].map(([name, tag]) => (
            <div key={name}>
              <div className="ref-label">{name} <span className="ref-tag">{tag}</span></div>
              <VariantPicker keyId={`char:${name}`} selected={sel[`char:${name}`] ?? 0} onSelect={pick} />
            </div>
          ))}
          <div>
            <div className="ref-label">Neo-Tokyo Alley, 2087 <span className="ref-tag">ENVIRONMENT</span></div>
            <VariantPicker keyId="env" selected={sel["env"] ?? 0} onSelect={pick} />
          </div>
        </div>
        <div style={{ height: 80 }} />
      </div>
      <div className="action-bar">
        <div className="action-msg">Select one variant per reference · used as conditioning for all 11 keyframes</div>
        <button className="btn" onClick={() => setStage("keyframes")}>Confirm references →</button>
      </div>
    </div>
  );

  if (stage === "keyframes") return (
    <div className="app">
      <style>{css}</style>
      <header className="header">
        <div className="logo">FRAME<span>CRAFT</span></div>
        <div className="badge">HACKATHON DEMO</div>
      </header>
      <StepsNav current="keyframes" />
      <div className="content">
        <div className="sh">
          <h2>Choose keyframe variants</h2>
          <p>
            11 boundary keyframes for 10 segments. Interior keyframes (kf_1 – kf_9) are shared —
            each serves as the last frame of the preceding segment and the first frame of the next.
            kf_0 opens the film; kf_10 closes it.
          </p>
        </div>

        <div className="kf-timeline">
          {MOCK_KF_DESCRIPTIONS.map((kf, idx) => (
            <div key={kf.keyframe_id}>
              <div className="kf-row">
                <div className="kf-id-col">
                  <div className={`kf-dot ${kf.role}`}>{kf.keyframe_id}</div>
                  <div className="kf-role">{kf.role}</div>
                </div>
                <div className="kf-variants-col">
                  <VariantPicker
                    keyId={`kf_${kf.keyframe_id}`}
                    selected={sel[`kf_${kf.keyframe_id}`] ?? 0}
                    onSelect={pick}
                  />
                </div>
                <div className="kf-prompt-col">
                  <div className="kf-prompt-lbl">PROMPT EXCERPT</div>
                  <div className="kf-prompt-txt">
                    {kf.prompt.slice(0, 110)}{kf.prompt.length > 110 ? "…" : ""}
                  </div>
                </div>
              </div>

              {idx < MOCK_KF_DESCRIPTIONS.length - 1 && (
                <div className="seg-connector">
                  <div className="seg-bar" />
                  <div className="seg-lbl">
                    SEGMENT {idx} — {MOCK_FRAGMENTS[idx].title} · 8s · LTX-Video interpolates kf_{idx} → kf_{idx+1}
                  </div>
                  <div className="seg-bar" />
                </div>
              )}
            </div>
          ))}
        </div>
        <div style={{ height: 80 }} />
      </div>
      <div className="action-bar">
        <div className="action-msg">11 keyframes · defaults selected · 10 parallel LTX-Video jobs will launch</div>
        <button className="btn" onClick={runVideoGen}>Generate film clips →</button>
      </div>
    </div>
  );

  if (stage === "video") {
    const doneCount = Object.values(videoStatus).filter(v => v === "done").length;
    return (
      <div className="app">
        <style>{css}</style>
        <header className="header">
          <div className="logo">FRAME<span>CRAFT</span></div>
          <div className="badge">HACKATHON DEMO</div>
        </header>
        <StepsNav current="video" />
        <div className="content">
          <div className="sh">
            <h2>{doneCount < 10 ? <><span className="spin" />Generating clips on Modal…</> : "All clips ready ✓"}</h2>
            <p>10 LTX-Video jobs dispatched simultaneously — each runs on its own A10G GPU container (concurrency_limit=10)</p>
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
            {MOCK_FRAGMENTS.map(f => {
              const status = videoStatus[f.fragment_id] || "queued";
              return (
                <div key={f.fragment_id} className={`vgen-card ${status}`}>
                  <div className="vgen-thumb">
                    {status === "done"
                      ? <GradImg seed={(f.fragment_id * 4 + 2) % 11} style={{ width: "100%", height: "100%", opacity: 1 }} />
                      : "🎬"}
                  </div>
                  <div className="vgen-kf-pair">
                    <div className="vgen-kf-thumb">kf_{f.fragment_id}</div>
                    <div className="kf-arr">→</div>
                    <div className="vgen-kf-thumb">kf_{f.fragment_id + 1}</div>
                  </div>
                  <div className="vgen-meta">
                    <div className="vgen-title">{f.title}</div>
                    <div className={`vgen-status ${status === "generating" ? "generating" : status === "done" ? "done" : ""}`}>
                      {status === "generating" && <span className="spin" />}
                      {{ queued: "Queued", generating: "Generating…", done: "Ready ✓" }[status]}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  if (stage === "final") return (
    <div className="app">
      <style>{css}</style>
      <header className="header">
        <div className="logo">FRAME<span>CRAFT</span></div>
        <div className="badge">HACKATHON DEMO</div>
      </header>
      <StepsNav current="final" />
      <div className="content">
        <div className="final-stage">
          <h2 style={{ fontSize: 30, fontWeight: 800, textAlign: "center" }}>Your film is ready ✓</h2>
          <div className="film-card">
            <div className="film-img"><GradImg seed={7} style={{ width: "100%", height: "100%" }} /></div>
            <div className="film-overlay">
              <div className="film-title">{idea || "Untitled Film"}</div>
              <div className="film-meta">1 min 20 sec · 10 clips · 11 keyframes · ffmpeg</div>
            </div>
          </div>
          <button className="dl-btn">↓ Download MP4</button>
          <button className="btn ghost" onClick={() => { setStage("idea"); setIdea(""); setLogs([]); setSel({}); }}>
            Make another film
          </button>
        </div>
      </div>
    </div>
  );
}