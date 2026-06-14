"use strict";
// Trace-flow visualization. Renders one task's orchestrator events as a per-revision flow
// of color-coded operator nodes. Input is the live per-attempt event buffers the client
// accumulates from the task's SSE stream: `attempts` = [{attempt_id, events:[...]}, ...] in
// order. Expand state lives in JS (keyed by task id); the client syncs <details> open state
// from the DOM into this set right before each re-render, so streaming new events never
// collapses what the user opened.

(function () {
  const KIND = {
    system:      { color: "#888888", bg: "var(--bg2)",      label: "SYSTEM" },
    user:        { color: "#e6b800", bg: "var(--yellow-bg)", label: "USER" },
    assistant:   { color: "#33cc33", bg: "var(--green-bg)",  label: "ASSISTANT" },
    observation: { color: "#3399ff", bg: "var(--blue-bg)",   label: "OBSERVATION" },
    error:       { color: "#ff4444", bg: "var(--red-bg)",    label: "ERROR" },
    call:        { color: "#666666", bg: "transparent",      label: "LLM CALL" },
    note:        { color: "#888888", bg: "transparent",      label: "NOTE" },
    step:        { color: "#666666", bg: "transparent",      label: "STEP" },
  };
  const OP_COLOR = { planner: "#cc66ff", question_explainer: "#66ccaa", retrieve: "#3399ff", extract: "#e6b800", lookup_external: "#ff9933", compute: "#33cc33", replanner: "#ff66aa" };
  const COMPACT_KINDS = new Set(["call", "note", "step"]);
  const TRUNC_CHARS = 40000; // observations longer than this collapse with a click-to-expand

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));
  const tok = (n) => Math.round(n / 4).toLocaleString();
  const opColor = (op) => OP_COLOR[op] || "#999999";
  const fsec = (s) => (s == null ? "?" : (+s).toFixed(2) + "s");

  // ── per-task view state ─────────────────────────────────────────────────────
  // activeRev=null means "follow the latest revision"; a tab click pins an index. `expanded`
  // holds open node keys ("n:<step>") and expanded long-event keys ("e:<idx>"). Node keys are
  // refreshed from the live DOM (via captureOpen) before each render; event keys are managed
  // by toggleEvent (no DOM equivalent — they toggle inline text, not a <details>).
  const VIEW = new Map();
  function viewState(taskId) {
    let vs = VIEW.get(taskId);
    if (!vs) { vs = { activeRev: null, expanded: new Set() }; VIEW.set(taskId, vs); }
    return vs;
  }

  window.TraceFlow = {
    onChange: null, // client sets this to its trace re-render function
    html(taskId, attempts) { return renderTask(taskId, attempts || []); },
    captureOpen(taskId, root) {
      // Sync node <details> open state from the live DOM into the expanded set so a rebuild
      // restores exactly what the user has open right now.
      const e = viewState(taskId).expanded;
      root.querySelectorAll("details.node[data-key]").forEach((d) => {
        const key = d.getAttribute("data-key");
        if (d.open) e.add(key); else e.delete(key);
      });
    },
    toggleEvent(taskId, key) { const e = viewState(taskId).expanded; if (e.has(key)) e.delete(key); else e.add(key); if (this.onChange) this.onChange(); },
    setRev(taskId, i) { viewState(taskId).activeRev = i; if (this.onChange) this.onChange(); },
  };

  // ── flatten per-attempt buffers into one continuous event stream ────────────
  // Each attempt restarts the orchestrator (t resets to ~0, step_idx restarts at 0); re-base
  // both per attempt so attempts become disjoint nodes / revision tabs instead of colliding.
  function flattenAttemptEvents(attempts) {
    const out = [];
    let tOffset = 0;
    let stepOffset = 0;
    for (const attempt of attempts) {
      const events = attempt.events || [];
      let maxT = 0;
      let maxStep = -1;
      for (const e of events) {
        if (typeof e.t === "number" && e.t > maxT) maxT = e.t;
        if (typeof e.step_idx === "number" && e.step_idx > maxStep) maxStep = e.step_idx;
      }
      for (const e of events) {
        const reb = { ...e, _idx: out.length };
        if (typeof e.t === "number") reb.t = e.t + tOffset;
        if (e.step_idx != null) reb.step_idx = e.step_idx + stepOffset;
        out.push(reb);
      }
      tOffset += maxT;
      stepOffset += maxStep + 1;
    }
    return out;
  }

  // ── model building ──────────────────────────────────────────────────────────
  function buildModel(events) {
    const plans = events.filter((e) => e.kind === "plan");
    const byStep = new Map();
    const order = [];
    const orphans = [];
    for (const e of events) {
      if (e.kind === "plan") continue;
      if (e.step_idx == null) { orphans.push(e); continue; }
      let g = byStep.get(e.step_idx);
      if (!g) { g = { step_idx: e.step_idx, op: e.op, branchId: null, boundary: null, events: [], startT: e.t, endT: e.t }; byStep.set(e.step_idx, g); order.push(e.step_idx); }
      if (!g.op && e.op) g.op = e.op;
      // branch_id is stamped on every event of a per-branch step, so a retrieve/lookup
      // rollout groups under its branch live — not only once the boundary step lands.
      if (e.branch_id != null && g.branchId == null) g.branchId = e.branch_id;
      if (e.t != null) { if (g.startT == null || e.t < g.startT) g.startT = e.t; if (g.endT == null || e.t > g.endT) g.endT = e.t; }
      if (e.kind === "step") g.boundary = e; else g.events.push(e);
    }
    order.sort((a, b) => a - b);
    const nodes = order.map((i) => finalizeNode(byStep.get(i)));
    const branchInfo = new Map();
    for (const p of plans) { for (const b of (p.data?.branches || [])) { if (b.branch_id != null) branchInfo.set(b.branch_id, b); } }
    const orchestration = nodes.filter((n) => n.branchId == null);
    return { plans, nodes, orchestration, orphans, branchInfo };
  }
  function finalizeNode(g) {
    const d = (g.boundary && typeof g.boundary.data === "object" && g.boundary.data) || {};
    const end = g.boundary?.t ?? g.endT;
    const indiv = d.elapsed_s;
    const start = (indiv != null && end != null) ? Math.max(0, +(end - indiv).toFixed(3)) : g.startT;
    return { ...g, branchId: g.branchId ?? d.branch_id ?? null, failed: !!d.error || g.boundary?.level === "warning", summary: d.summary, error: d.error, indiv, start, end };
  }

  // ── top-level render for one task ───────────────────────────────────────────
  function renderTask(taskId, attempts) {
    const events = flattenAttemptEvents(attempts);
    if (!events.length) return `<div class="trace-empty">No trace events yet.</div>`;
    const m = buildModel(events);
    const orphans = m.orphans.length
      ? `<div class="card"><div class="card-title">Other events</div>${m.orphans.map((e) => eventHtml(e, taskId)).join("")}</div>`
      : "";
    return `<div class="trace-flow">${flowSection(m, taskId)}${orphans}</div>`;
  }

  // ── merged execution flow: cumulative revision tabs, linearized node flow ────
  function flowSection(m, taskId) {
    const nodes = m.nodes.slice().sort((a, b) => a.step_idx - b.step_idx);
    const segments = [];
    let cur = [];
    for (const n of nodes) { cur.push(n); if (n.op === "compute") { segments.push(cur); cur = []; } }
    if (cur.length) segments.push(cur);
    if (!segments.length) return `<div class="card"><div class="card-title">Execution</div><div class="trace-empty">no operator steps</div></div>`;

    const nTabs = segments.length;
    const vs = viewState(taskId);
    const sel = vs.activeRev == null ? nTabs - 1 : Math.min(vs.activeRev, nTabs - 1);
    const tabs = segments.map((s, i) => {
      const label = m.plans[i]?.data?.label || `attempt ${i + 1}`;
      return `<div class="tab ${i === sel ? "active" : ""}" onclick="TraceFlow.setRev('${esc(taskId)}', ${i})">${esc(label)}${i === nTabs - 1 ? " ★" : ""}</div>`;
    }).join("");

    const items = [];
    for (let i = 0; i <= sel; i++) items.push(segmentHtml(segments[i], i, m, sel, taskId));
    return `<div class="card"><div class="card-title">Execution — ${nTabs} revision(s)</div>
      <div class="tabs">${tabs}</div>${items.join(arrow())}</div>`;
  }
  const arrow = () => `<div class="flow-arrow">↓</div>`;

  function segmentHtml(seg, i, m, sel, taskId) {
    const items = [];
    const rev = m.plans[i];
    if (i > 0 && rev?.data?.reason != null) {
      const dec = replannerReasonText(m, rev.t);
      items.push(`<div class="reason-box">
        <div class="rl">Replan trigger</div><div class="mono">${esc(rev.data.reason)}</div>
        ${rev.data.missing?.length ? `<div class="node-meta" style="margin-top:4px">missing: ${esc(rev.data.missing.join(", "))}</div>` : ""}
        ${dec ? `<div class="rl" style="margin-top:8px">Replanner decision</div><div class="msg-text">${esc(dec)}</div>` : ""}
      </div>`);
    }
    const setup = seg.filter((n) => n.op === "planner" || n.op === "question_explainer");
    const replanner = seg.filter((n) => n.op === "replanner");
    const compute = seg.filter((n) => n.op === "compute");
    // Selection (block_select) is the precision stage AFTER the retrieval funnel. It's per-branch,
    // but render it as its own "selection" section right after retrieve — NOT buried inside each
    // branch group — so it's visible on the page-index path (where retrieve is branchless). Exclude
    // it from the branch nodes (else it'd render twice).
    const selects = seg.filter((n) => n.op === "block_select");
    const branchNodes = seg.filter((n) => n.branchId != null && n.op !== "block_select");
    // The page-index retriever is a single shared sweep with no branch_id, so its step
    // doesn't group under a branch — render its funnel (ToC pick → date filter → sem filter)
    // standalone. `others` catches any remaining branch-less op so nothing is silently dropped.
    const retrieves = seg.filter((n) => n.branchId == null && n.op === "retrieve");
    const handledOps = new Set(["planner", "question_explainer", "replanner", "compute", "retrieve", "block_select"]);
    const others = seg.filter((n) => n.branchId == null && !handledOps.has(n.op));

    for (const n of replanner) items.push(nodeHtml(n, taskId));
    if (setup.length) items.push(`<div class="flow-group"><div class="flow-head"><span class="flow-label">planning${setup.length > 1 ? " · parallel" : ""}</span></div>${setup.map((n) => nodeHtml(n, taskId)).join("")}</div>`);

    for (const n of retrieves) items.push(retrieveFunnelHtml(n, taskId));
    if (selects.length) items.push(selectionSectionHtml(selects, m, taskId));

    const byBranch = new Map();
    for (const n of branchNodes) { if (!byBranch.has(n.branchId)) byBranch.set(n.branchId, []); byBranch.get(n.branchId).push(n); }
    for (const [bid, bnodes] of byBranch) items.push(branchGroupHtml(bid, bnodes, m, sel, taskId));

    for (const n of others) items.push(nodeHtml(n, taskId));
    for (const n of compute) items.push(nodeHtml(n, taskId));
    return items.join(arrow());
  }

  function branchGroupHtml(bid, bnodes, m, sel, taskId) {
    const info = m.branchInfo.get(bid) || { kind: (bnodes[0]?.op || "?") };
    const kind = info.kind || "?";
    const color = opColor(kind === "retrieve" ? "retrieve" : "lookup_external");
    const focusIds = new Set((m.plans[sel]?.data?.branches || []).map((b) => b.branch_id));
    const firstIds = new Set((m.plans[0]?.data?.branches || []).map((b) => b.branch_id));
    const status = !focusIds.has(bid) ? "dropped" : (!firstIds.has(bid) ? "added" : "executed");
    const muted = status === "dropped";
    const keyText = kind === "retrieve" ? (info.key || "") : (info.target || "");
    const fields = kind === "retrieve"
      ? [info.period && field("period", info.period), info.as_of && field("as_of", info.as_of), info.visual_only && field("visual_only", "true")]
      : [info.src && field("src", info.src)];
    let pageChips = "";
    for (const n of bnodes) { const s = n.summary; if (s && s.type === "pages") { pageChips = (s.pages || []).map((p) => `<span class="chip chip-retr">${esc(p.month)} p${esc(p.page)}</span>`).join("") || `<span class="chip chip-none">0 pages</span>`; } }
    const fieldsHtml = fields.filter(Boolean).join(" · ");
    const ordered = bnodes.slice().sort((a, b) => a.step_idx - b.step_idx);
    return `<div class="flow-group ${muted ? "muted" : ""}">
      <div class="flow-head">
        <span class="flow-label">branch</span>
        <span class="op-badge" style="background:${color}22;color:${color}">${esc(kind)}</span>
        <span class="status ${status}">${status}</span>
        <span class="flow-key">${esc(keyText)}</span>
        ${fieldsHtml ? `<div class="flow-fields">${fieldsHtml}</div>` : ""}
        ${pageChips ? `<div class="flow-pages">${pageChips}</div>` : ""}
      </div>
      ${ordered.map((n) => nodeHtml(n, taskId)).join("")}
    </div>`;
  }
  function field(k, v) { return `<span><b>${esc(k)}</b>: ${esc(v)}</span>`; }
  function replannerReasonText(m, revT) {
    const cands = m.orchestration.filter((n) => n.op === "replanner" && (n.end ?? 0) <= (revT ?? Infinity) + 0.001);
    const node = cands.sort((a, b) => (b.end ?? 0) - (a.end ?? 0))[0];
    if (!node) return "";
    const a = node.events.find((e) => e.kind === "assistant");
    return a ? (a.data?.text || a.message || "") : "";
  }

  // ── operator node (open-state persisted via data-key + captureOpen) ──────────
  function nodeHtml(node, taskId) {
    const op = node.op || "?";
    const color = opColor(op);
    const summary = node.failed ? `<span class="err">✗ ${esc(node.error || "failed")}</span>` : `<span>${esc(summaryText(node.summary))}</span>`;
    const timing = (node.start != null && node.end != null)
      ? `<span class="timing">${fsec(node.start)} → ${fsec(node.end)} · Δ ${fsec(node.indiv ?? (node.end - node.start))} · cum ${fsec(node.end)}</span>` : "";
    const body = node.op === "block_select"
      ? tournamentHtml(node, taskId)
      : (node.events.length ? node.events.map((e) => eventHtml(e, taskId)).join("") : `<div class="trace-empty">no internal events</div>`);
    const key = `n:${node.step_idx}`;
    const open = viewState(taskId).expanded.has(key) ? "open" : "";
    return `<details class="node ${node.failed ? "failed" : ""}" data-key="${key}" ${open}>
      <summary>
        <span class="ti">▶</span>
        <span class="op-badge" style="background:${color}22;color:${color}">${esc(op)}</span>
        ${timing}
        <span class="node-summary">${summary}</span>
      </summary>
      <div class="node-body">${body}</div>
    </details>`;
  }
  function summaryText(s) {
    if (!s || typeof s !== "object") return "(no summary)";
    switch (s.type) {
      case "pages": { const ps = (s.pages || []).map((p) => `${p.month ?? "?"} p${p.page ?? "?"}`); return `${ps.length} page${ps.length === 1 ? "" : "s"}` + (ps.length ? ": " + ps.join(", ") : ""); }
      case "values": { const vs = (s.values || []).map((v) => `${v.description}${v.unit ? ` (${v.unit})` : ""}`); return `${vs.length} value${vs.length === 1 ? "" : "s"}` + (vs.length ? ": " + vs.join("; ") : ""); }
      case "answer": return `answer: ${s.answer}`;
      case "plan": return `${(s.branches || []).length} branch(es)`;
      case "list": return `${s.n} item(s)`;
      case "none": return "(none)";
      case "scalar": return String(s.value);
      default: return s.value != null ? JSON.stringify(s.value) : String(s.type || "");
    }
  }

  // ── one event (color-coded message, timestamp + truncation) ─────────────────
  // System prompts and per-call LLM lines were debug-only toggles; we keep the trace
  // readable by always collapsing system prompts to a marker and omitting call lines.
  function eventHtml(evt, taskId) {
    const kind = evt.kind || "note";
    if (kind === "system") return `<div class="msg compact"><div class="msg-line">[system prompt]</div></div>`;
    if (kind === "call") return "";
    const style = KIND[kind] || KIND.note;
    const ts = evt.t != null ? `@${fsec(evt.t)}` : "";
    if (COMPACT_KINDS.has(kind)) {
      return `<div class="msg compact" style="border-left-color:${style.color}"><div class="msg-line">${ts ? `<span style="color:var(--text3)">${ts} </span>` : ""}${esc(evt.message || "")}</div></div>`;
    }
    const key = `e:${evt._idx}`;
    const expanded = viewState(taskId).expanded.has(key);
    const { html, total, truncated } = renderBody(evt, expanded);
    const tk = total ? `~${tok(total)} tok` : "";
    const btn = (truncated || (expanded && total > TRUNC_CHARS))
      ? `<button class="expand-btn" onclick="TraceFlow.toggleEvent('${esc(taskId)}', '${key}')">${expanded ? "▲ collapse" : `▼ show full (~${tok(total)} tokens)`}</button>` : "";
    return `<div class="msg" style="border-left-color:${style.color};background:${style.bg}">
      <div class="msg-bar" style="color:${style.color}"><span>${style.label}</span><span class="meta">${ts} ${tk}</span></div>
      ${html}${btn}
    </div>`;
  }
  function renderBody(evt, expanded) {
    const d = (evt.data && typeof evt.data === "object") ? evt.data : {};
    if (evt.kind === "observation" && Array.isArray(d.blocks)) {
      const total = d.blocks.reduce((n, b) => n + (b.text || "").length, 0);
      if (total <= TRUNC_CHARS || expanded) return { html: d.blocks.map(blockHtml).join("") || `<span class="trace-empty">empty</span>`, total, truncated: false };
      let used = 0;
      const out = [];
      for (const b of d.blocks) {
        const len = (b.text || "").length;
        if (used + len <= TRUNC_CHARS) { out.push(blockHtml(b)); used += len; }
        else { out.push(blockHtml({ ...b, text: (b.text || "").slice(0, Math.max(0, TRUNC_CHARS - used)) + " …" })); break; }
      }
      return { html: out.join(""), total, truncated: true };
    }
    const text = typeof d.text === "string" ? d.text : (evt.message || "");
    if (text.length <= TRUNC_CHARS || expanded) return { html: `<div class="msg-text">${esc(text)}</div>`, total: text.length, truncated: false };
    return { html: `<div class="msg-text">${esc(text.slice(0, TRUNC_CHARS))} …</div>`, total: text.length, truncated: true };
  }
  function blockHtml(b) {
    if (b && b.type === "chunk") {
      return `<div class="chunk"><div class="chunk-meta">chunk_id=${esc(b.chunk_id ?? "")} · doc_id=${esc(b.doc_id ?? "")}</div><pre class="chunk-text">${esc(b.text ?? "")}</pre></div>`;
    }
    return `<div class="msg-text">${esc((b && b.text) ?? "")}</div>`;
  }

  // ── page-index retrieval funnel + block-select tournament ────────────────────
  // Ported from eval/trace_viewer/app.html. The page-index retriever emits its stages as
  // ctx.emit events grouped under the shared `retrieve` step (no branch_id); the selection
  // tournament emits `block_select_call` events under each branch's `block_select` step.
  // The competition harness has no ground truth, so gold-page marking is omitted.
  const msg0 = (e) => e.message || "";
  // USD per token (input, output); thinking billed at the output rate.
  const PRICES = {
    "gemini-3.5-flash":       [0.30e-6, 2.50e-6],
    "gemini-3.1-flash-lite":  [0.10e-6, 0.40e-6],
    "gemini-3.1-pro-preview": [2.00e-6, 12.00e-6],
  };
  function callCost(model, inTok, outTok, thinkTok) {
    const p = PRICES[model];
    return p ? (inTok || 0) * p[0] + ((outTok || 0) + (thinkTok || 0)) * p[1] : null;
  }
  const fusd = (c) => (c == null ? "$?" : "$" + (c >= 0.1 ? c.toFixed(2) : c.toFixed(4)));
  // "call call_site=... model=... latency_s=... in_tok=... out_tok=... think_tok=..." envelope line
  function parseCallMsg(msg) {
    const m = /call call_site=(\S+) model=(\S+).*?latency_s=([\d.]+).*?in_tok=(\d+|None) out_tok=(\d+|None) think_tok=(\d+|None)/.exec(msg || "");
    if (!m) return null;
    const num = (s) => (s === "None" ? 0 : +s);
    return { site: m[1], model: m[2], latency: +m[3], inTok: num(m[4]), outTok: num(m[5]), thinkTok: num(m[6]) };
  }

  function stageHtml(name, stat, _survivorSet, body) {
    // Show ONLY the elimination rate (carried in `stat`, e.g. "kept 8/120 pages") — NOT a per-page
    // chip list, which floods ToC/date/sem with dozens of tags on a wide retrieval. `_survivorSet`
    // stays in the signature (callers still pass it) but is deliberately not rendered.
    return `<details class="stage"><summary><span class="ti">▶</span>
      <span class="stage-name">${esc(name)}</span><span class="stage-stat">${esc(stat)}</span></summary>
      <div class="stage-body">${body || `<span class="trace-empty">no detail</span>`}</div></details>`;
  }
  function rawEventsHtml(node, taskId) {
    return `<details class="node"><summary><span class="ti">▶</span><span class="node-summary">raw events (${node.events.length})</span></summary>
      <div class="node-body">${node.events.map((e) => eventHtml(e, taskId)).join("")}</div></details>`;
  }

  // ── retrieval funnel: ToC pick → date filter → semantic filter ──────────────
  function retrieveFunnelHtml(node, taskId) {
    const evts = node.events;
    const toc = evts.find((e) => msg0(e).startsWith("toc_pick_pages"));
    const sem = evts.find((e) => msg0(e).startsWith("semantic_filter"));
    if (!toc && !sem) return nodeHtml(node, taskId);   // golden / search_agent / old trace
    const stages = [];

    // 1 — ToC pick: per-era chapters picked, pages surviving (structured `data` on new runs;
    // parsed from the message repr on older ones).
    const pickInfo = (e) => {
      if (e.data?.picked) return e.data;
      const m = /pick_chapters era=\((.*?)\) picked=\[(.*?)\] pages=(\d+)/.exec(msg0(e));
      if (!m) return null;
      const strs = (s) => [...s.matchAll(/'([^']*)'|"([^"]*)"/g)].map((x) => x[1] ?? x[2]);
      return { era: strs(m[1]), picked: strs(m[2]), n_chapters: null, pages: +m[3] };
    };
    const eras = evts.filter((e) => msg0(e).startsWith("pick_chapters ")).map(pickInfo).filter(Boolean);
    const tocPages = toc?.data?.pages ? new Set(toc.data.pages) : null;
    const tocN = tocPages ? tocPages.size
      : ((/toc_pick_pages n=(\d+)/.exec(msg0(toc || {})) || [])[1]
         ?? `~${eras.reduce((n, d) => n + (d.pages || 0), 0)}`);
    const nCh = eras.reduce((n, d) => n + (d.picked || []).length, 0);
    const eraRows = eras.map((d) => {
      const chips = (d.picked || []).map((c) => `<span class="chip">${esc(c)}</span>`).join("") || `<span class="chip chip-none">no chapters picked</span>`;
      return `<div style="margin:5px 0"><div class="stage-stat mono">era ${esc((d.era || []).join(" → "))} · ${(d.picked || []).length}${d.n_chapters ? `/${d.n_chapters}` : ""} chapters · ${d.pages} pages</div>${chips}</div>`;
    }).join("");
    stages.push(stageHtml("ToC pick", `${nCh} chapters picked → ${tocN} pages`, tocPages, eraRows));

    // 2 — date filter: per-branch kept counts; the union of branch survivors.
    const yfs = evts.filter((e) => msg0(e).startsWith("year_filter"));
    if (yfs.length) {
      const hasPages = yfs.some((e) => e.data?.pages);
      const union = hasPages ? new Set() : null;
      let keptTotal = 0;
      const rows = yfs.map((e) => {
        const d = e.data || {};
        const km = /kept=(\d+)\/(\d+)/.exec(msg0(e)) || [];
        const kept = d.pages ? d.pages.length : (km[1] ?? "?");
        const total = d.total ?? km[2] ?? "?";
        const period = d.period ?? (/period='([^']*)'/.exec(msg0(e)) || [])[1];
        if (union && d.pages) for (const p of d.pages) union.add(p);
        keptTotal += +kept || 0;
        return `<div class="stage-stat" style="margin:4px 0">${d.key ? `<b>${esc(d.key)}</b> · ` : ""}${period ? `${esc(period)} · ` : ""}kept ${kept}/${total}</div>`;
      }).join("");
      const nLeft = union ? union.size : keptTotal;
      stages.push(stageHtml("Date filter", `${yfs.length} branch(es) → ${nLeft} pages${union ? "" : " (sum, may overlap)"}`, union, rows));
    }

    // 3 — semantic filter: calls / cost / wallclock, pages left.
    if (sem) {
      const d = sem.data || {};
      const pages = d.pages ? new Set(d.pages) : null;
      const sm = /kept=(\d+)\/(\d+) blocks_kept=(\d+) blocks=(\d+)/.exec(msg0(sem)) || [];
      const calls = evts.filter((e) => e.kind === "call" && msg0(e).includes("call_site=semfilter"))
                        .map((e) => ({ ...(parseCallMsg(e.message) || {}), t: e.t })).filter((c) => c.model);
      const cost = calls.reduce((s, c) => s + (callCost(c.model, c.inTok, c.outTok, c.thinkTok) || 0), 0);
      const wall = calls.length ? Math.max(...calls.map((c) => c.t)) - Math.min(...calls.map((c) => c.t - c.latency)) : null;
      const stat = `${calls.length || d.n_calls || "?"} calls · ${fusd(calls.length ? cost : null)} · wall ${fsec(wall)} → `
                 + `${pages ? pages.size : sm[1] ?? "?"}/${d.total ?? sm[2] ?? "?"} pages (${d.blocks_kept ?? sm[3] ?? "?"}/${d.blocks ?? sm[4] ?? "?"} blocks)`;
      const rows = calls.map((c, i) => `<div class="stage-stat mono">call ${i + 1}: ${fsec(c.latency)} · ${fusd(callCost(c.model, c.inTok, c.outTok, c.thinkTok))} · in ${c.inTok.toLocaleString()} tok · out ${(c.outTok + c.thinkTok).toLocaleString()} tok</div>`).join("");
      stages.push(stageHtml("Sem filter", stat, pages, rows));
    }

    return `<div class="flow-group">
      <div class="flow-head"><span class="flow-label">retrieval funnel</span>
        ${node.indiv != null ? `<span class="timing">Δ ${fsec(node.indiv)}</span>` : ""}
        ${node.failed ? `<span class="node-summary err">✗ ${esc(node.error || "failed")}</span>` : ""}
      </div>
      ${stages.join("")}
      ${rawEventsHtml(node, taskId)}
    </div>`;
  }

  // ── selection section (block_select, the stage after sem filter) ─────────────
  // Page-index retrieve is branchless (one funnel); selection runs PER BRANCH. Render it as one
  // "selection" section right after the funnel, each branch a collapsible tournament — so it reads
  // as the 4th funnel stage (ToC → date → sem → selection) instead of hiding inside branch groups.
  function selectionSectionHtml(selects, m, taskId) {
    let totIn = 0, totSel = 0;
    const parts = selects.slice().sort((a, b) => (a.branchId ?? 0) - (b.branchId ?? 0)).map((n) => {
      const sum = n.events.find((e) => msg0(e).startsWith("block_select key="));
      const mm = /candidates=(\d+).*?selected_blocks=(\d+)/.exec(sum ? msg0(sum) : "");
      if (mm) { totIn += +mm[1]; totSel += +mm[2]; }
      const info = m.branchInfo.get(n.branchId) || {};
      const key = info.key || "";
      return `<details class="stage"><summary><span class="ti">▶</span>
        <span class="stage-name">branch ${esc(n.branchId)}${key ? ` · ${esc(key)}` : ""}</span>
        <span class="stage-stat">${mm ? `${mm[1]} candidates → ${mm[2]} blocks` : "selected"}</span></summary>
        <div class="stage-body">${tournamentHtml(n, taskId)}</div></details>`;
    }).join("");
    return `<div class="flow-group">
      <div class="flow-head"><span class="flow-label">selection</span>
        <span class="stage-stat">${selects.length} branch(es) · ${totIn} candidates → ${totSel} blocks</span></div>
      ${parts}</div>`;
  }

  // ── block-select tournament ─────────────────────────────────────────────────
  function tcallHtml(c) {
    const d = c.data, blocks = d.blocks || [];
    const kept = blocks.filter((b) => b.kept).length;
    const cost = callCost(d.model, d.in_tok, d.out_tok, d.think_tok);
    const lines = blocks.map((b) => {
      const cls = b.kept ? "" : "dropped";
      return `<div class="tblock ${cls}">${b.kept ? "✓" : "✗"} ${esc(b.page)} ${esc(b.title || "")}</div>`;
    }).join("");
    return `<details class="tcall"><summary>${blocks.length} → ${kept}
      <span class="tcall-meta"> · ${fusd(cost)} · ${fsec(d.latency_s)}</span></summary>
      <div class="tcall-body">${lines}</div></details>`;
  }
  function tournamentLegacyHtml(node, taskId) {
    // Old trace without per-call `block_select_call` events: per-call cost/latency from the
    // call envelopes (inputs/keeps per call are unrecoverable — parallel calls interleave),
    // plus the round summaries.
    const env = node.events.filter((e) => e.kind === "call").map((e) => parseCallMsg(e.message)).filter(Boolean);
    const summaries = node.events.filter((e) => /^block_select(_round|_interval| )/.test(msg0(e)))
      .map((e) => `<div class="stage-stat mono">${esc(msg0(e))}</div>`).join("");
    const boxes = env.map((c, i) => `<details class="tcall"><summary>call ${i + 1}
      <span class="tcall-meta"> · ${fusd(callCost(c.model, c.inTok, c.outTok, c.thinkTok))} · ${fsec(c.latency)} · in ${c.inTok.toLocaleString()} tok</span></summary>
      <div class="tcall-body"><div class="stage-stat mono">${esc(c.model)} · out ${(c.outTok + c.thinkTok).toLocaleString()} tok</div></div></details>`).join("");
    const note = `<div class="hint">old trace — per-call inputs/keeps need a run with \`block_select_call\` events</div>`;
    return summaries + (boxes ? `<div class="tcalls" style="margin:6px 0">${boxes}</div>` : "") + note + rawEventsHtml(node, taskId);
  }
  function tournamentHtml(node, taskId) {
    const calls = node.events.filter((e) => msg0(e).startsWith("block_select_call") && Array.isArray(e.data?.blocks));
    if (!calls.length) return tournamentLegacyHtml(node, taskId);
    const ivals = new Map();
    for (const c of calls) { const k = c.data.interval ?? "(all)"; if (!ivals.has(k)) ivals.set(k, []); ivals.get(k).push(c); }
    const sums = new Map();
    for (const e of node.events) { if (msg0(e).startsWith("block_select_interval") && e.data) sums.set(e.data.interval ?? "(all)", e.data); }
    const parts = [];
    for (const [label, cs] of ivals) {
      const rounds = new Map();
      for (const c of cs) { const r = String(c.data.round); if (!rounds.has(r)) rounds.set(r, []); rounds.get(r).push(c); }
      const order = [...rounds.keys()].sort((a, b) => (a === "final" ? 1 : b === "final" ? -1 : (+a) - (+b)));
      const roundRows = order.map((r) => {
        const rcs = rounds.get(r).slice().sort((a, b) => (a.data.batch || 0) - (b.data.batch || 0));
        const tin = rcs.reduce((n, c) => n + (c.data.blocks || []).length, 0);
        const tkept = rcs.reduce((n, c) => n + (c.data.blocks || []).filter((b) => b.kept).length, 0);
        return `<div class="round-row"><div class="round-label">${r === "final" ? "final call" : "round " + esc(r)} — ${rcs.length} call(s) · in ${tin} → kept ${tkept}</div>
          <div class="tcalls">${rcs.map((c) => tcallHtml(c)).join("")}</div></div>`;
      }).join("");
      const lastCs = rounds.get(order[order.length - 1]) || [];
      const surv = new Set();
      for (const c of lastCs) for (const b of (c.data.blocks || [])) if (b.kept) surv.add(b.page);
      const sum = sums.get(label);
      const survChips = surv.size ? [...surv].map((p) => `<span class="chip chip-retr">${esc(p)}</span>`).join("") : `<span class="chip chip-none">0 kept</span>`;
      parts.push(`<div style="margin:6px 0 14px">
        <div class="flow-head"><span class="flow-label">interval</span><span class="flow-key mono">${esc(label)}</span>
          <span class="stage-stat">${sum ? `${sum.candidates} candidates → ${sum.selected} selected` : ""}</span>
          <span style="margin-left:auto">${survChips}</span></div>
        ${roundRows}</div>`);
    }
    return parts.join("") + rawEventsHtml(node, taskId);
  }
})();
