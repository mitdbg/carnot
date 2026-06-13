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
    const branchNodes = seg.filter((n) => n.branchId != null);

    for (const n of replanner) items.push(nodeHtml(n, taskId));
    if (setup.length) items.push(`<div class="flow-group"><div class="flow-head"><span class="flow-label">planning${setup.length > 1 ? " · parallel" : ""}</span></div>${setup.map((n) => nodeHtml(n, taskId)).join("")}</div>`);

    const byBranch = new Map();
    for (const n of branchNodes) { if (!byBranch.has(n.branchId)) byBranch.set(n.branchId, []); byBranch.get(n.branchId).push(n); }
    for (const [bid, bnodes] of byBranch) items.push(branchGroupHtml(bid, bnodes, m, sel, taskId));

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
    const body = node.events.length ? node.events.map((e) => eventHtml(e, taskId)).join("") : `<div class="trace-empty">no internal events</div>`;
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
})();
