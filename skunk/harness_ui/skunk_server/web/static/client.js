"use strict";
// Read-only monitoring console. Two SSE streams from the same origin:
//   GET /api/stream            -> compact {round, tasks[]} status, re-sent on any change
//   GET /api/stream/<task_id>  -> the selected task's trace events (backfill, then live)
// Status updates patch the header/metrics/task-list in place; trace events append to the
// selected task's per-attempt buffers and re-render ONLY the #trace container (which
// preserves expanded <details> across renders). The two never touch each other's DOM, so
// streaming events never collapse what you've opened.

const app = document.getElementById("app");
let round = {};
let tasks = [];                       // latest status snapshot's task summaries
let selectedTaskId = null;
let selectedRoundTab = null;
let lastRoundNum = null;

// Selected task's live trace, accumulated from its event stream.
let taskSource = null;                // EventSource for the selected task's events
let attemptOrder = [];                // attempt_ids in first-seen order
const attemptBuffers = new Map();     // attempt_id -> events[]
let traceFrame = null;                // pending rAF handle for coalesced trace renders

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));
const js = (s) => String(s ?? "").replace(/\\/g, "\\\\").replace(/'/g, "\\'");

// ── status badges ────────────────────────────────────────────────────────────
function badgeClass(task) {
  switch (task.status) {
    case "SUBMITTED":
    case "SCORED":
      return Number(task.points) === 0 ? "submitted-zero" : "submitted";
    case "FAILED": return "failed";
    case "CANCELLED": return "cancelled";
    case "READY": return "ready";
    case "PROCESSING": return "running";
    case "SUBMITTING": return "submitting";
    default: return "queued";
  }
}
function badgeLabel(task) {
  if ((task.status === "SUBMITTED" || task.status === "SCORED") && task.points != null) {
    return `${task.status} (${task.points} pts)`;
  }
  return task.status;
}
function counts(all) {
  const by = (fn) => all.filter(fn).length;
  return {
    total: all.length,
    running: by((t) => t.status === "PROCESSING"),
    ready: by((t) => t.status === "READY"),
    submitted: by((t) => ["SUBMITTED", "SCORED"].includes(t.status)),
    failed: by((t) => t.status === "FAILED"),
    cancelled: by((t) => t.status === "CANCELLED"),
  };
}

// ── round timer ──────────────────────────────────────────────────────────────
function updateRoundTimer() {
  const el = document.getElementById("roundTimer");
  if (!el) return;
  if (!round.ends_at || round.status !== "ACTIVE") { el.textContent = "--:--"; return; }
  const remaining = Math.max(0, Math.floor((new Date(round.ends_at).getTime() - Date.now()) / 1000));
  const m = String(Math.floor(remaining / 60)).padStart(2, "0");
  const s = String(remaining % 60).padStart(2, "0");
  el.textContent = `${m}:${s}`;
}
setInterval(updateRoundTimer, 1000);

// ── skeleton (built once) ──────────────────────────────────────────────────────
function buildSkeleton() {
  app.innerHTML = `<section class="workspace">
    <aside class="list-panel">
      <div class="panel-head"><div class="panel-title">Tasks</div><span class="pill"><strong id="taskCount">0</strong></span></div>
      <div id="roundTabs" class="round-tabs"></div>
      <div id="taskList" class="task-list"></div>
    </aside>
    <section class="detail-panel">
      <div class="panel-head"><div class="panel-title">Task Details</div><span id="detailBadge" class="status queued">none</span></div>
      <div id="detail" class="detail-body"><div class="detail-empty">Select a task to follow its trace.</div></div>
    </section>
  </section>`;
}

// ── status rendering (never touches #trace) ─────────────────────────────────────
function renderStatus() {
  document.getElementById("connection").textContent = round.connection_status || "-";
  document.getElementById("roundNum").textContent = round.round_num ?? "-";
  document.getElementById("roundStatus").textContent = round.status || "-";
  document.getElementById("resubmitsLeft").textContent = round.resubmits_left ?? "-";
  updateRoundTimer();
  const c = counts(tasks);
  document.getElementById("mTotal").textContent = c.total;
  document.getElementById("mRunning").textContent = c.running;
  document.getElementById("mReady").textContent = c.ready;
  document.getElementById("mSubmitted").textContent = c.submitted;
  document.getElementById("mFailed").textContent = c.failed;
  document.getElementById("mCancelled").textContent = c.cancelled;

  const rounds = [...new Set(tasks.map((t) => t.round_num))].sort((a, b) => a - b);
  if (round.round_num != null && round.round_num !== lastRoundNum) {
    lastRoundNum = round.round_num;
    if (rounds.includes(round.round_num)) selectedRoundTab = round.round_num;
  }
  if (selectedRoundTab == null || !rounds.includes(selectedRoundTab)) {
    selectedRoundTab = rounds.length ? rounds[rounds.length - 1] : null;
  }
  const visible = tasks.filter((t) => t.round_num === selectedRoundTab);

  document.getElementById("roundTabs").innerHTML = rounds.length > 1
    ? rounds.map((r) => `<button class="round-tab ${r === selectedRoundTab ? "active" : ""}" onclick="selectRoundTab(${Number(r)})">Round ${esc(r)}</button>`).join("")
    : "";
  document.getElementById("taskCount").textContent = String(visible.length);
  document.getElementById("taskList").innerHTML = visible.length
    ? visible.map((task) => `<button class="task-row ${task.task_id === selectedTaskId ? "selected" : ""}" onclick="selectTask('${js(task.task_id)}')">
        <div class="row-main">
          <div class="row-title">R${esc(task.round_num)} / ${esc(task.question_id)}</div>
          <div class="row-preview">${esc(task.prompt || "")}</div>
        </div>
        <span class="status ${badgeClass(task)}">${esc(badgeLabel(task))}</span>
      </button>`).join("")
    : `<div class="empty">No tasks yet.</div>`;

  // Auto-follow the first task if nothing is selected; else patch the open detail header.
  if (!selectedTaskId && visible.length) {
    selectTask(visible[0].task_id);
  } else if (selectedTaskId && !visible.some((t) => t.task_id === selectedTaskId)) {
    // Selected task left the visible round; keep the stream but reflect it's not in view.
    updateDetailHeader();
  } else {
    updateDetailHeader();
  }
}

function updateDetailHeader() {
  const task = tasks.find((t) => t.task_id === selectedTaskId);
  const badge = document.getElementById("detailBadge");
  if (badge) {
    badge.textContent = task ? badgeLabel(task) : "none";
    badge.className = `status ${task ? badgeClass(task) : "queued"}`;
  }
  if (task) {
    const st = document.getElementById("dStatus");
    const ans = document.getElementById("dAnswer");
    if (st) st.textContent = badgeLabel(task);
    if (ans) ans.textContent = task.answer || "—";
  }
}

// ── task selection + trace streaming ────────────────────────────────────────────
function selectTask(taskId) {
  if (taskId === selectedTaskId) return;
  selectedTaskId = taskId;
  // reset trace state for the newly-selected task
  if (taskSource) { taskSource.close(); taskSource = null; }
  attemptOrder = [];
  attemptBuffers.clear();

  const task = tasks.find((t) => t.task_id === taskId);
  const detail = document.getElementById("detail");
  detail.innerHTML = `
    <div class="task-head">R${esc(task?.round_num ?? "?")} / ${esc(task?.question_id ?? taskId)}</div>
    <div class="prompt-label">PROMPT</div>
    <div class="prompt-text">${esc(task?.prompt || "")}</div>
    <div class="answer-row">
      <div><span class="k">Status</span><strong id="dStatus">${esc(task ? badgeLabel(task) : "-")}</strong></div>
      <div><span class="k">Answer</span><strong id="dAnswer">${esc(task?.answer || "—")}</strong></div>
    </div>
    <div id="trace" class="trace-host"></div>`;

  // refresh list highlight without a full status re-render
  document.querySelectorAll("#taskList .task-row").forEach((row) => row.classList.remove("selected"));
  updateDetailHeader();
  renderTrace();

  taskSource = new EventSource(`/api/stream/${encodeURIComponent(taskId)}`);
  taskSource.onmessage = (e) => onTraceEvent(taskId, JSON.parse(e.data));
}
window.selectTask = selectTask;

function selectRoundTab(roundNum) {
  selectedRoundTab = roundNum;
  renderStatus();
}
window.selectRoundTab = selectRoundTab;

function onTraceEvent(taskId, msg) {
  if (taskId !== selectedTaskId) return; // a stale stream that hasn't closed yet
  const { attempt_id, event } = msg;
  let buf = attemptBuffers.get(attempt_id);
  if (!buf) { buf = []; attemptBuffers.set(attempt_id, buf); attemptOrder.push(attempt_id); }
  buf.push(event);
  if (traceFrame == null) traceFrame = requestAnimationFrame(flushTrace);
}
function flushTrace() {
  traceFrame = null;
  renderTrace();
}
function renderTrace() {
  const host = document.getElementById("trace");
  if (!host || !selectedTaskId) return;
  TraceFlow.captureOpen(selectedTaskId, host); // snapshot open <details> before rebuild
  const attempts = attemptOrder.map((id) => ({ attempt_id: id, events: attemptBuffers.get(id) || [] }));
  host.innerHTML = TraceFlow.html(selectedTaskId, attempts);
}
TraceFlow.onChange = renderTrace; // setRev / toggleEvent re-render through here

// ── status stream ────────────────────────────────────────────────────────────
function connectStatus() {
  const source = new EventSource("/api/stream");
  source.onmessage = (e) => {
    const data = JSON.parse(e.data);
    round = data.round || {};
    tasks = data.tasks || [];
    renderStatus();
  };
  source.onerror = () => {
    document.getElementById("connection").textContent = "reconnecting…";
    // EventSource auto-reconnects; nothing else to do.
  };
}

buildSkeleton();
connectStatus();
