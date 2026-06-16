"use strict";
// Read-only monitoring console. Two SSE streams from the same origin:
//   GET /api/stream            -> compact {round, tasks[]} status, re-sent on any change
//   GET /api/stream/<task_id>  -> the selected task's trace events (backfill, then live)
// The status stream renders a fixed 5-column grid of task cards (sorted by UID, never reordered
// by status). Clicking a card opens a full-screen DETAIL OVERLAY with that task's prompt/answer/
// actions + live trace; clicking a review tag on a card opens the REVIEW OVERLAY at that review.
// Trace events only touch the detail overlay's #trace container (which preserves expanded
// <details> across renders), so streaming never collapses what you've opened.

const app = document.getElementById("app");
let round = {};
let tasks = [];                       // latest status snapshot's task summaries
let selectedRoundTab = null;
let lastRoundNum = null;

// Detail overlay: the task whose detail/trace modal is open (null = closed). Its live trace is
// accumulated from that task's event stream.
let detailTaskId = null;
let taskSource = null;                // EventSource for the open task's events
let attemptOrder = [];                // attempt_ids in first-seen order
const attemptBuffers = new Map();     // attempt_id -> events[]
let traceFrame = null;                // pending rAF handle for coalesced trace renders

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));
const js = (s) => String(s ?? "").replace(/\\/g, "\\\\").replace(/'/g, "\\'");

// True when another operator holds this task's review lock — their client is annotating it, so
// its review/submit actions are greyed out for everyone else. This browser's lock identity is
// defined in review_overlay.js (loads first) and exposed as window.CLIENT_ID.
const lockedByOther = (task) => !!task.locked_by && task.locked_by !== window.CLIENT_ID;

// ── status badge ───────────────────────────────────────────────────────────────
// One status tag per task. Review state is shown by the separate review-kind tags, NOT folded
// into the status, so the badge always reads the real lifecycle state.
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

// ── review kinds ─────────────────────────────────────────────────────────────────
const reviewCount = (task) => (task.reviews || []).length;
// True while any of a task's reviews has an in-flight LLM refine of its extracted candidates
// (the natural-language feedback path), surfaced as an "Updating" indicator.
const isRefining = (task) => (task.reviews || []).some((r) => r.refining);
// replan_approval (a mandatory "approve the proposed replan" gate after a NeedsMore) blocks the
// whole run, so it's the most urgent; then lookup, visual QA (figure), extract validation.
const REVIEW_KIND_RANK = { replan_approval: 0, lookup: 1, figure: 2, verify_extract: 3 };
const REVIEW_KIND_LABEL = { replan_approval: "Approve Replan", lookup: "Lookup", figure: "Visual QA", verify_extract: "Extract" };
// Fuller labels for the detail overlay's per-kind review buttons (the cards use the short ones).
const REVIEW_BUTTON_LABEL = { replan_approval: "Approve Replan", lookup: "External Lookup", figure: "Visual QA", verify_extract: "Extract" };

// The review entries a task surfaces, ordered by urgency. Extract + lookup are ONE family — never
// two tags: the verify_extract "pool" review already contains the lookup-derived values (their
// `external` candidates), so we drop the separate lookup tag and flip the single label to "Lookup"
// when the reviewable values include an external lookup. Each entry: `openKind` = the review the
// tag opens, `cls` = its colour class, `label` (when set) overrides the per-call label map.
function reviewTagEntries(task) {
  const reviews = task.reviews || [];
  const byKind = {};
  for (const r of reviews) byKind[r.kind] = (byKind[r.kind] || 0) + 1;
  const entries = [];
  if (byKind.verify_extract || byKind.lookup) {
    const openKind = byKind.verify_extract ? "verify_extract" : "lookup";
    // "Lookup" when a standalone lookup review is all there is, or the pool review carries any
    // external (no-corpus-provenance) value.
    const isLookup = !byKind.verify_extract
      || reviews.some((r) => r.kind === "verify_extract"
        && (((r.guidance && r.guidance.candidates) || []).some((c) => c.external)));
    entries.push({ openKind, cls: isLookup ? "lookup" : "verify_extract",
      count: byKind[openKind], label: isLookup ? "Lookup" : "Extract" });
  }
  for (const k of Object.keys(byKind)) {
    if (k === "verify_extract" || k === "lookup") continue;
    entries.push({ openKind: k, cls: k, count: byKind[k] });
  }
  return entries.sort((a, b) => (REVIEW_KIND_RANK[a.openKind] ?? 9) - (REVIEW_KIND_RANK[b.openKind] ?? 9));
}

// One clickable tag per review entry on a card (e.g. "Extract", "Visual QA (2)"). Clicking a tag
// opens the review overlay at that entry's kind (stopPropagation so the card's own click — which
// opens the detail overlay — doesn't also fire). Greyed when locked.
function cardKindTags(task) {
  const locked = lockedByOther(task);
  return reviewTagEntries(task)
    .map((e) => {
      const label = `${esc(e.label || REVIEW_KIND_LABEL[e.openKind] || e.openKind)}${e.count > 1 ? ` (${e.count})` : ""}`;
      return locked
        ? `<span class="tag kind-${esc(e.cls)} locked-out" title="Locked by another reviewer">${label}</span>`
        : `<span class="tag kind-${esc(e.cls)}" onclick="openReview('${js(task.task_id)}', '${js(e.openKind)}', event)">${label}</span>`;
    })
    .join("");
}

// Whether a submission has been scored, and if so whether it was correct (null until known).
function feedbackFor(task) {
  if (task.correct == null) return null;
  return { correct: !!task.correct, points: task.points };
}

// ── round timer ──────────────────────────────────────────────────────────────────
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
  // Cards fill the page directly (no panel chrome / "Tasks" header).
  app.innerHTML = `<div id="roundTabs" class="round-tabs"></div>
    <div id="taskGrid" class="task-grid"></div>`;
}

// ── task card ──────────────────────────────────────────────────────────────────
// Fixed-size card: UID + a single status tag + open review-kind tags + indicators. Sized (in CSS,
// with overflow:hidden) for the worst case so a tag can never bleed past the card. Clicking the
// card body opens the detail overlay; review-kind tags open the review overlay (see cardKindTags).
function taskCardHTML(task) {
  const indicators = [
    task.revising ? `<span class="tag ind ind-revising">Revising</span>` : "",
    isRefining(task) ? `<span class="tag ind ind-refining">Updating</span>` : "",
    lockedByOther(task) ? `<span class="tag ind ind-lock" title="Locked by another reviewer">🔒 In review</span>` : "",
  ].join("");
  const corpusTags = window.CorpusUI ? window.CorpusUI.summaryTags(task.corpus_summary) : "";
  return `<div class="task-card ${task.task_id === detailTaskId ? "selected" : ""}" onclick="openDetail('${js(task.task_id)}')">
    <div class="card-uid">${esc(task.question_id)}</div>
    <div class="card-tags">
      <span class="tag status ${badgeClass(task)}">${esc(badgeLabel(task))}</span>
      ${cardKindTags(task)}
      ${corpusTags}
      ${indicators}
    </div>
  </div>`;
}

// ── status rendering ───────────────────────────────────────────────────────────
function renderStatus() {
  document.getElementById("connection").textContent = round.connection_status || "-";
  document.getElementById("roundNum").textContent = round.round_num ?? "-";
  document.getElementById("roundStatus").textContent = round.status || "-";
  document.getElementById("resubmitsLeft").textContent = round.resubmits_left ?? "-";
  updateRoundTimer();

  const rounds = [...new Set(tasks.map((t) => t.round_num))].sort((a, b) => a - b);
  if (round.round_num != null && round.round_num !== lastRoundNum) {
    lastRoundNum = round.round_num;
    if (rounds.includes(round.round_num)) selectedRoundTab = round.round_num;
  }
  if (selectedRoundTab == null || !rounds.includes(selectedRoundTab)) {
    selectedRoundTab = rounds.length ? rounds[rounds.length - 1] : null;
  }
  // Sorted by UID and NOT by status, so a card never moves under you as its status changes.
  const visible = tasks
    .filter((t) => t.round_num === selectedRoundTab)
    .sort((a, b) => String(a.question_id).localeCompare(String(b.question_id)));

  document.getElementById("roundTabs").innerHTML = rounds.length > 1
    ? rounds.map((r) => `<button class="round-tab ${r === selectedRoundTab ? "active" : ""}" onclick="selectRoundTab(${Number(r)})">Round ${esc(r)}</button>`).join("")
    : "";
  document.getElementById("taskGrid").innerHTML = visible.length
    ? visible.map(taskCardHTML).join("")
    : `<div class="empty">No tasks yet.</div>`;

  // Keep an open detail overlay's header/actions fresh (its trace is driven by its own stream).
  if (detailTaskId) refreshDetail();
}

// ── detail overlay ──────────────────────────────────────────────────────────────
function detailRoot() { return document.getElementById("detailOverlay"); }

function openDetail(taskId) {
  if (taskId === detailTaskId) return;
  // reset trace state for the newly-opened task
  if (taskSource) { taskSource.close(); taskSource = null; }
  attemptOrder = [];
  attemptBuffers.clear();
  detailTaskId = taskId;

  const task = tasks.find((t) => t.task_id === taskId);
  const el = detailRoot();
  el.innerHTML = `
    <div class="detail-backdrop" onclick="closeDetailBackdrop(event)"></div>
    <div class="detail-panel">
      <div class="detail-head">
        <div class="detail-head-title">
          <span class="detail-task">R${esc(task?.round_num ?? "?")} / ${esc(task?.question_id ?? taskId)}</span>
          <span id="detailBadge" class="status ${task ? badgeClass(task) : "queued"}">${esc(task ? badgeLabel(task) : "-")}</span>
        </div>
        <button class="detail-close" title="Close" onclick="closeDetail()">✕</button>
      </div>
      <div class="detail-scroll">
        <div class="prompt-label">PROMPT</div>
        <div class="prompt-text">${esc(task?.prompt || "")}</div>
        <div id="dCorpus" class="detail-corpus">${window.CorpusUI ? window.CorpusUI.summaryTags(task?.corpus_summary) : ""}</div>
        <div class="answer-row">
          <div><span class="k">Status</span><strong id="dStatus">${esc(task ? badgeLabel(task) : "-")}</strong></div>
          <div><span class="k">Answer</span><strong id="dAnswer">${esc(task?.answer || "—")}</strong></div>
        </div>
        <div id="dActions" class="detail-actions"></div>
        <div id="trace" class="trace-host"></div>
      </div>
    </div>`;
  el.hidden = false;
  renderDetailActions(task);
  renderTrace();
  // refresh the grid's selected highlight without waiting for the next status tick
  document.querySelectorAll("#taskGrid .task-card").forEach((c) => c.classList.remove("selected"));

  taskSource = new EventSource(`/api/stream/${encodeURIComponent(taskId)}`);
  taskSource.onmessage = (e) => onTraceEvent(taskId, JSON.parse(e.data));
}
window.openDetail = openDetail;

function closeDetail() {
  if (taskSource) { taskSource.close(); taskSource = null; }
  detailTaskId = null;
  attemptOrder = [];
  attemptBuffers.clear();
  const el = detailRoot();
  if (el) { el.hidden = true; el.innerHTML = ""; }
  renderStatus();  // drop the card's selected highlight
}
window.closeDetail = closeDetail;

function closeDetailBackdrop(event) {
  if (event && event.target && event.target.classList.contains("detail-backdrop")) closeDetail();
}
window.closeDetailBackdrop = closeDetailBackdrop;

// Patch the open overlay's header + actions in place from the latest snapshot (its trace is live
// off its own stream). No-op if it's closed.
function refreshDetail() {
  const task = tasks.find((t) => t.task_id === detailTaskId);
  const badge = document.getElementById("detailBadge");
  if (badge) {
    badge.textContent = task ? badgeLabel(task) : "none";
    badge.className = `status ${task ? badgeClass(task) : "queued"}`;
  }
  const st = document.getElementById("dStatus");
  const ans = document.getElementById("dAnswer");
  const corpus = document.getElementById("dCorpus");
  if (st) st.textContent = task ? badgeLabel(task) : "-";
  if (ans) ans.textContent = task?.answer || "—";
  if (corpus) corpus.innerHTML = window.CorpusUI ? window.CorpusUI.summaryTags(task?.corpus_summary) : "";
  renderDetailActions(task);
}

// The Review (open overlay) + Submit buttons and the per-task feedback live in the detail overlay;
// changes arrive over the status SSE stream and refresh through refreshDetail.
function renderDetailActions(task) {
  const host = document.getElementById("dActions");
  if (!host) return;
  if (!task) { host.innerHTML = ""; return; }
  const locked = lockedByOther(task);
  let html = "";
  if (task.revising) {
    html += `<span class="detail-feedback revising">Revising answer after review…</span>`;
  }
  if (isRefining(task)) {
    html += `<span class="detail-feedback refining">Updating extraction from your feedback…</span>`;
  }
  // One button per review entry, ordered by urgency, so the reviewer picks which to do first.
  // Extract + lookup are one family (see reviewTagEntries) — a single button whose label reads
  // "Lookup" when the values include one. Each opens the overlay straight to that entry's kind.
  html += reviewTagEntries(task)
    .map((e) => {
      const label = `${e.label || REVIEW_BUTTON_LABEL[e.openKind] || e.openKind}${e.count > 1 ? ` (${e.count})` : ""}`;
      return locked
        ? `<button class="primary" disabled title="Locked by another reviewer">${esc(label)}</button>`
        : `<button class="primary" onclick="openReview('${js(task.task_id)}', '${js(e.openKind)}', event)">${esc(label)}</button>`;
    })
    .join("");
  if (task.status === "READY") {
    html += locked
      ? `<button class="primary" disabled title="Locked by another reviewer">Submit Answer</button>`
      : `<button class="primary" onclick="submitTask('${js(task.task_id)}', event)">Submit Answer</button>`;
  } else if (task.status === "SUBMITTING") {
    html += `<span class="detail-feedback pending">Submitting…</span>`;
  }
  if (locked) {
    html += `<span class="detail-feedback">🔒 Locked by another reviewer</span>`;
  }
  const fb = feedbackFor(task);
  if (fb) {
    const pts = fb.points != null ? ` (${fb.correct ? "+" : ""}${fb.points} pts)` : "";
    html += fb.correct
      ? `<span class="detail-feedback correct">Correct${pts}</span>`
      : `<span class="detail-feedback incorrect">Incorrect${pts}</span>`;
  }
  host.innerHTML = html;
}

// ── manual submission ──────────────────────────────────────────────────────────
async function submitTask(taskId, event) {
  if (event) { event.stopPropagation(); event.preventDefault(); }
  try {
    await apiJson(`/api/submit/${encodeURIComponent(taskId)}`, { client_id: window.CLIENT_ID });
  } catch (err) {
    flashSubmitError(taskId, err.message || String(err));
  }
  // On success the status SSE stream pushes SUBMITTING -> SUBMITTED/SCORED and the UI refreshes.
}
window.submitTask = submitTask;

function flashSubmitError(taskId, message) {
  const host = taskId === detailTaskId ? document.getElementById("dActions") : null;
  if (host) {
    const note = document.createElement("span");
    note.className = "detail-feedback error";
    note.textContent = message;
    host.appendChild(note);
    setTimeout(() => note.remove(), 6000);
  } else {
    alert(`Submit failed for ${taskId}: ${message}`);
  }
}

// ── review overlay open ──────────────────────────────────────────────────────────
// kind (optional) jumps the overlay straight to that review kind (a card's review tag); null opens
// the task's first review (the detail overlay's Review button).
function openReview(taskId, kind, event) {
  if (event) { event.stopPropagation(); event.preventDefault(); }
  ReviewOverlay.open(taskId, kind || undefined);
}
window.openReview = openReview;

function selectRoundTab(roundNum) {
  selectedRoundTab = roundNum;
  renderStatus();
}
window.selectRoundTab = selectRoundTab;

// Esc closes the detail overlay (the review overlay manages its own dismissal).
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && detailTaskId && !ReviewOverlay.isOpen()) closeDetail();
});

// ── trace streaming (drives the detail overlay's #trace) ─────────────────────────
function onTraceEvent(taskId, msg) {
  if (taskId !== detailTaskId) return; // a stale stream that hasn't closed yet
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
  if (!host || !detailTaskId) return;
  TraceFlow.captureOpen(detailTaskId, host); // snapshot open <details> before rebuild
  const attempts = attemptOrder.map((id) => ({ attempt_id: id, events: attemptBuffers.get(id) || [] }));
  host.innerHTML = TraceFlow.html(detailTaskId, attempts);
}
TraceFlow.onChange = renderTrace; // setRev / toggleEvent re-render through here

// ── status stream ────────────────────────────────────────────────────────────
function connectStatus() {
  const source = new EventSource("/api/stream");
  source.onmessage = (e) => {
    const data = JSON.parse(e.data);
    round = data.round || {};
    tasks = data.tasks || [];
    // Keep the review overlay's task data fresh BEFORE rendering so a review resolved/cancelled
    // elsewhere drops out live.
    ReviewOverlay.syncTasks(tasks);
    renderStatus();
  };
  source.onerror = () => {
    document.getElementById("connection").textContent = "reconnecting…";
    // EventSource auto-reconnects; nothing else to do.
  };
}

buildSkeleton();
connectStatus();
