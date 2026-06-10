const SERVER = document.body.dataset.serverUrl || document.documentElement.dataset.serverUrl || "";
let state = { worker: null, round: {}, tasks: [], workers: [] };
let workerId = null;
let selectedTaskId = null;
const app = document.getElementById("app");
const lastEvent = document.getElementById("lastEvent");
const roundTimerMetric = document.getElementById("roundTimerMetric");
const roundTimer = document.getElementById("roundTimer");

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function js(value) {
  return String(value ?? "").replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

function workerLabel(id) {
  const worker = state.workers.find(item => item.worker_id === id);
  if (!worker) return String(id).slice(0, 4);
  return `${worker.display_name || "Human Worker"} (${worker.worker_id.slice(0, 4)})`;
}

function workerMnemonic(workerId) {
  const worker = state.workers.find(item => item.worker_id === workerId);
  if (!worker) return String(workerId).slice(0, 4);
  const name = worker.display_name?.trim() || "Human Worker";
  return `${name} (${worker.worker_id.slice(0, 4)})`;
}

function websocketUrl() {
  const url = new URL(SERVER);
  return `${url.protocol === "https:" ? "wss:" : "ws:"}//${url.host}/ws`;
}

function connect() {
  const socket = new WebSocket(websocketUrl());
  socket.onmessage = event => {
    state = JSON.parse(event.data);
    workerId = state.worker?.worker_id || workerId;
    render();
  };
  socket.onclose = () => setTimeout(connect, 1000);
  socket.onerror = () => socket.close();
}

async function hydrate() {
  try {
    const response = await fetch(SERVER + "/api/state");
    if (!response.ok) return;
    const snapshot = await response.json();
    // A WebSocket snapshot includes this browser's worker identity. Do not
    // overwrite it if the socket won the race with this initial HTTP request.
    if (!state.worker) {
      state = snapshot;
      render();
    }
  } catch {
    // The WebSocket reconnect loop remains the authoritative recovery path.
  }
}

async function command(path, body, method = "POST") {
  const response = await fetch(SERVER + path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`${response.status}: ${text}`);
  return text ? JSON.parse(text) : null;
}

async function run(action) {
  try {
    await action;
  } catch (error) {
    const target = document.getElementById("error");
    if (target) target.textContent = error.message;
  }
}

function activeAssignment(task) {
  return task.assignments?.find(item => item.worker_id === workerId && item.status === "ACTIVE");
}

function counts(tasks) {
  const by = fn => tasks.filter(fn).length;
  return {
    total: tasks.length,
    running: by(task => task.status === "PROCESSING"),
    ready: by(task => task.status === "READY"),
    submitted: by(task => ["SUBMITTED", "SCORED"].includes(task.status)),
    failed: by(task => task.status === "FAILED"),
    completed: by(task => ["SUBMITTED", "SCORED", "FAILED", "CANCELLED"].includes(task.status)),
  };
}

function formatPoints(points) {
  const whole = Number(points);
  if (!Number.isFinite(whole)) return null;
  return whole === 1 ? "1pt" : `${whole}pts`;
}

function taskBadge(task) {
  const submission = task.submissions?.[task.submissions.length - 1];
  if (task.status === "SUBMITTED" || task.status === "SCORED") {
    const pointsLabel = formatPoints(submission?.points_awarded);
    return pointsLabel ? `SUBMITTED (${pointsLabel})` : "SUBMITTED";
  }
  if (task.status === "SUBMITTING" && submission?.status === "PENDING") {
    return "SUBMITTING";
  }
  return task.status;
}

function taskBadgeClass(task) {
  if (task.status === "SUBMITTED" || task.status === "SCORED") return "submitted";
  if (task.status === "FAILED") return "failed";
  if (task.status === "READY") return "ready";
  if (task.status === "PROCESSING") return "running";
  if (task.status === "SUBMITTING") return "submitting";
  return "queued";
}

function parseReasoningPayload(reasoning) {
  if (typeof reasoning !== "string" || !reasoning.trim().startsWith("{")) return null;
  try {
    const parsed = JSON.parse(reasoning);
    if (parsed && Array.isArray(parsed.branches)) return parsed;
  } catch {
    return null;
  }
  return null;
}

function formatReasoningValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function reasoningVisual(reasoning, answer) {
  const payload = parseReasoningPayload(reasoning);
  if (!payload) return "";
  const branches = payload.branches || [];
  const code = payload.python_code || "";
  const attempts = payload.python_attempts || [];
  return `<section class="reasoning-shell">
    <div class="label">Reasoning Flow</div>
    <div class="reasoning-flow">
      <div class="flow-branches">
      ${branches.map(branch => `
        <article class="flow-lane ${esc(branch.status || "ok")}">
          <div class="flow-node search-node">
            <span class="flow-kicker">${esc((branch.kind || "branch").replace("_", " "))} ${esc(branch.branch_id)}</span>
            <strong>${esc(formatReasoningValue(branch.searched) || "No search detail")}</strong>
          </div>
          <div class="flow-arrow" aria-hidden="true"></div>
          <div class="flow-node value-node">
            <span class="flow-kicker">${esc(branch.output_step || "extracted values")}</span>
            <strong>${esc(formatReasoningValue(branch.output) || "No value captured")}</strong>
          </div>
        </article>
      `).join("")}
      </div>
      <div class="flow-merge" aria-hidden="true"></div>
      <div class="flow-node compute-node">
        <span class="flow-kicker">Python compute${payload.python_attempt ? ` · attempt ${esc(payload.python_attempt)}` : ""}</span>
        <pre class="code-block">${esc(code || "No python code captured.")}</pre>
        ${attempts.length > 1 ? `<span class="flow-note">${esc(attempts.length)} compute attempts captured</span>` : ""}
      </div>
      <div class="flow-arrow vertical" aria-hidden="true"></div>
      <div class="flow-node answer-node">
        <span class="flow-kicker">Final answer</span>
        <strong>${esc(answer || "-")}</strong>
      </div>
    </div>
  </section>`;
}

function taskElapsedSeconds(task) {
  const attempts = task.attempts || [];
  if (!attempts.length) return null;
  return attempts.reduce((total, attempt) => {
    const started = Date.parse(attempt.started_at);
    const completed = Date.parse(attempt.completed_at || new Date().toISOString());
    return total + (Number.isFinite(started) && Number.isFinite(completed)
      ? Math.max(0, completed - started) / 1000
      : 0);
  }, 0);
}

function formatClock(timestamp) {
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) return "-";
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "--:--";
  const safe = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const secs = safe % 60;
  return hours > 0
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function updateRoundTimer() {
  const deadline = state?.round?.ends_at ? Date.parse(state.round.ends_at) : NaN;
  roundTimerMetric.classList.remove("warning", "expired");
  if (!Number.isFinite(deadline)) {
    roundTimer.textContent = "--:--";
    return;
  }
  const remainingSeconds = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
  roundTimer.textContent = formatDuration(remainingSeconds);
  if (remainingSeconds === 0) roundTimerMetric.classList.add("expired");
  else if (remainingSeconds <= 60) roundTimerMetric.classList.add("warning");
}

function selectTask(taskId) {
  selectedTaskId = taskId;
  render();
}

function sourceDocHtml(doc) {
  const label = String(doc ?? "");
  const bulletinMatch = label.match(/^Treasury Bulletin ([0-9]{4}-(?:0[1-9]|1[0-2])) PDF page (\d+)$/);
  if (bulletinMatch) {
    const month = bulletinMatch[1];
    const page = Number(bulletinMatch[2]);
    return `<a class="doc" href="/api/source/${encodeURIComponent(month)}#page=${page}" target="_blank" rel="noopener noreferrer" title="Open ${esc(label)}">${esc(label)}</a>`;
  }
  if (/^https?:\/\//i.test(label) || /#p\d+$/i.test(label)) {
    return `<a class="doc" href="${esc(label)}" target="_blank" rel="noopener noreferrer" title="Open ${esc(label)}">${esc(label)}</a>`;
  }
  return `<span class="doc" title="${esc(label)}">${esc(label)}</span>`;
}

function renderDetail(task) {
  const candidates = task.answer_candidates || [];
  const failures = task.failures || [];
  const submissions = task.submissions || [];
  const candidate = candidates[candidates.length - 1];
  const failure = failures[failures.length - 1];
  const submission = submissions[submissions.length - 1];
  const assignment = activeAssignment(task);
  const assignmentSummary = task.assignments?.length
    ? `<div class="label">Assignments</div><pre>${esc(task.assignments.map(item => `${workerLabel(item.worker_id)}: ${item.status}`).join("\n"))}</pre>`
    : "";
  const cupFeedback = task.cup_feedback?.length
    ? `<div class="label">Cup Feedback</div><pre>${esc(task.cup_feedback.join("\n"))}</pre>`
    : "";
  const submissionLine = submission
    ? `<div class="label">Submission</div><pre>${esc(submission.status)}${submission.correct === null || submission.correct === undefined ? "" : ` | correct=${submission.correct} | points=${submission.points_awarded ?? "-"}`}</pre>`
    : "";
  const answerSection = candidate
    ? `${candidate.source_docs?.length ? `<div class="label">Source Docs</div><div class="docs">${candidate.source_docs.map(sourceDocHtml).join("")}</div>` : ""}`
    : "";
  const reasoningPayload = candidate ? parseReasoningPayload(candidate.reasoning) : null;
  const reasoningSection = candidate
    ? reasoningVisual(candidate.reasoning, candidate.answer_text)
      || `<div class="label">Reasoning</div><pre>${esc(candidate.reasoning)}</pre>`
    : "";
  const detailActions = [];
  if ((task.status === "READY" || task.status === "FAILED") && workerId) {
    detailActions.push(`<button class="primary" onclick="claimTask('${js(task.task_id)}')">Claim</button>`);
  }
  if (task.status === "READY") {
    const assignmentArgs = assignment
      ? `, '${js(assignment.assignment_id)}', ${assignment.task_version}`
      : "";
    detailActions.push(`<button class="primary" onclick="submitReady('${js(task.task_id)}'${assignmentArgs})">Submit</button>`);
  }
  if (assignment) {
    detailActions.push(`<input id="feedback_${js(assignment.assignment_id)}" placeholder="Retry feedback">
      <button onclick="retryTask('${js(assignment.assignment_id)}',${assignment.task_version})">Retry</button>`);
    if (task.status === "FAILED") {
      detailActions.push(`<input id="answer_${js(assignment.assignment_id)}" placeholder="Direct final answer">
        <textarea id="reasoning_${js(assignment.assignment_id)}" placeholder="Reasoning"></textarea>
        <textarea id="sources_${js(assignment.assignment_id)}" placeholder="Source docs, one per line"></textarea>
        <button class="primary" onclick="directAnswer('${js(assignment.assignment_id)}',${assignment.task_version})">Submit Direct Answer</button>`);
    }
    detailActions.push(`<button onclick="releaseAssignment('${js(assignment.assignment_id)}')">Release</button>`);
  }
  const elapsed = taskElapsedSeconds(task);
  const taskMeta = `
    <div class="kv task-meta">
      <div><span>Status</span><strong>${esc(task.status)}</strong></div>
      <div class="answer-metric"><span>Answer</span><strong>${esc(candidate?.answer_text || "-")}</strong></div>
      <div><span>Elapsed</span><strong>${elapsed === null ? "-" : esc(formatDuration(elapsed))}</strong></div>
      <div><span>Updated</span><strong>${esc(formatClock(task.updated_at))}</strong></div>
      <div><span>Replans</span><strong>${esc(reasoningPayload?.summary?.replan_count ?? "-")}</strong></div>
    </div>`;
  return `<div class="detail-body">
    <p class="prompt">R${esc(task.round_num)} / ${esc(task.question_id)}</p>
    <div class="label">Prompt</div><pre>${esc(task.prompt)}</pre>
    ${taskMeta}
    ${answerSection}
    ${reasoningSection}
    ${submissionLine}
    ${cupFeedback}
    ${assignmentSummary}
    ${failure ? `<div class="message">${esc(failure.error_type)}: ${esc(failure.error_message)}</div>` : ""}
    ${task.error ? `<div class="message">${esc(task.error)}</div>` : ""}
    ${task.rejection_reason ? `<div class="message">${esc(task.rejection_reason)}</div>` : ""}
    <div class="actions">${detailActions.join("")}</div>
  </div>`;
}

function render() {
  const round = state.round || {};
  const tasks = state.tasks || [];
  if (selectedTaskId && !tasks.some(task => task.task_id === selectedTaskId)) {
    selectedTaskId = null;
  }
  const selectedTask = tasks.find(task => task.task_id === selectedTaskId) || tasks[0] || null;
  if (selectedTask && selectedTask.task_id !== selectedTaskId) selectedTaskId = selectedTask.task_id;
  document.getElementById("serverUrl").textContent = SERVER;
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
  document.getElementById("mCompleted").textContent = c.completed;
  lastEvent.textContent = state.last_event || "Waiting for events...";
  const listHtml = tasks.length
    ? tasks.map(task => {
        const claimedBy = task.assignments?.find(item => item.status === "ACTIVE");
        const claimedLabel = claimedBy
          ? `<span class="status claimed">CLAIMED · ${esc(workerMnemonic(claimedBy.worker_id))}</span>`
          : `<span class="status ${taskBadgeClass(task)}">${esc(taskBadge(task))}</span>`;
        return `<button class="task-row ${task.task_id === selectedTaskId ? "selected" : ""}" onclick="selectTask('${js(task.task_id)}')">
        <div class="row-main">
          <div class="row-title">R${esc(task.round_num)} / ${esc(task.question_id)}</div>
          <div class="row-preview">${esc(task.prompt || "")}</div>
        </div>
        ${claimedLabel}
      </button>`;
      }).join("")
    : `<div class="empty">No tasks yet.</div>`;
  const detailHtml = selectedTask ? renderDetail(selectedTask) : `<div class="detail-empty">No current-round tasks.</div>`;
  let html = `<section class="workerbar">
    <div id="error" class="error"></div>`;
  if (state.worker) {
    html += `<strong>${esc(workerLabel(workerId))}</strong>
      <input id="displayName" value="${esc(state.worker.display_name || "")}" placeholder="Worker name">
      <button onclick="saveName()">Save</button>`;
  }
  html += `</section>
  <section class="workspace">
    <aside class="list-panel">
      <div class="panel-head">
        <div class="panel-title">Tasks</div>
        <span class="pill"><strong>${esc(String(tasks.length))}</strong></span>
      </div>
      <div class="task-list">${listHtml}</div>
    </aside>
    <section class="detail-panel">
      <div class="panel-head">
        <div class="panel-title">Task Details</div>
        <span class="status ${selectedTask ? taskBadgeClass(selectedTask) : "queued"}">${selectedTask ? esc(taskBadge(selectedTask)) : "none"}</span>
      </div>
      ${detailHtml}
    </section>
  </section>`;
  app.innerHTML = html;
}

window.saveName = () => run(command(`/api/workers/${workerId}`, {
  display_name: document.getElementById("displayName").value,
}, "PUT"));
window.claimTask = taskId => run(command("/api/assignments", { worker_id: workerId, task_id: taskId }));
window.releaseAssignment = assignmentId => run(command(`/api/assignments/${assignmentId}/release`, { worker_id: workerId }));
window.submitReady = (taskId, assignmentId = null, version = null) => run(command("/api/submissions", {
  task_id: taskId,
  ...(assignmentId ? { worker_id: workerId, assignment_id: assignmentId, task_version: version } : {}),
}));
window.retryTask = (assignmentId, version) => run(command("/api/retries", {
  worker_id: workerId, assignment_id: assignmentId, task_version: version,
  feedback: document.getElementById(`feedback_${assignmentId}`).value,
}));
window.directAnswer = (assignmentId, version) => run(command("/api/human-answers", {
  worker_id: workerId, assignment_id: assignmentId, task_version: version,
  answer_text: document.getElementById(`answer_${assignmentId}`).value,
  reasoning: document.getElementById(`reasoning_${assignmentId}`).value,
  source_docs: document.getElementById(`sources_${assignmentId}`).value.split("\n").map(x => x.trim()).filter(Boolean),
}));
window.selectTask = selectTask;

hydrate();
connect();
setInterval(updateRoundTimer, 1000);
