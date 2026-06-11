const SERVER = document.body.dataset.serverUrl || document.documentElement.dataset.serverUrl || "";
let state = { worker: null, round: {}, tasks: [], workers: [] };
let workerId = null;
let selectedTaskId = null;
const app = document.getElementById("app");
const lastEvent = document.getElementById("lastEvent");
const roundTimerMetric = document.getElementById("roundTimerMetric");
const roundTimer = document.getElementById("roundTimer");
const extractionModal = document.getElementById("extractionModal");
const extractionModalKicker = document.getElementById("extractionModalKicker");
const extractionModalTitle = document.getElementById("extractionModalTitle");
const extractionModalBody = document.getElementById("extractionModalBody");
let reasoningModalBranches = [];
let sourcePageCleanup = null;
let corpusDocuments = [];
let corpusDocumentsLoaded = false;
let corpusDocumentsError = "";
const interventionSourceSelections = new Map();
const interventionSourceQueries = new Map();

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

async function loadCorpusDocuments() {
  try {
    const response = await fetch("/api/corpus-documents");
    if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
    corpusDocuments = await response.json();
  } catch (error) {
    corpusDocumentsError = error.message;
  } finally {
    corpusDocumentsLoaded = true;
    render();
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
    awaitHuman: by(task => task.status === "AWAIT_HUMAN"),
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
  if (task.status === "AWAIT_HUMAN") return "await-human";
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

function retrieveNodeHtml(branch) {
  const searched = branch.searched || {};
  if (branch.kind !== "retrieve") {
    return `<strong>${esc(searched.target || "External lookup")}</strong>
      ${searched.src ? `<span class="flow-detail">${esc(searched.src)}</span>` : ""}`;
  }
  return `<strong>${esc(searched.key || "Retrieve")}</strong>
    ${searched.period ? `<span class="flow-detail">${esc(searched.period)}</span>` : ""}
    ${searched.as_of ? `<span class="flow-detail">as of ${esc(searched.as_of)}</span>` : ""}
    ${searched.visual_only === true ? `<span class="flow-badge">visual only</span>` : ""}`;
}

function extractedSourceDocsHtml(entry) {
  if (!entry?.bulletin || !Array.isArray(entry.pages) || !entry.pages.length) return "";
  return `<div class="value-sources">
    ${entry.pages.map(page => sourceDocHtml(
      `Treasury Bulletin ${entry.bulletin} PDF page ${page}`
    )).join("")}
  </div>`;
}

function extractedValueHtml(entry) {
  const rawValue = entry?.value;
  const inferredKind = rawValue && typeof rawValue === "object"
    ? Object.values(rawValue).some(value => value && typeof value === "object")
      ? "table"
      : "vector"
    : "scalar";
  const kind = entry?.value_kind || entry?.type || inferredKind;
  const unit = entry?.unit ? String(entry.unit) : "";
  if (kind === "scalar") {
    return `<div class="extracted-scalar">
      ${entry.description ? `<span>${esc(entry.description)}</span>` : ""}
      <strong>${esc(entry.value)}${unit ? ` ${esc(unit)}` : ""}</strong>
      ${extractedSourceDocsHtml(entry)}
    </div>`;
  }
  if (kind === "vector" && rawValue && typeof rawValue === "object") {
    const title = [entry.description, unit].filter(Boolean).join(" · ");
    return `<div class="extracted-vector">
      ${title ? `<strong class="vector-title">${esc(title)}</strong>` : ""}
      <table>
        <tbody>
          ${Object.entries(rawValue).map(([label, value]) => `
            <tr><th>${esc(label)}</th><td>${esc(value)}</td></tr>
          `).join("")}
        </tbody>
      </table>
      ${extractedSourceDocsHtml(entry)}
    </div>`;
  }
  if (kind === "table" && rawValue && typeof rawValue === "object") {
    const rows = Object.entries(rawValue);
    const columns = [...new Set(rows.flatMap(([, row]) => (
      row && typeof row === "object" ? Object.keys(row) : []
    )))];
    const title = [entry.description, unit].filter(Boolean).join(" · ");
    return `<div class="extracted-vector">
      ${title ? `<strong class="vector-title">${esc(title)}</strong>` : ""}
      <div class="table-scroll">
        <table>
          <thead><tr><th>${esc(entry.row_name || "")}</th>${columns.map(column => `<th>${esc(column)}</th>`).join("")}</tr></thead>
          <tbody>
            ${rows.map(([label, row]) => `<tr>
              <th>${esc(label)}</th>
              ${columns.map(column => `<td>${esc(row?.[column] ?? "")}</td>`).join("")}
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
      ${extractedSourceDocsHtml(entry)}
    </div>`;
  }
  return `<pre class="value-fallback">${esc(formatReasoningValue(entry))}</pre>`;
}

function sourcePageCardHtml(source) {
  const label = `Treasury Bulletin ${source.bulletin} PDF page ${source.page}`;
  return `<button type="button" class="extraction-card source-card" onclick="openSourcePage('${js(source.bulletin)}', ${Number(source.page)})">
    <span class="flow-kicker">Source page</span>
    <strong>${esc(label)}</strong>
  </button>`;
}

function blockCardHtml(block, branchIndex, blockIndex) {
  const title = block.title || (block.block_index === null ? "Whole page" : `Block ${block.block_index}`);
  const continuationPages = (block.member_pages || []).filter(source => (
    source.bulletin !== block.bulletin || source.page !== block.page
  ));
  return `<button type="button" class="extraction-card block-card" onclick="openExtractionModal(${branchIndex},${blockIndex})">
    <span class="flow-kicker">${esc(block.kind || "block")} · page ${esc(block.page)} · index ${esc(block.block_index ?? "whole page")}</span>
    <strong>${esc(title)}</strong>
    ${continuationPages.length ? `<span class="block-field"><b>Continues</b>${continuationPages.map(source => `${source.bulletin} p.${source.page}`).join(", ")}</span>` : ""}
  </button>`;
}

function extractionNodeHtml(branch, branchIndex) {
  const blocks = Array.isArray(branch.blocks) ? branch.blocks : [];
  if (!blocks.length) {
    return `<span class="flow-empty">No selected block metadata captured.</span>`;
  }
  const pageRows = [];
  const rowsByPage = new Map();
  blocks.forEach((block, blockIndex) => {
    const key = `${block.bulletin}:${block.page}`;
    if (!rowsByPage.has(key)) {
      const row = {
        source: { bulletin: block.bulletin, page: block.page },
        blocks: [],
      };
      rowsByPage.set(key, row);
      pageRows.push(row);
    }
    rowsByPage.get(key).blocks.push({ block, blockIndex });
  });
  return `<div class="page-extraction-list">
    ${pageRows.map(row => `
      <div class="page-extraction-row">
        ${sourcePageCardHtml(row.source)}
        <div class="page-block-list">
          ${row.blocks.map(({ block, blockIndex }) => blockCardHtml(block, branchIndex, blockIndex)).join("")}
        </div>
      </div>
    `).join("")}
  </div>`;
}

function reasoningVisual(reasoning, answer) {
  const payload = parseReasoningPayload(reasoning);
  if (!payload) return "";
  const branches = payload.branches || [];
  const branchOffset = reasoningModalBranches.length;
  reasoningModalBranches.push(...branches);
  const code = payload.python_code || "";
  const attempts = payload.python_attempts || [];
  return `<section class="reasoning-shell">
    <div class="label">Reasoning Flow</div>
    <div class="reasoning-flow">
      <div class="flow-branches">
      ${branches.map((branch, branchIndex) => `
        <article class="flow-lane ${esc(branch.status || "ok")}">
          <div class="flow-node search-node">
            <span class="flow-kicker">${esc((branch.kind || "branch").replace("_", " "))} ${esc(branch.branch_id)}</span>
            ${retrieveNodeHtml(branch)}
          </div>
          <div class="flow-arrow" aria-hidden="true"></div>
          <div class="flow-node value-node">
            <span class="flow-kicker">${esc(branch.output_step || "selected blocks")}</span>
            ${branch.kind === "retrieve"
              ? extractionNodeHtml(branch, branchOffset + branchIndex)
              : `<strong>${esc(formatReasoningValue(branch.output) || "No value captured")}</strong>`}
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

function previousRoundFlowHtml(intervention) {
  const guidance = intervention?.guidance || {};
  const branches = Array.isArray(guidance.previous_round_plan)
    ? guidance.previous_round_plan
    : [];
  if (!branches.length) return "";
  const branchOffset = reasoningModalBranches.length;
  reasoningModalBranches.push(...branches);

  return `<section class="reasoning-shell previous-round-flow">
    <div class="label">Previous Round Flow</div>
    <div class="reasoning-flow">
      <div class="flow-branches">
        ${branches.map((branch, branchIndex) => {
          const blocks = Array.isArray(branch.blocks) ? branch.blocks : [];
          const consideredPages = Array.isArray(branch.considered_pages)
            ? branch.considered_pages.filter(page => (
                page?.bulletin && page?.page !== null && page?.page !== undefined
              ))
            : [];
          const outcome = branch.outcome_status || "succeeded";
          return `<article class="flow-lane ${esc(outcome)}">
            <div class="flow-node search-node">
              <span class="flow-kicker">${esc((branch.kind || "branch").replace("_", " "))} ${esc(branch.branch_id ?? "")}</span>
              ${retrieveNodeHtml(branch)}
            </div>
            <div class="flow-arrow" aria-hidden="true"></div>
            <div class="flow-node value-node">
              <span class="flow-kicker">${esc((branch.execution_status || "executed").replace("_", " "))} · ${esc(outcome)}</span>
              ${branch.error
                ? `<strong>${esc(branch.error)}</strong>`
                : branch.kind === "retrieve"
                  ? extractionNodeHtml(branch, branchOffset + branchIndex)
                  : `<strong>${esc(formatReasoningValue(branch.output) || "No value captured")}</strong>`}
              ${outcome === "failed" && !blocks.length && consideredPages.length
                ? `<details class="considered-pages">
                    <summary>Candidate pages (${esc(consideredPages.length)})</summary>
                    <div class="value-sources">${consideredPages.map(page => (
                      sourceDocHtml(`Treasury Bulletin ${page.bulletin} PDF page ${page.page}`)
                    )).join("")}</div>
                  </details>`
                : ""}
            </div>
          </article>`;
        }).join("")}
      </div>
    </div>
  </section>`;
}

function openExtractionModal(branchIndex, blockIndex) {
  const branch = reasoningModalBranches[branchIndex];
  const block = branch?.blocks?.[blockIndex];
  if (!branch || !block) return;
  sourcePageCleanup?.();
  sourcePageCleanup = null;
  extractionModal.classList.remove("source-page-modal");
  extractionModalBody.classList.remove("source-page-body");
  const values = branch.output?.type === "values" && Array.isArray(branch.output.values)
    ? branch.output.values.filter(value => (
        value.source_block_page === block.page
        && (value.source_block_index ?? null) === (block.block_index ?? null)
      ))
    : [];
  extractionModalKicker.textContent = "Extraction Results";
  extractionModalTitle.textContent = block.title
    || (block.block_index === null ? `Whole page ${block.page}` : `Block ${block.block_index}`);
  extractionModalBody.innerHTML = `
    ${block.summary ? `<p class="block-summary">${esc(block.summary)}</p>` : ""}
    ${values.length
      ? `<div class="modal-values">${values.map(extractedValueHtml).join("")}</div>`
      : `<div class="empty">No extracted values were mapped to this block.</div>`}
  `;
  extractionModal.showModal();
}

function openSourcePage(month, page) {
  const label = `Treasury Bulletin ${month} PDF page ${page}`;
  const imageUrl = `/api/source/${encodeURIComponent(month)}/page/${page}.png`;
  sourcePageCleanup?.();
  extractionModal.classList.add("source-page-modal");
  extractionModalBody.classList.add("source-page-body");
  extractionModalKicker.textContent = "Source Page";
  extractionModalTitle.textContent = label;
  extractionModalBody.innerHTML = `
    <div class="source-page-actions">
      <a href="/api/source/${encodeURIComponent(month)}#page=${page}" target="_blank" rel="noopener noreferrer">Open full PDF</a>
      <div class="source-page-controls" aria-label="Page zoom controls">
        <button type="button" data-zoom="out" title="Zoom out">−</button>
        <output class="source-page-zoom" aria-live="polite">100%</output>
        <button type="button" data-zoom="in" title="Zoom in">+</button>
        <button type="button" data-zoom="fit">Fit</button>
        <button type="button" data-zoom="reset">Reset</button>
      </div>
    </div>
    <div class="source-page-viewport">
      <span class="source-page-loading">Rendering page...</span>
    </div>
  `;
  const viewport = extractionModalBody.querySelector(".source-page-viewport");
  const loading = extractionModalBody.querySelector(".source-page-loading");
  const zoomOutput = extractionModalBody.querySelector(".source-page-zoom");
  const image = document.createElement("img");
  image.className = "source-page-image";
  image.alt = label;
  image.draggable = false;
  viewport.appendChild(image);

  let scale = 1;
  let translateX = 0;
  let translateY = 0;
  let dragging = false;
  let pointerX = 0;
  let pointerY = 0;

  function renderTransform() {
    image.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    zoomOutput.value = `${Math.round(scale * 100)}%`;
    viewport.classList.toggle("can-pan", scale > 1);
  }

  function setScale(nextScale, originX = viewport.clientWidth / 2, originY = viewport.clientHeight / 2) {
    const boundedScale = Math.min(5, Math.max(0.25, nextScale));
    const ratio = boundedScale / scale;
    translateX = originX - (originX - translateX) * ratio;
    translateY = originY - (originY - translateY) * ratio;
    scale = boundedScale;
    renderTransform();
  }

  function fitPage() {
    if (!image.naturalWidth || !image.naturalHeight) return;
    scale = Math.min(
      viewport.clientWidth / image.naturalWidth,
      viewport.clientHeight / image.naturalHeight,
    );
    translateX = (viewport.clientWidth - image.naturalWidth * scale) / 2;
    translateY = (viewport.clientHeight - image.naturalHeight * scale) / 2;
    renderTransform();
  }

  function resetPage() {
    scale = 1;
    translateX = (viewport.clientWidth - image.naturalWidth) / 2;
    translateY = 16;
    renderTransform();
  }

  function handleWheel(event) {
    event.preventDefault();
    const bounds = viewport.getBoundingClientRect();
    const factor = Math.exp(-event.deltaY * 0.0015);
    setScale(scale * factor, event.clientX - bounds.left, event.clientY - bounds.top);
  }

  function handlePointerDown(event) {
    if (event.button !== 0) return;
    dragging = true;
    pointerX = event.clientX;
    pointerY = event.clientY;
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add("is-panning");
  }

  function handlePointerMove(event) {
    if (!dragging) return;
    translateX += event.clientX - pointerX;
    translateY += event.clientY - pointerY;
    pointerX = event.clientX;
    pointerY = event.clientY;
    renderTransform();
  }

  function handlePointerUp(event) {
    if (!dragging) return;
    dragging = false;
    if (viewport.hasPointerCapture(event.pointerId)) {
      viewport.releasePointerCapture(event.pointerId);
    }
    viewport.classList.remove("is-panning");
  }

  const controls = extractionModalBody.querySelector(".source-page-controls");
  function handleControls(event) {
    const action = event.target.closest("button")?.dataset.zoom;
    if (action === "in") setScale(scale * 1.25);
    else if (action === "out") setScale(scale / 1.25);
    else if (action === "fit") fitPage();
    else if (action === "reset") resetPage();
  }

  function handleLoad() {
    loading?.remove();
    fitPage();
  }

  function handleError() {
    if (loading) {
      loading.className = "message";
      loading.textContent = "The requested page could not be rendered.";
    }
    image.remove();
  }

  controls.addEventListener("click", handleControls);
  viewport.addEventListener("wheel", handleWheel, { passive: false });
  viewport.addEventListener("pointerdown", handlePointerDown);
  viewport.addEventListener("pointermove", handlePointerMove);
  viewport.addEventListener("pointerup", handlePointerUp);
  viewport.addEventListener("pointercancel", handlePointerUp);
  viewport.addEventListener("dblclick", fitPage);
  image.addEventListener("load", handleLoad);
  image.addEventListener("error", handleError);
  sourcePageCleanup = () => {
    controls.removeEventListener("click", handleControls);
    viewport.removeEventListener("wheel", handleWheel);
    viewport.removeEventListener("pointerdown", handlePointerDown);
    viewport.removeEventListener("pointermove", handlePointerMove);
    viewport.removeEventListener("pointerup", handlePointerUp);
    viewport.removeEventListener("pointercancel", handlePointerUp);
    viewport.removeEventListener("dblclick", fitPage);
    image.removeEventListener("load", handleLoad);
    image.removeEventListener("error", handleError);
  };

  extractionModal.showModal();
  image.src = imageUrl;
}

function resetExtractionModal() {
  sourcePageCleanup?.();
  sourcePageCleanup = null;
  extractionModal.classList.remove("source-page-modal");
  extractionModalBody.classList.remove("source-page-body");
  extractionModalBody.innerHTML = "";
}

function closeExtractionModal() {
  extractionModal.close();
  resetExtractionModal();
}

extractionModal.addEventListener("close", resetExtractionModal);

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

function updateTaskElapsedTimer() {
  const elapsedElement = document.getElementById("taskElapsed");
  if (!elapsedElement) return;
  const task = state.tasks?.find(item => item.task_id === selectedTaskId);
  const elapsed = task ? taskElapsedSeconds(task) : null;
  elapsedElement.textContent = elapsed === null ? "-" : formatDuration(elapsed);
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
      return `<a class="doc" href="/api/source/${encodeURIComponent(month)}/page/${page}.png" onclick="event.preventDefault(); openSourcePage('${js(month)}', ${page})" title="Show ${esc(label)}">${esc(label)}</a>`;
  }
  const bulletinDocumentMatch = label.match(/^Treasury Bulletin ([0-9]{4}-(?:0[1-9]|1[0-2])) PDF$/);
  if (bulletinDocumentMatch) {
    const month = bulletinDocumentMatch[1];
    return `<a class="doc" href="/api/source/${encodeURIComponent(month)}" target="_blank" rel="noopener noreferrer" title="Open ${esc(label)}">${esc(label)}</a>`;
  }
  if (/^https?:\/\//i.test(label) || /#p\d+$/i.test(label)) {
    return `<a class="doc" href="${esc(label)}" target="_blank" rel="noopener noreferrer" title="Open ${esc(label)}">${esc(label)}</a>`;
  }
  return `<span class="doc" title="${esc(label)}">${esc(label)}</span>`;
}

function selectedInterventionSources(pickerKey) {
  if (!interventionSourceSelections.has(pickerKey)) {
    interventionSourceSelections.set(pickerKey, new Set());
  }
  return interventionSourceSelections.get(pickerKey);
}

function matchingCorpusDocuments(query) {
  const tokens = String(query || "").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  if (!tokens.length) return corpusDocuments;
  return corpusDocuments.filter(document => {
    const searchable = [
      document.id,
      document.title,
      document.filename,
      document.reference,
    ].join(" ").toLowerCase();
    return tokens.every(token => searchable.includes(token));
  });
}

function corpusDocumentPickerContents(pickerKey) {
  const selected = selectedInterventionSources(pickerKey);
  const query = interventionSourceQueries.get(pickerKey) || "";
  const matches = matchingCorpusDocuments(query);
  let status = "";
  if (!corpusDocumentsLoaded) {
    status = `<div class="corpus-document-empty">Loading corpus documents...</div>`;
  } else if (corpusDocumentsError) {
    status = `<div class="corpus-document-empty failed">Could not load corpus documents: ${esc(corpusDocumentsError)}</div>`;
  } else if (!corpusDocuments.length) {
    status = `<div class="corpus-document-empty">No corpus documents configured.</div>`;
  } else if (!matches.length) {
    status = `<div class="corpus-document-empty">No documents match “${esc(query)}”.</div>`;
  }
  return `
    <div class="selected-corpus-documents">
      ${Array.from(selected).map(reference => `
        <button type="button" class="selected-document-chip" onclick="removeInterventionSource('${js(pickerKey)}', '${js(reference)}')">
          ${esc(reference)} <span aria-hidden="true">×</span>
        </button>`).join("")}
    </div>
    <input
      id="intervention_source_search_${esc(pickerKey)}"
      class="corpus-document-search"
      value="${esc(query)}"
      placeholder="Search corpus documents"
      autocomplete="off"
      oninput="filterInterventionSources('${js(pickerKey)}', this.value)"
    >
    <div class="corpus-document-options">
      ${status || matches.map(document => `
        <label class="corpus-document-option">
          <input
            type="checkbox"
            ${selected.has(document.reference) ? "checked" : ""}
            onchange="toggleInterventionSource('${js(pickerKey)}', '${js(document.reference)}', this.checked)"
          >
          <span>
            <strong>${esc(document.title)}</strong>
            <small>${esc(document.filename)}</small>
          </span>
        </label>`).join("")}
    </div>`;
}

function corpusDocumentPickerHtml(pickerKey, label = "Source Documents") {
  return `<div class="corpus-document-field">
    <div class="label">${esc(label)}</div>
    <div id="intervention_sources_${esc(pickerKey)}" class="corpus-document-picker">
      ${corpusDocumentPickerContents(pickerKey)}
    </div>
  </div>`;
}

function renderCorpusDocumentPicker(pickerKey, focusSearch = false) {
  const picker = document.getElementById(`intervention_sources_${pickerKey}`);
  if (!picker) return;
  picker.innerHTML = corpusDocumentPickerContents(pickerKey);
  if (focusSearch) {
    const search = document.getElementById(`intervention_source_search_${pickerKey}`);
    search?.focus();
    if (search) search.setSelectionRange(search.value.length, search.value.length);
  }
}

function interventionHtml(intervention) {
  const isMine = intervention.status === "CLAIMED" && intervention.claimed_by === workerId;
  const requestedSources = intervention.source_docs?.length
    ? `<div class="docs">${intervention.source_docs.map(sourceDocHtml).join("")}</div>`
    : "";
  const responseSources = intervention.response_source_docs?.length
    ? `<div class="docs">${intervention.response_source_docs.map(sourceDocHtml).join("")}</div>`
    : "";
  const guidance = intervention.guidance || {};
  const likelyPages = Array.isArray(guidance.likely_pages)
    ? guidance.likely_pages.filter(page => page?.bulletin && page?.page !== null && page?.page !== undefined)
    : [];
  const partialPlan = Array.isArray(guidance.partial_plan) ? guidance.partial_plan : [];
  const failedBranches = Array.isArray(guidance.failed_branches) ? guidance.failed_branches : [];
  const failedRetrieveBranches = failedBranches.filter(item => (
    Number.isInteger(item.branch_id) && item.branch?.kind === "retrieve"
  ));
  const directiveBranches = failedRetrieveBranches.length
    ? failedRetrieveBranches.map(item => ({
        branch_id: item.branch_id,
        key: item.branch.key,
      }))
    : partialPlan.filter(branch => (
        Number.isInteger(branch.branch_id) && branch.kind === "retrieve"
      ));
  const missingValues = intervention.kind === "missing_data" && Array.isArray(guidance.missing)
    ? guidance.missing.map(value => String(value).trim()).filter(Boolean)
    : [];
  const guidanceHtml = Object.keys(guidance).length
    ? `<div class="intervention-guidance">
        ${guidance.recovery_round
          ? `<div class="guidance-metric"><span>Recovery round</span><strong>${esc(guidance.recovery_round)}</strong></div>`
          : ""}
        ${likelyPages.length
          ? `<div class="label">Likely Source Pages</div><div class="docs">${likelyPages.map(page => (
              sourceDocHtml(`Treasury Bulletin ${page.bulletin} PDF page ${page.page}`)
            )).join("")}</div>`
          : ""}
        ${partialPlan.length
          ? `<div class="label">Partial Plan</div><div class="guidance-list">${partialPlan.map(branch => `
              <div class="guidance-item">
                <strong>${esc((branch.kind || "branch").replace("_", " "))} ${esc(branch.branch_id ?? "")}</strong>
                <span>${esc(branch.key || branch.target || "")}</span>
                ${branch.period ? `<small>${esc(branch.period)}</small>` : ""}
              </div>`).join("")}</div>`
          : ""}
        ${failedBranches.length
          ? `<div class="label">Failed Branches</div><div class="guidance-list">${failedBranches.map(item => `
              <div class="guidance-item failed">
                <strong>${esc(item.branch?.key || item.branch?.target || item.branch?.kind || "Branch")}</strong>
                <span>${esc(item.error || "")}</span>
              </div>`).join("")}</div>`
          : ""}
      </div>`
    : "";
  const missingSummary = missingValues.length && !isMine
    ? `<div class="missing-information">
        <div class="label">Missing Information</div>
        ${missingValues.map(value => `<div class="missing-information-item">${esc(value)}</div>`).join("")}
      </div>`
    : "";
  const requestContext = missingValues.length
    ? `<details class="request-context">
        <summary>Request context</summary>
        <p>${esc(intervention.instructions)}</p>
        ${intervention.context ? `<p class="intervention-context">${esc(intervention.context)}</p>` : ""}
        ${requestedSources}
        ${guidanceHtml}
      </details>`
    : `<p>${esc(intervention.instructions)}</p>
      ${intervention.context ? `<p class="intervention-context">${esc(intervention.context)}</p>` : ""}
      ${requestedSources}
      ${guidanceHtml}`;
  let controls = "";
  if (intervention.status === "PENDING" && workerId) {
    controls = `<button class="primary" onclick="claimIntervention('${js(intervention.intervention_id)}')">Claim Request</button>`;
  } else if (isMine) {
    const directivePickers = directiveBranches.map(branch => {
      const pickerKey = `${intervention.intervention_id}:${branch.branch_id}`;
      return corpusDocumentPickerHtml(
        pickerKey,
        `Documents for branch ${branch.branch_id}: ${branch.key || "retrieval"}`,
      );
    }).join("");
    controls = `
      ${missingValues.length
        ? `<div id="intervention_missing_${esc(intervention.intervention_id)}" class="missing-information-form">
            <div class="label">Missing Information</div>
            ${missingValues.map((value, index) => `
              <label for="intervention_missing_${esc(intervention.intervention_id)}_${index}">${esc(value)}</label>
              <input
                id="intervention_missing_${esc(intervention.intervention_id)}_${index}"
                data-missing-label="${esc(value)}"
                placeholder="Enter value"
                required
              >`).join("")}
          </div>`
        : `<textarea id="intervention_response_${esc(intervention.intervention_id)}" placeholder="Human response"></textarea>`}
      ${directivePickers}
      <button class="primary" onclick="resolveIntervention('${js(intervention.intervention_id)}')">Send Response</button>
      <button onclick="releaseIntervention('${js(intervention.intervention_id)}')">Release</button>`;
  } else if (intervention.status === "CLAIMED") {
    controls = `<span class="status claimed">CLAIMED · ${esc(workerMnemonic(intervention.claimed_by))}</span>`;
  }
  return `<article class="intervention-card ${intervention.status === "RESOLVED" ? "resolved" : ""}">
    <div class="intervention-head">
      <strong>${esc(intervention.kind)}</strong>
      <span class="status ${intervention.status === "RESOLVED" ? "submitted" : "await-human"}">${esc(intervention.status)}</span>
    </div>
    ${missingSummary}
    ${missingValues.length && isMine ? `<div class="actions">${controls}</div>` : ""}
    ${requestContext}
    ${intervention.response ? `<div class="label">Response</div><pre>${esc(intervention.response)}</pre>${responseSources}` : ""}
    ${missingValues.length && isMine ? "" : `<div class="actions">${controls}</div>`}
  </article>`;
}

function renderDetail(task) {
  reasoningModalBranches = [];
  const candidates = task.answer_candidates || [];
  const failures = task.failures || [];
  const submissions = task.submissions || [];
  const candidate = candidates[candidates.length - 1];
  const failure = failures[failures.length - 1];
  const submission = submissions[submissions.length - 1];
  const assignment = activeAssignment(task);
  const interventions = task.human_interventions || [];
  const latestIntervention = interventions[interventions.length - 1];
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
  const pendingReplanCount = interventions.reduce(
    (maximum, intervention) => Math.max(
      maximum,
      Number(intervention.guidance?.recovery_round || 0),
    ),
    0,
  );
  const replanDisplay = reasoningPayload?.summary?.replan_count
    ?? (pendingReplanCount || "-");
  const reasoningSection = candidate
    ? reasoningVisual(candidate.reasoning, candidate.answer_text)
      || `<div class="label">Reasoning</div><pre>${esc(candidate.reasoning)}</pre>`
    : "";
  const interventionSection = interventions.length
    ? `<div class="label">Human Interventions</div>
      <div class="intervention-list">
        ${interventions.map(intervention => interventionHtml(intervention)).join("")}
      </div>`
    : "";
  const previousRoundFlow = previousRoundFlowHtml(latestIntervention);
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
      <div><span>Elapsed</span><strong id="taskElapsed">${elapsed === null ? "-" : esc(formatDuration(elapsed))}</strong></div>
      <div><span>Updated</span><strong>${esc(formatClock(task.updated_at))}</strong></div>
      <div><span>Replans</span><strong>${esc(replanDisplay)}</strong></div>
    </div>`;
  return `<div class="detail-body">
    <p class="prompt">R${esc(task.round_num)} / ${esc(task.question_id)}</p>
    <div class="label">Prompt</div><pre>${esc(task.prompt)}</pre>
    ${taskMeta}
    ${answerSection}
    ${previousRoundFlow}
    ${interventionSection}
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
  document.getElementById("mAwaitHuman").textContent = c.awaitHuman;
  document.getElementById("mReady").textContent = c.ready;
  document.getElementById("mSubmitted").textContent = c.submitted;
  document.getElementById("mFailed").textContent = c.failed;
  document.getElementById("mCompleted").textContent = c.completed;
  lastEvent.textContent = state.last_event || "Waiting for events...";
  const listHtml = tasks.length
    ? tasks.map(task => {
        const claimedIntervention = task.human_interventions?.find(item => item.status === "CLAIMED");
        const claimedBy = task.assignments?.find(item => item.status === "ACTIVE");
        const claimedLabel = claimedBy
          ? `<span class="status claimed">CLAIMED · ${esc(workerMnemonic(claimedBy.worker_id))}</span>`
          : claimedIntervention
            ? `<span class="status claimed">HUMAN · ${esc(workerMnemonic(claimedIntervention.claimed_by))}</span>`
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
window.claimIntervention = interventionId => run(command(
  `/api/interventions/${interventionId}/claim`,
  { worker_id: workerId },
));
window.releaseIntervention = interventionId => run(command(
  `/api/interventions/${interventionId}/release`,
  { worker_id: workerId },
));
window.filterInterventionSources = (interventionId, query) => {
  interventionSourceQueries.set(interventionId, query);
  renderCorpusDocumentPicker(interventionId, true);
};
window.toggleInterventionSource = (interventionId, reference, checked) => {
  const selected = selectedInterventionSources(interventionId);
  if (checked) selected.add(reference);
  else selected.delete(reference);
  renderCorpusDocumentPicker(interventionId, true);
};
window.removeInterventionSource = (interventionId, reference) => {
  selectedInterventionSources(interventionId).delete(reference);
  renderCorpusDocumentPicker(interventionId, true);
};
window.resolveIntervention = interventionId => {
  const task = state.tasks.find(item => (
    item.human_interventions?.some(intervention => intervention.intervention_id === interventionId)
  ));
  const intervention = task?.human_interventions?.find(item => item.intervention_id === interventionId);
  const guidance = intervention?.guidance || {};
  const failed = Array.isArray(guidance.failed_branches)
    ? guidance.failed_branches.filter(item => (
        Number.isInteger(item.branch_id) && item.branch?.kind === "retrieve"
      ))
    : [];
  const directiveBranches = failed.length
    ? failed.map(item => item.branch_id)
    : (Array.isArray(guidance.partial_plan) ? guidance.partial_plan : [])
        .filter(branch => Number.isInteger(branch.branch_id) && branch.kind === "retrieve")
        .map(branch => branch.branch_id);
  const retrievalDirectives = directiveBranches.map(branchId => {
    const documents = Array.from(
      selectedInterventionSources(`${interventionId}:${branchId}`),
    );
    return { branch_id: branchId, documents };
  }).filter(item => item.documents.length);
  const missingInputs = Array.from(
    document.querySelectorAll(`#intervention_missing_${interventionId} input[data-missing-label]`),
  );
  const response = missingInputs.length
    ? missingInputs
        .filter(input => input.value.trim())
        .map(input => `${input.dataset.missingLabel}: ${input.value.trim()}`)
        .join("\n")
    : (document.getElementById(`intervention_response_${interventionId}`)?.value || "");
  if (!response.trim() && !retrievalDirectives.length) {
    missingInputs[0]?.reportValidity();
    return;
  }
  const sourceDocs = [
    ...new Set(retrievalDirectives.flatMap(item => item.documents)),
  ];
  return run(command(
    `/api/interventions/${interventionId}/resolve`,
    {
      worker_id: workerId,
      response,
      source_docs: sourceDocs,
      retrieval_directives: retrievalDirectives,
    },
  ));
};
window.selectTask = selectTask;
window.openExtractionModal = openExtractionModal;
window.openSourcePage = openSourcePage;
window.closeExtractionModal = closeExtractionModal;

loadCorpusDocuments();
hydrate();
connect();
setInterval(() => {
  updateRoundTimer();
  updateTaskElapsedTimer();
}, 1000);
