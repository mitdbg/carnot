"use strict";
// Immersive human-review overlay. Opened for a task that has open reviews; it covers the app
// with a dimmed backdrop (the round timer in the header stays visible behind it) and walks the
// operator through that task's reviews one at a time. Each review is one of three kinds:
//   lookup         -> instruction + a value field (confirm/correct an external lookup)
//   figure         -> page viewer + a value field (read the answer off a chart)
//   verify_extract -> page viewer + an editable table/list of the model's extracted values
// Submitting POSTs the edited values to /api/reviews/<id>/resolve; the overlay then advances to
// the task's next open review, and closes (back to the main UI) after the last one. "X" exits
// without committing. The status SSE stream is the source of truth for which reviews are open;
// `syncTasks` re-reads it each tick so a resolved/cancelled review drops out live.

const ReviewOverlay = (function () {
  let activeTaskId = null;
  let reviews = [];            // open reviews for the active task (from the latest snapshot)
  let index = 0;               // which review we're showing
  let tasksRef = [];           // latest task snapshots (kept in sync by syncTasks)
  let submitting = false;

  // viewer state
  let pages = [];              // [{month, page}] parsed from the review's source_docs
  let pageIdx = 0;
  let zoom = 1, panX = 0, panY = 0;

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));

  function root() { return document.getElementById("reviewOverlay"); }

  function parsePages(sourceDocs) {
    const out = [];
    const seen = new Set();
    for (const doc of sourceDocs || []) {
      const m = /Treasury Bulletin (\d{4}-\d{2}) PDF page (\d+)/.exec(String(doc));
      if (m) {
        const key = `${m[1]}/${m[2]}`;
        if (!seen.has(key)) { seen.add(key); out.push({ month: m[1], page: Number(m[2]) }); }
      }
    }
    return out;
  }

  function isOpen() { return activeTaskId !== null; }

  function open(taskId) {
    const task = tasksRef.find((t) => t.task_id === taskId);
    if (!task || !(task.reviews || []).length) return;
    activeTaskId = taskId;
    reviews = task.reviews.slice();
    index = 0;
    render();
  }

  function close() {
    activeTaskId = null;
    reviews = [];
    index = 0;
    const el = root();
    if (el) el.hidden = true;
  }

  // Re-read the live snapshot. If the active task's open reviews changed (resolved elsewhere,
  // cancelled at round close, or the task vanished), reconcile: drop closed ones, and close the
  // overlay when nothing is left so it never strands on a stale review.
  function syncTasks(tasks) {
    tasksRef = tasks || [];
    if (!isOpen()) return;
    const task = tasksRef.find((t) => t.task_id === activeTaskId);
    const open = task ? (task.reviews || []) : [];
    const current = reviews[index];
    reviews = open.slice();
    if (!reviews.length) { close(); return; }
    // Keep showing the same review if it's still open; else clamp into range.
    const stillThere = current && reviews.findIndex((r) => r.review_id === current.review_id);
    index = (stillThere != null && stillThere >= 0) ? stillThere : Math.min(index, reviews.length - 1);
    render();
  }

  function current() { return reviews[index] || null; }

  // ── candidate editing ────────────────────────────────────────────────────────
  function candidates(review) {
    const g = review.guidance || {};
    return Array.isArray(g.candidates) ? g.candidates : [];
  }

  function candidateRow(c, i) {
    const valueStr = (c.value && typeof c.value === "object") ? JSON.stringify(c.value, null, 2) : String(c.value ?? "");
    const kind = c.kind || "scalar";
    return `<div class="review-cand" data-i="${i}">
      <label class="review-field"><span>description</span>
        <input type="text" data-f="description" value="${esc(c.description ?? "")}"></label>
      <label class="review-field"><span>value</span>
        <textarea data-f="value" rows="${kind === "scalar" ? 1 : 3}">${esc(valueStr)}</textarea></label>
      <div class="review-field-row">
        <label class="review-field"><span>unit</span>
          <input type="text" data-f="unit" value="${esc(c.unit ?? "")}"></label>
        <label class="review-field"><span>kind</span>
          <select data-f="kind">
            ${["scalar", "vector", "table"].map((k) => `<option value="${k}" ${k === kind ? "selected" : ""}>${k}</option>`).join("")}
          </select></label>
      </div>
      <div class="review-field-row">
        <label class="review-field"><span>index_name</span>
          <input type="text" data-f="index_name" value="${esc(c.index_name ?? "")}"></label>
        <label class="review-field"><span>row_name</span>
          <input type="text" data-f="row_name" value="${esc(c.row_name ?? "")}"></label>
        <label class="review-field"><span>col_name</span>
          <input type="text" data-f="col_name" value="${esc(c.col_name ?? "")}"></label>
      </div>
    </div>`;
  }

  // Read the edited rows back into AnnotatedValue field objects (positionally aligned with the
  // shown candidates, so the recompute overlays them onto the cached entries by index).
  function collectEdited() {
    const rows = root().querySelectorAll(".review-cand");
    const out = [];
    for (const row of rows) {
      const obj = {};
      for (const el of row.querySelectorAll("[data-f]")) {
        const f = el.getAttribute("data-f");
        let v = el.value;
        if (f === "value") {
          const t = v.trim();
          try { v = JSON.parse(t); } catch { v = t; }  // primitives parse (5, "x"); objects too
        } else if (v === "") {
          v = (f === "index_name" || f === "row_name" || f === "col_name") ? null : "";
        }
        obj[f] = v;
      }
      out.push(obj);
    }
    return out;
  }

  // ── page viewer ──────────────────────────────────────────────────────────────
  function applyTransform() {
    const img = root().querySelector(".review-page-img");
    if (img) img.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    const label = root().querySelector(".review-page-label");
    if (label && pages.length) label.textContent = `${pages[pageIdx].month}  p.${pages[pageIdx].page}  (${pageIdx + 1}/${pages.length})`;
  }
  function setPage(i) {
    if (!pages.length) return;
    pageIdx = Math.max(0, Math.min(pages.length - 1, i));
    zoom = 1; panX = 0; panY = 0;
    const img = root().querySelector(".review-page-img");
    if (img) img.src = `/api/source/${pages[pageIdx].month}/page/${pages[pageIdx].page}.png`;
    applyTransform();
  }
  function wireViewer() {
    const vp = root().querySelector(".review-viewport");
    if (!vp) return;
    vp.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = vp.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      const factor = Math.exp(-e.deltaY * 0.0015);          // trackpad pinch sends ctrlKey+wheel
      const next = Math.max(1, Math.min(8, zoom * factor));
      // keep the point under the cursor stationary while zooming
      panX = cx - (cx - panX) * (next / zoom);
      panY = cy - (cy - panY) * (next / zoom);
      zoom = next;
      applyTransform();
    }, { passive: false });
    let dragging = false, sx = 0, sy = 0;
    vp.addEventListener("mousedown", (e) => { dragging = true; sx = e.clientX - panX; sy = e.clientY - panY; vp.classList.add("is-panning"); });
    window.addEventListener("mousemove", (e) => { if (!dragging) return; panX = e.clientX - sx; panY = e.clientY - sy; applyTransform(); });
    window.addEventListener("mouseup", () => { dragging = false; vp.classList.remove("is-panning"); });
  }

  // ── render ───────────────────────────────────────────────────────────────────
  function render() {
    const el = root();
    if (!el) return;
    const review = current();
    if (!review) { close(); return; }
    el.hidden = false;
    pages = parsePages(review.source_docs);
    pageIdx = 0; zoom = 1; panX = 0; panY = 0;
    const task = tasksRef.find((t) => t.task_id === activeTaskId);
    const title = task ? `R${esc(task.round_num)} / ${esc(task.question_id)}` : esc(activeTaskId);
    const kindLabel = { lookup: "External Lookup", figure: "Visual QA", verify_extract: "Extract Validation" }[review.kind] || review.kind;
    const cands = candidates(review);
    const editor = cands.length
      ? cands.map(candidateRow).join("")
      : candidateRow({ description: "", value: "", kind: "scalar" }, 0);
    const viewer = pages.length ? `
      <div class="review-viewer">
        <div class="review-viewer-bar">
          <button class="review-nav" onclick="ReviewOverlay.prevPage()">‹</button>
          <span class="review-page-label"></span>
          <button class="review-nav" onclick="ReviewOverlay.nextPage()">›</button>
          <span class="review-zoom-hint">scroll / pinch to zoom · drag to pan</span>
        </div>
        <div class="review-viewport"><img class="review-page-img" alt="source page" draggable="false"></div>
      </div>` : "";

    el.innerHTML = `
      <div class="review-backdrop" onclick="ReviewOverlay.requestExit(event)"></div>
      <div class="review-panel ${pages.length ? "with-viewer" : ""}">
        <div class="review-head">
          <div>
            <span class="review-kind">${esc(kindLabel)}</span>
            <span class="review-task">${title}</span>
            <span class="review-count">review ${index + 1} of ${reviews.length}</span>
          </div>
          <button class="review-close" title="Exit without committing" onclick="ReviewOverlay.close()">✕</button>
        </div>
        <div class="review-body">
          <div class="review-left">
            <div class="review-prompt">${esc(task?.prompt || "")}</div>
            <div class="review-instructions">${esc(review.instructions || "")}</div>
            <div class="review-editor">${editor}</div>
            <div class="review-actions">
              <button class="review-submit primary" onclick="ReviewOverlay.submit()">Submit</button>
              <button class="review-accept" onclick="ReviewOverlay.acceptAsIs()" title="Keep the model's answer unchanged">Accept as-is</button>
              <span class="review-error" id="reviewError"></span>
            </div>
          </div>
          ${viewer}
        </div>
      </div>`;
    if (pages.length) { wireViewer(); setPage(0); }
  }

  function showError(msg) {
    const e = document.getElementById("reviewError");
    if (e) e.textContent = msg;
  }

  async function post(review, response) {
    if (submitting) return;
    submitting = true;
    showError("");
    try {
      const resp = await fetch(`/api/reviews/${encodeURIComponent(review.review_id)}/resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ response, source_docs: review.source_docs || [] }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || !data.ok) { showError(data.error || `resolve failed (${resp.status})`); submitting = false; return; }
      // Optimistically drop this review locally and advance; the SSE snapshot will confirm.
      reviews = reviews.filter((r) => r.review_id !== review.review_id);
      if (index >= reviews.length) index = reviews.length - 1;
      submitting = false;
      if (!reviews.length) close(); else render();
    } catch (err) {
      showError(String(err));
      submitting = false;
    }
  }

  function submit() {
    const review = current();
    if (!review) return;
    let edited;
    try { edited = collectEdited(); } catch (err) { showError(String(err)); return; }
    post(review, JSON.stringify(edited));
  }

  function acceptAsIs() {
    const review = current();
    if (review) post(review, "");   // empty response = keep the model's answer, no recompute
  }

  function requestExit(event) {
    // Click on the dimmed backdrop exits without committing (same as the ✕).
    if (event && event.target && event.target.classList.contains("review-backdrop")) close();
  }

  return {
    open, close, isOpen, syncTasks,
    submit, acceptAsIs, requestExit,
    prevPage: () => setPage(pageIdx - 1),
    nextPage: () => setPage(pageIdx + 1),
  };
})();
window.ReviewOverlay = ReviewOverlay;
