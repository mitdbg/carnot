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
  let fitScale = 1;            // scale that fits the whole page in the viewport (and the min zoom)

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

  // Re-read the live snapshot WITHOUT clobbering the operator's work. Status snapshots arrive
  // every tick, so we must NOT rebuild the DOM (which would wipe in-progress edits and reset the
  // page viewer's pan/zoom) while the operator is on a review that's still open. We only act when
  // the CURRENT review disappeared externally (resolved elsewhere / cancelled at round close):
  // then advance to the next, or close if none remain. Post-submit advancing is the explicit
  // render() path in post(), not here.
  function syncTasks(tasks) {
    tasksRef = tasks || [];
    if (!isOpen()) return;
    const task = tasksRef.find((t) => t.task_id === activeTaskId);
    const open = task ? (task.reviews || []) : [];
    const cur = reviews[index];
    const curStillOpen = cur && open.some((r) => r.review_id === cur.review_id);
    reviews = open.slice();
    if (!reviews.length) { close(); return; }
    if (curStillOpen) {
      // Same review still in progress — keep the live DOM (edits + viewer) intact, just keep our
      // index pointed at it and refresh the "review N of M" counter text in place.
      index = reviews.findIndex((r) => r.review_id === cur.review_id);
      updateCount();
      return;
    }
    // The current review vanished out from under us → advance (or clamp) and rebuild once.
    index = Math.min(index, reviews.length - 1);
    render();
  }

  function updateCount() {
    const el = root() && root().querySelector(".review-count");
    if (el) el.textContent = `review ${index + 1} of ${reviews.length}`;
  }

  function current() { return reviews[index] || null; }

  // ── candidate editing ────────────────────────────────────────────────────────
  function candidates(review) {
    const g = review.guidance || {};
    return Array.isArray(g.candidates) ? g.candidates : [];
  }

  // One value box, decluttered: only description / unit / value are editable; kind, index_name,
  // row_name, col_name and machine provenance are preserved from the original extraction via the
  // `_src` index (the card's data-src). A ✕ removes the box entirely (the value is then dropped
  // from what's fed to the recompute). `i` is the value's ORIGINAL index in the candidate list.
  function candidateRow(c, i) {
    const valueStr = (c.value && typeof c.value === "object") ? JSON.stringify(c.value, null, 2) : String(c.value ?? "");
    const kind = c.kind || "scalar";
    const meta = kind === "scalar" ? "" : `<span class="review-cand-kind">${esc(kind)}${c.index_name ? " · " + esc(c.index_name) : ""}</span>`;
    return `<div class="review-cand" data-src="${i}">
      <div class="review-cand-head">
        ${meta}
        <button class="review-cand-del" title="Remove this value" onclick="ReviewOverlay.deleteCard(${i})">✕</button>
      </div>
      <label class="review-field"><span>description</span>
        <input type="text" data-f="description" value="${esc(c.description ?? "")}"></label>
      <label class="review-field"><span>unit</span>
        <input type="text" data-f="unit" value="${esc(c.unit ?? "")}"></label>
      <label class="review-field"><span>value</span>
        <textarea data-f="value" class="review-value" oninput="ReviewOverlay.autosize(this)">${esc(valueStr)}</textarea></label>
    </div>`;
  }

  // Read the surviving value boxes back into source-indexed override items
  // ({_src, description, unit, value}); deleted boxes are simply absent. The recompute overlays
  // the edited fields onto the cached entry at `_src`, preserving its structure + provenance.
  function collectEdited() {
    const rows = root().querySelectorAll(".review-cand");
    const out = [];
    for (const row of rows) {
      const src = Number(row.getAttribute("data-src"));
      const obj = { _src: Number.isFinite(src) ? src : null };
      for (const el of row.querySelectorAll("[data-f]")) {
        const f = el.getAttribute("data-f");
        let v = el.value;
        if (f === "value") {
          const t = v.trim();
          try { v = JSON.parse(t); } catch { v = t; }  // primitives parse (5, "x"); objects too
        }
        obj[f] = v;
      }
      out.push(obj);
    }
    return out;
  }

  function deleteCard(srcIndex) {
    const card = root() && root().querySelector(`.review-cand[data-src="${srcIndex}"]`);
    if (card) card.remove();
  }

  // Grow a value textarea to fit its content (so a 12-month vector shows fully), capped so a huge
  // table still scrolls instead of taking the whole panel.
  function autosize(ta) {
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight + 2, Math.round(window.innerHeight * 0.4)) + "px";
  }

  // ── page viewer ──────────────────────────────────────────────────────────────
  function applyTransform() {
    const img = root().querySelector(".review-page-img");
    if (img) img.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    const label = root().querySelector(".review-page-label");
    if (label && pages.length) label.textContent = `${pages[pageIdx].month}  p.${pages[pageIdx].page}  (${pageIdx + 1}/${pages.length})`;
  }
  // Fit the loaded page to the viewport and center it. Called on each image load (natural size is
  // only known then); the fit scale also becomes the minimum zoom so the whole page is reachable.
  function fitPage() {
    const vp = root() && root().querySelector(".review-viewport");
    const img = root() && root().querySelector(".review-page-img");
    if (!vp || !img || !img.naturalWidth) return;
    const vw = vp.clientWidth, vh = vp.clientHeight;
    fitScale = Math.min(vw / img.naturalWidth, vh / img.naturalHeight) || 1;
    zoom = fitScale;
    panX = (vw - img.naturalWidth * zoom) / 2;
    panY = (vh - img.naturalHeight * zoom) / 2;
    applyTransform();
  }
  function setPage(i) {
    if (!pages.length) return;
    pageIdx = Math.max(0, Math.min(pages.length - 1, i));
    const img = root().querySelector(".review-page-img");
    if (img) {
      img.onload = fitPage;                  // fit once the new page's natural size is known
      img.src = `/api/source/${pages[pageIdx].month}/page/${pages[pageIdx].page}.png`;
      if (img.complete && img.naturalWidth) fitPage();  // cached image: onload may not refire
    }
  }
  function wireViewer() {
    const vp = root().querySelector(".review-viewport");
    if (!vp) return;
    vp.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = vp.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      const factor = Math.exp(-e.deltaY * 0.0015);          // trackpad pinch sends ctrlKey+wheel
      const next = Math.max(fitScale, Math.min(8, zoom * factor));  // can't zoom past full-page
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
    pageIdx = 0; zoom = 1; panX = 0; panY = 0; fitScale = 1;
    const task = tasksRef.find((t) => t.task_id === activeTaskId);
    const title = task ? `R${esc(task.round_num)} / ${esc(task.question_id)}` : esc(activeTaskId);
    const kindLabel = { lookup: "External Lookup", figure: "Visual QA", verify_extract: "Extract Validation" }[review.kind] || review.kind;
    const ctx = contextLine(review);
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
            ${ctx ? `<div class="review-context">${esc(ctx)}</div>` : ""}
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
    el.querySelectorAll(".review-value").forEach(autosize);  // fit each value box to its content
  }

  // A one-line "what this value is about" hint from the branch identity, so terse per-value
  // descriptions still carry the question's context.
  function contextLine(review) {
    const b = (review.guidance || {}).branch || {};
    if (b.kind === "lookup_external") return b.target ? `Lookup: ${b.target}` : "";
    if (b.key) return `Looking for: ${b.key}${b.period ? ` · period ${b.period}` : ""}`;
    return "";
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
    submit, acceptAsIs, requestExit, deleteCard, autosize,
    prevPage: () => setPage(pageIdx - 1),
    nextPage: () => setPage(pageIdx + 1),
  };
})();
window.ReviewOverlay = ReviewOverlay;
