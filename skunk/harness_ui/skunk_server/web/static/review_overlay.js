"use strict";
// Immersive human-review overlay. Opened for a task that has open reviews; it covers the app
// with a dimmed backdrop (the round timer in the header stays visible behind it) and walks the
// operator through that task's reviews one at a time. Each review is one of three kinds:
//   lookup         -> instruction + a value field (confirm/correct an external lookup)
//   figure         -> page viewer + a pre-filled AnnotatedValue JSON template (read it off the chart)
//   verify_extract -> page viewer + an editable table/list of the model's extracted values
// Submitting POSTs the edited values to /api/reviews/<id>/resolve; the overlay then RETURNS to
// the main UI (one annotation per visit — no auto-advance to the task's next review). "X" exits
// without committing. The status SSE stream is the source of truth for which reviews are open;
// `syncTasks` re-reads it each tick so a resolved/cancelled review drops out live.
//
// Per-UID lock: opening the overlay acquires this task's review lock in the backend
// (/api/reviews/lock/<task_id>) so two operators can't annotate the same question at once. While
// open we heartbeat to keep the lease alive; closing (submit / accept / X / backdrop / tab-close)
// releases it. A holder that vanishes is freed by the backend's TTL sweep.

// Stable per-browser-session id, used to hold/release the review lock. Defined here (this file is
// `defer`-loaded before client.js) and exposed so client.js can tell "locked by me" from "locked
// by someone else".
const CLIENT_ID = (function () {
  try {
    let id = sessionStorage.getItem("skunkClientId");
    if (!id) {
      id = (window.crypto && crypto.randomUUID)
        ? crypto.randomUUID()
        : `c-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
      sessionStorage.setItem("skunkClientId", id);
    }
    return id;
  } catch (e) {
    return `c-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
  }
})();
window.CLIENT_ID = CLIENT_ID;

const LOCK_HEARTBEAT_MS = 7000;  // re-acquire (refresh the lease) well within the backend TTL

const ReviewOverlay = (function () {
  let activeTaskId = null;
  let reviews = [];            // open reviews for the active task (from the latest snapshot)
  let index = 0;               // which review we're showing
  let tasksRef = [];           // latest task snapshots (kept in sync by syncTasks)
  let submitting = false;
  let heartbeatTimer = null;   // setInterval handle refreshing the review lock while open

  // viewer state
  let pages = [];              // [{month, page}] parsed from the review's source_docs
  let pageValues = [];         // [{month,page,values:[desc,...]}] from guidance — value(s) per page
  let pageIdx = 0;
  let zoom = 1, panX = 0, panY = 0;
  let fitScale = 1;            // scale that fits the whole page in the viewport (and the min zoom)

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));

  // Order reviews within a question by kind: external lookup, then visual QA, then extract —
  // so auto-advancing walks them in the same priority the task list uses.
  const KIND_RANK = { lookup: 0, figure: 1, verify_extract: 2 };
  const sortReviews = (rs) => rs.slice().sort((a, b) => (KIND_RANK[a.kind] ?? 3) - (KIND_RANK[b.kind] ?? 3));

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

  // ── review lock ────────────────────────────────────────────────────────────────
  // The backend is the source of truth for the lock; the greyed-out buttons in the task list are
  // only a hint. Re-acquiring with the same client_id refreshes the lease, so acquire IS the
  // heartbeat. Returns true iff we hold the lock afterward.
  async function acquireLock(taskId) {
    try {
      const resp = await fetch(`/api/reviews/lock/${encodeURIComponent(taskId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: CLIENT_ID }),
      });
      const data = await resp.json().catch(() => ({}));
      return !!(resp.ok && data.ok);
    } catch (err) {
      return false;
    }
  }

  function releaseLock(taskId) {
    if (!taskId) return;
    const body = JSON.stringify({ client_id: CLIENT_ID });
    // Prefer sendBeacon so a tab-close / navigation still releases the lock; fall back to a
    // keepalive fetch where sendBeacon is unavailable.
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon(
          `/api/reviews/unlock/${encodeURIComponent(taskId)}`,
          new Blob([body], { type: "application/json" }),
        );
        return;
      }
    } catch (err) { /* fall through to fetch */ }
    fetch(`/api/reviews/unlock/${encodeURIComponent(taskId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => {});
  }

  function startHeartbeat() {
    stopHeartbeat();
    heartbeatTimer = setInterval(async () => {
      if (!isOpen()) return;
      const held = await acquireLock(activeTaskId);
      if (!held) {
        // Lost the lease (expired then taken by someone else) — exit to the main screen.
        showError("");
        close();
        alert("Your review lock expired and this question was handed to another reviewer.");
      }
    }, LOCK_HEARTBEAT_MS);
  }

  function stopHeartbeat() {
    if (heartbeatTimer !== null) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
  }

  async function open(taskId) {
    const task = tasksRef.find((t) => t.task_id === taskId);
    if (!task || !(task.reviews || []).length) return;
    // Acquire the per-UID lock BEFORE showing anything, so two operators can't open the same
    // question at once. If it was grabbed between render and click, bail with a notice.
    const held = await acquireLock(taskId);
    if (!held) { alert("This question is being reviewed by someone else."); return; }
    activeTaskId = taskId;
    reviews = sortReviews(task.reviews);
    index = 0;
    startHeartbeat();
    render();
  }

  function close() {
    const taskId = activeTaskId;
    stopHeartbeat();
    activeTaskId = null;
    reviews = [];
    index = 0;
    const el = root();
    if (el) el.hidden = true;
    releaseLock(taskId);  // free the lock for other operators (idempotent if we didn't hold it)
  }

  // Re-read the live snapshot WITHOUT clobbering the operator's work. Status snapshots arrive
  // every tick, so we must NOT rebuild the DOM (which would wipe in-progress edits and reset the
  // page viewer's pan/zoom) while the operator is on a review that's still open. We exit to the
  // main screen if we lost the lock, or if the CURRENT review disappeared externally (resolved
  // elsewhere / cancelled at round close) — we never auto-advance to another review.
  function syncTasks(tasks) {
    tasksRef = tasks || [];
    if (!isOpen()) return;
    const task = tasksRef.find((t) => t.task_id === activeTaskId);
    // Lost the lock to another reviewer (lease expired and was reclaimed) → back to main. The
    // heartbeat is the primary detector; this is a belt-and-suspenders check off the snapshot.
    if (task && task.locked_by && task.locked_by !== CLIENT_ID) { close(); return; }
    const open = task ? (task.reviews || []) : [];
    const cur = reviews[index];
    const curStillOpen = cur && open.some((r) => r.review_id === cur.review_id);
    if (!curStillOpen) { close(); return; }  // current review gone → return to main screen
    // Same review still in progress — keep the live DOM (edits + viewer) intact, just keep our
    // index pointed at it and refresh the "review N of M" counter text in place.
    reviews = sortReviews(open);
    index = reviews.findIndex((r) => r.review_id === cur.review_id);
    updateCount();
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

  // The figure (Visual QA) editor: a pre-filled AnnotatedValue JSON template the human edits while
  // reading the chart, built server-side with `description` set to the retrieval target. The
  // model's chart read is unreliable, so we DON'T anchor the human with confirm/correct cards here.
  // Fall back to a minimal scalar template keyed on the branch's retrieval target if absent.
  function figureTemplate(review) {
    const g = review.guidance || {};
    if (typeof g.value_template === "string") return g.value_template;
    const target = (g.branch && g.branch.key) || "";
    return JSON.stringify({ description: target, value: "", unit: "" }, null, 2);
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
      ${c.source ? `<div class="review-cand-source"><span class="review-notes-label">source</span>${esc(c.source)}</div>` : ""}
      ${c.notes ? `<div class="review-cand-notes"><span class="review-notes-label">notes</span>${esc(c.notes)}</div>` : ""}
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
    if (label && pages.length) {
      const pg = pages[pageIdx];
      const pv = pageValues.find((v) => v.month === pg.month && Number(v.page) === Number(pg.page));
      const vals = pv && (pv.values || []).length ? `  —  ${pv.values.join(", ")}` : "";
      label.textContent = `Treasury Bulletin ${pg.month}  ·  p.${pg.page}  (${pageIdx + 1}/${pages.length})${vals}`;
    }
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
    pageValues = (review.guidance && review.guidance.page_values) || [];
    pageIdx = 0; zoom = 1; panX = 0; panY = 0; fitScale = 1;
    const task = tasksRef.find((t) => t.task_id === activeTaskId);
    const title = task ? `R${esc(task.round_num)} / ${esc(task.question_id)}` : esc(activeTaskId);
    const kindLabel = { lookup: "External Lookup", figure: "Visual QA", verify_extract: "Extract Validation", missing_data: "Missing Data" }[review.kind] || review.kind;
    // missing_data is a "provide the value(s) the run couldn't find" request — not a validation of
    // model candidates — so it has no candidates and gets its own framing (header + no accept-as-is).
    const isMissing = review.kind === "missing_data";
    // figure (Visual QA): the model's chart read is unreliable, so the human authors the answer in a
    // pre-filled AnnotatedValue JSON template (description = retrieval target) instead of cards.
    const isFigure = review.kind === "figure";
    const ctx = contextLine(review);
    const cands = candidates(review);
    // Each kind picks its editor: missing_data → free-form replan instruction; figure → JSON
    // template textarea; everything else → editable value cards (confirm/correct the model's read).
    let editor, editorLabel;
    if (isMissing) {
      editor = `<textarea id="missingDataInput" class="review-missing-input" rows="4"
           placeholder="Tell the replanner what to do differently — e.g. which series/table/bulletin to read, how to interpret the question, or where the value actually lives."></textarea>`;
      editorLabel = `<div class="review-section-label">Instruction for the replanner</div>`;
    } else if (isFigure) {
      editor = `<textarea id="figureJsonInput" class="review-json-input" spellcheck="false" rows="8">${esc(figureTemplate(review))}</textarea>`;
      editorLabel = `<div class="review-section-label">AnnotatedValue — read the value(s) off the chart</div>`;
    } else {
      editor = cands.length
        ? cands.map(candidateRow).join("")
        : candidateRow({ description: "", value: "", kind: "scalar" }, 0);
      editorLabel = "";
    }
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
            <span class="review-kind kind-${esc(review.kind)}">${esc(kindLabel)}</span>
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
            ${editorLabel}
            <div class="review-editor">${editor}</div>
            <div class="review-actions">
              <button class="review-submit primary" onclick="ReviewOverlay.submit()">Submit</button>
              ${isMissing ? "" : `<button class="review-accept" onclick="ReviewOverlay.acceptAsIs()" title="Keep the model's answer unchanged">Accept as-is</button>`}
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
    const g = review.guidance || {};
    if (review.kind === "missing_data") {
      const miss = (Array.isArray(g.missing) ? g.missing : []).filter(Boolean);
      return miss.length ? `Missing: ${miss.join(", ")}` : "";
    }
    const b = g.branch || {};
    if (b.kind === "lookup_external") return b.target ? `Lookup: ${b.target}${b.src ? ` · requested source: ${b.src}` : ""}` : "";
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
      // One annotation per visit: return to the main screen (and release the lock) — no
      // auto-advance to the task's next open review. The SSE snapshot will confirm the resolve.
      submitting = false;
      close();
    } catch (err) {
      showError(String(err));
      submitting = false;
    }
  }

  function submit() {
    const review = current();
    if (!review) return;
    if (review.kind === "missing_data") {
      // Free-form instruction → sent as the raw response; the server injects it into the replan prompt.
      const ta = root().querySelector("#missingDataInput");
      const text = (ta ? ta.value : "").trim();
      if (!text) { showError("Enter an instruction for the replanner."); return; }
      post(review, text);
      return;
    }
    if (review.kind === "figure") {
      // The edited AnnotatedValue JSON template → sent raw (already source-indexed for recompute).
      // Validate it parses before sending so a typo surfaces here, not in a silent recompute failure.
      const ta = root().querySelector("#figureJsonInput");
      const text = (ta ? ta.value : "").trim();
      if (!text) { showError("Fill in the value(s) read off the chart."); return; }
      try { JSON.parse(text); } catch (err) { showError("Not valid JSON: " + err.message); return; }
      post(review, text);
      return;
    }
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

  // Release the lock if the tab is closed / navigated away with the overlay still open, so the
  // task doesn't stay locked until the backend TTL sweep. `pagehide` fires on bfcache too.
  function releaseOnExit() { if (isOpen()) releaseLock(activeTaskId); }
  window.addEventListener("pagehide", releaseOnExit);
  window.addEventListener("beforeunload", releaseOnExit);

  return {
    open, close, isOpen, syncTasks,
    submit, acceptAsIs, requestExit, deleteCard, autosize,
    prevPage: () => setPage(pageIdx - 1),
    nextPage: () => setPage(pageIdx + 1),
  };
})();
window.ReviewOverlay = ReviewOverlay;
