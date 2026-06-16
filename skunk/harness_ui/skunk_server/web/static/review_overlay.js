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

window.apiJson = async function apiJson(url, body, options = {}) {
  const method = options.method || "POST";
  const headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
  const fetchOptions = Object.assign({}, options, { method, headers });
  if (body !== null && body !== undefined) fetchOptions.body = JSON.stringify(body);
  const resp = await fetch(url, fetchOptions);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || !data.ok) throw new Error(data.error || `Request failed (${resp.status})`);
  return data;
};

const LOCK_HEARTBEAT_MS = 7000;  // re-acquire (refresh the lease) well within the backend TTL

const ReviewOverlay = (function () {
  let activeTaskId = null;
  let reviews = [];            // open reviews for the active task (from the latest snapshot)
  let index = 0;               // which review we're showing
  let tasksRef = [];           // latest task snapshots (kept in sync by syncTasks)
  let submitting = false;
  let heartbeatTimer = null;   // setInterval handle refreshing the review lock while open

  // Natural-language refine state. `lastRefining` tracks the current review's `refining` flag
  // across status snapshots so we can detect when a backend refine starts (show spinner) or
  // finishes (re-render the cards with the LLM's result). `preRefineCandidates` is the displayed
  // state captured at Apply time, used to power the one-click Undo after a refine lands.
  let lastRefining = false;
  let preRefineCandidates = null;

  // Baseline for "did the cards change?" on a card-based review (extract/lookup). Captured once
  // per review (the original model extraction) so a single Submit can resolve empty (accept-as-is,
  // no recompute) when nothing changed, or send the payload (recompute) when it did. Snapshotting
  // via collectEdited() keeps the baseline and the submit-time read on the identical normalization.
  let baselineSig = null;
  let baselineReviewId = null;

  // viewer state
  let pages = [];              // [{month, page}] parsed from the review's source_docs
  let pageValues = [];         // [{month,page,values:[desc,...]}] from guidance — value(s) per page
  let pageIdx = 0;
  let zoom = 1, panX = 0, panY = 0;
  let fitScale = 1;            // scale that fits the whole page in the viewport (and the min zoom)

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));

  // Order reviews within a question by kind: external lookup, then visual QA, then extract —
  // so auto-advancing walks them in the same priority the task list uses.
  const KIND_RANK = { lookup: 0, figure: 1, verify_extract: 2, replan_approval: 3 };
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
      await apiJson(`/api/reviews/lock/${encodeURIComponent(taskId)}`, { client_id: CLIENT_ID });
      return true;
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
    apiJson(`/api/reviews/unlock/${encodeURIComponent(taskId)}`, { client_id: CLIENT_ID }, { keepalive: true })
      .catch(() => {});
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

  // `kind` (optional) jumps straight to that review kind (a card's review-kind tag); falls back
  // to the first review when absent (the detail overlay's Review button).
  async function open(taskId, kind) {
    const task = tasksRef.find((t) => t.task_id === taskId);
    if (!task || !(task.reviews || []).length) return;
    // Acquire the per-UID lock BEFORE showing anything, so two operators can't open the same
    // question at once. If it was grabbed between render and click, bail with a notice.
    const held = await acquireLock(taskId);
    if (!held) { alert("This question is being reviewed by someone else."); return; }
    activeTaskId = taskId;
    reviews = sortReviews(task.reviews);
    index = kind ? Math.max(0, reviews.findIndex((r) => r.kind === kind)) : 0;
    preRefineCandidates = null;
    baselineReviewId = null;
    lastRefining = !!(current() && current().refining);  // so a reopen mid-refine shows the spinner
    startHeartbeat();
    window.addEventListener("resize", onViewerResize);  // re-fit the page on window/zoom change
    render();
  }

  function close() {
    const taskId = activeTaskId;
    stopHeartbeat();
    window.removeEventListener("resize", onViewerResize);
    activeTaskId = null;
    reviews = [];
    index = 0;
    const el = root();
    if (el) el.hidden = true;
    releaseLock(taskId);  // free the lock for other operators (idempotent if we didn't hold it)
  }

  // Re-fit the current page when the window resizes or the browser zoom changes, so the operator
  // never has to manually zoom per screen. rAF-coalesced via scheduleFit; resets pan/zoom (the
  // intent on a viewport-size change).
  function onViewerResize() { if (isOpen() && pages.length) scheduleFit(); }

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
    // A backend refine flipped state: re-render (the refine deliberately replaces the cards).
    // Going busy → show the spinner; busy → done → render the LLM's revised candidates + a toast
    // with Undo. Steady state (no change) preserves the live DOM so manual edits aren't wiped.
    const nowRefining = !!(reviews[index] && reviews[index].refining);
    if (nowRefining !== lastRefining) {
      const completed = lastRefining && !nowRefining;
      lastRefining = nowRefining;
      render();
      if (completed) showRefineToast();
      return;
    }
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
  // row_name, col_name and machine provenance are preserved from the original extraction. The
  // card is self-contained: `data-full` carries the whole source candidate (incl. its `_src`,
  // the cached-entry index the recompute overlays onto) so re-renders / undo / refine never have
  // to map cards back to an array. `data-src` mirrors `_src` for the resolve path ("" = a value
  // the human/LLM added, with no cached entry). `i` is the card's render position; `_src` rides
  // along from the candidate so it survives a refine that reorders/adds/drops entries.
  function candidateRow(c, i) {
    const valueStr = (c.value && typeof c.value === "object") ? JSON.stringify(c.value, null, 2) : String(c.value ?? "");
    const kind = c.kind || "scalar";
    const meta = kind === "scalar" ? "" : `<span class="review-cand-kind">${esc(kind)}${c.index_name ? " · " + esc(c.index_name) : ""}</span>`;
    // A value with no corpus provenance is an external lookup, not a corpus extract. Badge it
    // (lookup vs extract) so the human prioritizes verifying it against its cited source rather
    // than a Bulletin page — they live in the same review, and extract cards still get their PDFs.
    const isLookup = !!c.external;
    const typeBadge = isLookup
      ? `<span class="review-cand-type is-lookup" title="External lookup — verify against the cited source, not a Bulletin page">lookup</span>`
      : `<span class="review-cand-type is-extract">extract</span>`;
    // For a lookup, a comment-style annotation on the JSON: what was looked up (target) and where
    // it came from (src). It sits just above the value box so it reads as an inline JSON comment,
    // but is NOT inside the editable payload — so it never breaks JSON parsing on submit.
    const extNote = isLookup
      ? `<div class="review-cand-extnote">// external lookup — target: ${esc(c.retrieve_key || c.description || "—")}${c.source ? `, src: ${esc(c.source)}` : ", src: (unattributed)"}</div>`
      : "";
    const hasSrc = c && Object.prototype.hasOwnProperty.call(c, "_src");
    const srcVal = hasSrc ? c._src : i;                       // null stays null (an added value)
    const srcAttr = (srcVal === null || srcVal === undefined) ? "" : String(srcVal);
    const full = Object.assign({}, c, { _src: (srcVal === undefined ? i : srcVal) });
    return `<div class="review-cand ${isLookup ? "is-lookup" : ""}" data-src="${srcAttr}" data-full="${esc(JSON.stringify(full))}">
      <div class="review-cand-head">
        ${typeBadge}
        ${meta}
        <button class="review-cand-del" title="Remove this value" onclick="ReviewOverlay.deleteCard(this)">✕</button>
      </div>
      <label class="review-field"><span>description</span>
        <input type="text" data-f="description" value="${esc(c.description ?? "")}"></label>
      <label class="review-field"><span>unit</span>
        <input type="text" data-f="unit" value="${esc(c.unit ?? "")}"></label>
      ${extNote}
      <label class="review-field"><span>value</span>
        <textarea data-f="value" class="review-value" oninput="ReviewOverlay.autosize(this)">${esc(valueStr)}</textarea></label>
      ${c.notes ? `<div class="review-cand-notes"><span class="review-notes-label">notes</span>${esc(c.notes)}</div>` : ""}
    </div>`;
  }

  // Build the editable card list HTML from a candidates array (one card per value, a single empty
  // scalar when there are none). Shared by the main render and the undo path.
  function renderCards(cands) {
    return (cands && cands.length)
      ? cands.map(candidateRow).join("")
      : candidateRow({ description: "", value: "", kind: "scalar" }, 0);
  }

  // The edited value of one card: its `value` box parsed (primitives like 5/"x" and objects),
  // description + unit as text.
  function readEditedFields(row) {
    const out = {};
    for (const el of row.querySelectorAll("[data-f]")) {
      const f = el.getAttribute("data-f");
      let v = el.value;
      if (f === "value") {
        const t = v.trim();
        try { v = JSON.parse(t); } catch { v = t; }  // primitives parse (5, "x"); objects too
      }
      out[f] = v;
    }
    return out;
  }

  // Read the surviving value boxes back into source-indexed override items
  // ({_src, description, unit, value}); deleted boxes are simply absent. The recompute overlays
  // the edited fields onto the cached entry at `_src` (`""` data-src → null → a fresh value).
  function collectEdited() {
    const rows = root().querySelectorAll(".review-cand");
    const out = [];
    for (const row of rows) {
      const raw = row.getAttribute("data-src");
      const src = (raw === "" || raw === null) ? NaN : Number(raw);
      const obj = Object.assign({ _src: Number.isFinite(src) ? src : null }, readEditedFields(row));
      out.push(obj);
    }
    return out;
  }

  // Read the surviving cards back as FULL candidate objects (the whole source candidate from
  // `data-full`, with the human's edits overlaid). This is what the natural-language refine sends
  // so the LLM sees the complete JSONs (kind/shape/_src), not just the editable subset.
  function collectFullCandidates() {
    const rows = root().querySelectorAll(".review-cand");
    const out = [];
    for (const row of rows) {
      let full = {};
      try { full = JSON.parse(row.getAttribute("data-full") || "{}"); } catch { full = {}; }
      out.push(Object.assign(full, readEditedFields(row)));
    }
    return out;
  }

  function deleteCard(btn) {
    const card = btn && btn.closest && btn.closest(".review-cand");
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
  // Fit the loaded page to the viewport and center it; the fit scale also becomes the minimum
  // zoom so the whole page is always reachable. Returns false (so scheduleFit retries) when the
  // viewport or image isn't measurable yet — the prior `|| 1` fallback on a 0-sized viewport is
  // what produced the blown-up "ridiculously zoomed" open.
  function fitPage() {
    const vp = root() && root().querySelector(".review-viewport");
    const img = root() && root().querySelector(".review-page-img");
    if (!vp || !img) return false;
    const vw = vp.clientWidth, vh = vp.clientHeight;
    if (!vw || !vh || !img.naturalWidth || !img.naturalHeight) return false;  // not laid out yet
    fitScale = Math.min(vw / img.naturalWidth, vh / img.naturalHeight);
    zoom = fitScale;
    panX = (vw - img.naturalWidth * zoom) / 2;
    panY = (vh - img.naturalHeight * zoom) / 2;
    applyTransform();
    return true;
  }
  // Fit on the next frame (after the panel + image have laid out), retrying a few frames until the
  // viewport/image are measurable. Used on open, page change, and window resize / browser-zoom.
  function scheduleFit(tries) {
    tries = tries == null ? 12 : tries;
    requestAnimationFrame(() => { if (!fitPage() && tries > 0) scheduleFit(tries - 1); });
  }
  function setPage(i) {
    if (!pages.length) return;
    pageIdx = Math.max(0, Math.min(pages.length - 1, i));
    const img = root().querySelector(".review-page-img");
    if (img) {
      img.onload = () => scheduleFit();      // fit once the new page's natural size is known
      img.src = `/api/source/${pages[pageIdx].month}/page/${pages[pageIdx].page}.png`;
      scheduleFit();                          // cached image / already-laid-out: fit post-layout
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

  // The action controls, rendered both above and below the editor. For verify_extract this
  // includes the natural-language feedback box (the LLM revises all candidate JSONs at once;
  // `review.refining` drives the busy/spinner state). A single Submit; submit() decides whether
  // to resolve empty (accept-as-is, no recompute) or send the payload. Uses CLASSES (not ids)
  // since it appears twice — refine()/submit()/showError operate across both copies.
  function controlsBar(review) {
    if (review.kind === "replan_approval") {
      // Single Submit: empty feedback → resolve empty (execute the proposed plan as-is); feedback
      // typed → submit it (the orchestrator re-runs the replan with it, then executes).
      return `<div class="review-controls"><div class="review-actions">
        <button class="review-submit primary" onclick="ReviewOverlay.submit()" title="Empty feedback executes the proposed plan as-is; feedback re-plans">Submit</button>
        <span class="review-error"></span>
      </div></div>`;
    }
    const isExtract = review.kind === "verify_extract";
    const busy = !!review.refining;
    const refine = isExtract ? `
      <div class="review-refine">
        <div class="review-section-label">Revise with natural-language feedback</div>
        <textarea class="review-refine-input" rows="3" ${busy ? "disabled" : ""}
          placeholder="Describe a correction for the LLM to apply across the values — e.g. 'every value is the next year's figure; shift them all back one year.'"></textarea>
        <div class="review-refine-actions">
          <button class="review-refine-btn" onclick="ReviewOverlay.refine()" ${busy ? "disabled" : ""}>${busy ? "✨ Updating extraction…" : "Apply feedback"}</button>
          <span class="review-refine-status ${busy ? "is-busy" : ""}">${busy ? "Working… you can leave and come back." : ""}</span>
        </div>
      </div>` : "";
    return `<div class="review-controls">
      ${refine}
      <div class="review-actions">
        <button class="review-submit primary" onclick="ReviewOverlay.submit()" title="Unchanged values are kept as-is; edits trigger a recompute">Submit</button>
        <span class="review-error"></span>
      </div>
    </div>`;
  }

  // ── replan approval (read-only context + steer feedback) ─────────────────────
  // The replanner already ran; the human sees the data-prep output, the previous plan, the
  // compute MissingData, and the PROPOSED plan, then Approves (resolve empty) or Rejects with
  // feedback (resolve text → the orchestrator re-runs the replan with it injected).
  function valueShort(val) {
    if (val == null) return "";
    if (typeof val === "object") {
      try { const s = JSON.stringify(val); return s.length > 140 ? s.slice(0, 137) + "…" : s; }
      catch (e) { return String(val); }
    }
    return String(val);
  }
  function planBranchesHtml(branches) {
    if (!Array.isArray(branches) || !branches.length) return `<div class="rp-empty">— none —</div>`;
    return `<ol class="rp-plan">` + branches.map((b) => {
      const id = b.branch_id != null ? `<span class="rp-bid">#${esc(b.branch_id)}</span>` : "";
      if (b.kind === "lookup_external") {
        return `<li>${id}<span class="rp-kind rp-lookup">lookup</span> <span class="rp-key">${esc(b.target || "")}</span>${b.src ? ` <span class="rp-src">src: ${esc(b.src)}</span>` : ""}</li>`;
      }
      const period = b.period ? ` <span class="rp-period">· ${esc(b.period)}</span>` : "";
      return `<li>${id}<span class="rp-kind rp-retrieve">retrieve</span> <span class="rp-key">${esc(b.key || "")}</span>${period}</li>`;
    }).join("") + `</ol>`;
  }
  function poolValuesHtml(values) {
    if (!Array.isArray(values) || !values.length) return `<div class="rp-empty">— no values —</div>`;
    return `<ul class="rp-pool">` + values.map((v) => {
      const prov = v.bulletin ? ` <span class="rp-prov">${esc(v.bulletin)}${Array.isArray(v.pages) && v.pages.length ? ` p${esc(v.pages.join(","))}` : ""}</span>` : "";
      const unit = v.unit ? ` <span class="rp-unit">${esc(v.unit)}</span>` : "";
      return `<li><span class="rp-desc">${esc(v.description || "(value)")}</span> <span class="rp-val">${esc(valueShort(v.value))}</span>${unit}${prov}</li>`;
    }).join("") + `</ul>`;
  }
  function replanApprovalHtml(review) {
    const g = review.guidance || {};
    const missing = (Array.isArray(g.missing) ? g.missing : []).filter(Boolean);
    return `
      <div class="rp-grid">
        <section class="rp-sec">
          <div class="review-section-label">Data-prep output (what compute read)</div>
          ${poolValuesHtml(g.data_prep_output)}
        </section>
        <section class="rp-sec">
          <div class="review-section-label">Why compute couldn't answer</div>
          <div class="rp-reason">${esc(g.reason || "")}</div>
          ${missing.length ? `<div class="rp-missing">Missing: ${esc(missing.join(", "))}</div>` : ""}
        </section>
        <section class="rp-sec">
          <div class="review-section-label">Previous plan</div>
          ${planBranchesHtml(g.previous_plan)}
        </section>
        <section class="rp-sec rp-proposed">
          <div class="review-section-label">Proposed new plan</div>
          ${planBranchesHtml(g.proposed_plan)}
        </section>
      </div>
      <div class="review-section-label">If the proposed plan is wrong, tell the replanner what to change (then Reject)</div>
      <textarea id="replanFeedbackInput" class="review-missing-input" rows="3"
        placeholder="e.g. keep the 1990 bond branch — don't drop it; read Table FD-1, not the summary."></textarea>`;
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
    const kindLabel = { lookup: "External Lookup", figure: "Visual QA", verify_extract: "Extract Validation", replan_approval: "Replan Approval" }[review.kind] || review.kind;
    // replan_approval: the proposed plan is already computed; the human approves it or rejects with
    // steer feedback. Read-only context + a feedback box, no candidates, custom controls.
    const isReplan = review.kind === "replan_approval";
    // figure (Visual QA): the model's chart read is unreliable, so the human authors the answer in a
    // pre-filled AnnotatedValue JSON template (description = retrieval target) instead of cards.
    const isFigure = review.kind === "figure";
    const ctx = contextLine(review);
    const cands = candidates(review);
    // Each kind picks its editor: replan_approval → read-only context + feedback; figure → JSON
    // template textarea; everything else → editable value cards (confirm/correct the model's read).
    let editor, editorLabel;
    if (isReplan) {
      editor = replanApprovalHtml(review);
      editorLabel = "";
    } else if (isFigure) {
      editor = `<textarea id="figureJsonInput" class="review-json-input" spellcheck="false" rows="8">${esc(figureTemplate(review))}</textarea>`;
      editorLabel = `<div class="review-section-label">AnnotatedValue — read the value(s) off the chart</div>`;
    } else {
      editor = renderCards(cands);
      editorLabel = "";
    }
    // The controls (NL-feedback box for verify_extract + Submit) are rendered BOTH above and below
    // the editor, so a long JSON list can be acted on without scrolling to the bottom. The editor
    // itself stays single in the middle.
    const topControls = isReplan ? "" : controlsBar(review);
    const bottomControls = controlsBar(review);
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
            ${topControls}
            ${editorLabel}
            <div class="review-editor">${editor}</div>
            ${bottomControls}
          </div>
          ${viewer}
        </div>
      </div>`;
    if (pages.length) { wireViewer(); setPage(0); }
    el.querySelectorAll(".review-value").forEach(autosize);  // fit each value box to its content
    // Capture the card baseline once per review (the first render = the original model extraction),
    // so it survives refine re-renders: refined cards then read as "changed" (→ recompute) while an
    // undo restores the cards back to the baseline (→ accept-as-is). See submit().
    if (!isReplan && !isFigure && review.review_id !== baselineReviewId) {
      baselineSig = JSON.stringify(collectEdited());
      baselineReviewId = review.review_id;
    }
  }

  // A one-line "what this value is about" hint from the branch identity, so terse per-value
  // descriptions still carry the question's context.
  function contextLine(review) {
    const g = review.guidance || {};
    if (review.kind === "replan_approval") {
      const miss = (Array.isArray(g.missing) ? g.missing : []).filter(Boolean);
      return miss.length ? `Missing: ${miss.join(", ")}` : "";
    }
    const b = g.branch || {};
    if (b.kind === "lookup_external") return b.target ? `Lookup: ${b.target}${b.src ? ` · requested source: ${b.src}` : ""}` : "";
    if (b.key) return `Looking for: ${b.key}${b.period ? ` · period ${b.period}` : ""}`;
    return "";
  }

  function showError(msg) {
    const r = root();
    if (r) r.querySelectorAll(".review-error").forEach((e) => { e.textContent = msg; });
  }

  async function post(review, response) {
    if (submitting) return;
    submitting = true;
    showError("");
    try {
      await apiJson(`/api/reviews/${encodeURIComponent(review.review_id)}/resolve`, {
        response,
        source_docs: review.source_docs || [],
        client_id: CLIENT_ID,
      });
      // One annotation per visit: return to the main screen (and release the lock) — no
      // auto-advance to the task's next open review. The SSE snapshot will confirm the resolve.
      submitting = false;
      close();
    } catch (err) {
      showError(String(err));
      submitting = false;
    }
  }

  // Single Submit per review: resolve empty (accept-as-is / approve, no recompute) when nothing
  // changed, otherwise send the payload (recompute / re-plan). Each kind defines "unchanged".
  function submit() {
    const review = current();
    if (!review) return;
    if (review.kind === "replan_approval") {
      // Empty feedback = approve (execute the proposed plan as-is); feedback = re-plan with it.
      const ta = root().querySelector("#replanFeedbackInput");
      const text = (ta ? ta.value : "").trim();
      post(review, text);
      return;
    }
    if (review.kind === "figure") {
      // An untouched (or empty) template = accept-as-is; an authored value = submit the JSON.
      // Validate it parses before sending so a typo surfaces here, not in a silent recompute failure.
      const ta = root().querySelector("#figureJsonInput");
      const text = (ta ? ta.value : "").trim();
      if (!text || text === figureTemplate(review).trim()) { post(review, ""); return; }
      try { JSON.parse(text); } catch (err) { showError("Not valid JSON: " + err.message); return; }
      post(review, text);
      return;
    }
    // Cards (extract/lookup): unchanged from the baseline → empty (keep the model's answer, no
    // recompute); any edit/add/delete/refine → send the payload to trigger a recompute.
    let edited;
    try { edited = collectEdited(); } catch (err) { showError(String(err)); return; }
    const sig = JSON.stringify(edited);
    post(review, sig === baselineSig ? "" : sig);
  }

  // ── natural-language refine ────────────────────────────────────────────────────
  // Send the displayed JSONs + the reviewer's feedback to the backend, which revises them with an
  // LLM (async). We don't close or block here: the status snapshot flips review.refining, and
  // syncTasks drives the spinner + the re-render with the result. preRefineCandidates is stashed
  // so the post-refine toast can offer a one-click Undo (purely client-side, off the live DOM).
  // The feedback box + Apply button appear twice (top and bottom). Read whichever copy the
  // reviewer typed in; disable both Apply buttons while the refine is in flight.
  function refineButtons() { return [...root().querySelectorAll(".review-refine-btn")]; }
  function setRefineBusy(busy) {
    refineButtons().forEach((b) => { b.disabled = busy; b.textContent = busy ? "✨ Updating extraction…" : "Apply feedback"; });
  }

  async function refine() {
    const review = current();
    if (!review || review.kind !== "verify_extract") return;
    const feedback = [...root().querySelectorAll(".review-refine-input")]
      .map((i) => i.value.trim()).find((v) => v) || "";
    if (!feedback) { showError("Enter feedback describing the correction."); return; }
    let candidatesPayload;
    try { candidatesPayload = collectFullCandidates(); } catch (err) { showError(String(err)); return; }
    preRefineCandidates = candidatesPayload;
    setRefineBusy(true);
    showError("");
    try {
      await apiJson(`/api/reviews/${encodeURIComponent(review.review_id)}/refine`, {
        feedback,
        candidates: candidatesPayload,
        client_id: CLIENT_ID,
      });
      // Success: the backend set review.refining=true and published; syncTasks takes over from
      // here (spinner now, revised cards + toast on completion). Nothing more to do.
    } catch (err) {
      showError(err.message || String(err));
      setRefineBusy(false);
    }
  }

  // Restore the pre-refine cards straight into the editor DOM (the submission source of truth),
  // so Undo survives later status snapshots without a server round-trip.
  function undoRefine() {
    if (!preRefineCandidates) return;
    const ed = root() && root().querySelector(".review-editor");
    if (ed) {
      ed.innerHTML = renderCards(preRefineCandidates);
      ed.querySelectorAll(".review-value").forEach(autosize);
    }
    const toast = root() && root().querySelector(".review-refine-toast");
    if (toast) toast.remove();
  }

  function showRefineToast() {
    const left = root() && root().querySelector(".review-left");
    if (!left) return;
    const existing = left.querySelector(".review-refine-toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "review-refine-toast";
    toast.innerHTML = `<span class="review-toast-msg">✨ Extraction updated from your feedback.</span>
      <button class="review-toast-undo" onclick="ReviewOverlay.undoRefine()">Undo</button>
      <button class="review-toast-dismiss" title="Dismiss" onclick="this.parentNode.remove()">✕</button>`;
    left.insertBefore(toast, left.firstChild);
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
    submit, requestExit, deleteCard, autosize,
    refine, undoRefine,
    prevPage: () => setPage(pageIdx - 1),
    nextPage: () => setPage(pageIdx + 1),
  };
})();
window.ReviewOverlay = ReviewOverlay;
