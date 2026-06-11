/* ----------------------------------------------------------------------------
   DopTrack results gallery — viewer logic
   Data is provided by data.js as window.RESULTS (works on file:// and Pages).
---------------------------------------------------------------------------- */
(function () {
  "use strict";

  const DATA = window.RESULTS;
  if (!DATA) { document.body.innerHTML = "<p style='padding:40px'>data.js failed to load.</p>"; return; }

  // satellite accent colours
  const SAT_COLOURS = {
    "Delfi-C3":   "#5ec8f2",
    "FUNcube-1":  "#7c8cf8",
    "Nayif-1":    "#5fe3a1",
    "Delfi-n3Xt": "#f2c14e",
  };
  const colourFor = s => SAT_COLOURS[s] || "#9fb0c0";

  // flatten passes, tagging each with its satellite
  const ALL = [];
  DATA.satellites.forEach(sat => {
    sat.passes.forEach(p => ALL.push(Object.assign({ satellite: sat.name }, p)));
  });

  // ── SNR colour scale (dB) ────────────────────────────────────────────────
  function snrColour(v) {
    if (v == null || isNaN(v)) return "#6b7c8d";
    if (v >= 8)  return "#5fe3a1";   // strong
    if (v >= 3)  return "#9ad26b";
    if (v >= 0)  return "#f2c14e";   // marginal
    return "#f2776e";                // below noise
  }

  // ── state ────────────────────────────────────────────────────────────────
  const state = { sat: "all", q: "", sort: "date_desc" };
  let view = [];   // current filtered + sorted list
  let modalIdx = -1;

  const $ = sel => document.querySelector(sel);
  const fmt = (v, d = 1) => (v == null || isNaN(v)) ? "–" : Number(v).toFixed(d);

  // ── header meta ──────────────────────────────────────────────────────────
  function renderHead() {
    const c = DATA.config || {};
    $("#headMeta").innerHTML = [
      `<span class="chip"><b>${DATA.n_passes}</b> passes</span>`,
      `<span class="chip"><b>${DATA.n_satellites}</b> satellites</span>`,
      `<span class="chip">smoother <b>${c.smoother || "?"}</b></span>`,
      `<span class="chip">N <b>${c.N || "?"}</b></span>`,
      `<span class="chip">f<sub>s</sub> <b>${(c.fs_hz/1000)||"?"}k</b></span>`,
    ].join("");
    $("#footText").textContent =
      `Generated ${DATA.generated} · classical spectrogram estimator · ${DATA.n_passes} passes`;
  }

  // ── overview cards ─────────────────────────────────────────────────────────
  function renderOverview() {
    const host = $("#overview");
    host.innerHTML = "";
    DATA.satellites.forEach(sat => {
      const c = colourFor(sat.name);
      const st = sat.stats || {};
      const peak = st.peak_snr_db || {};
      const interp = st.interp_pass_pct, bwt = st.bw_test_pass_pct;
      const dur = st.duration_s || {};

      const el = document.createElement("div");
      el.className = "ov-card";
      el.style.setProperty("--c", c);
      el.dataset.sat = sat.name;
      el.innerHTML = `
        <div class="ov-name"><span class="ov-dot"></span>${sat.name}
          <span class="ov-count">${sat.count} passes</span></div>
        <div class="ov-metrics">
          <div class="ov-metric"><div class="v" style="color:${snrColour(peak.median)}">${fmt(peak.median)}</div><div class="l">med peak SNR</div></div>
          <div class="ov-metric"><div class="v">${fmt(peak.max)}</div><div class="l">best dB</div></div>
          <div class="ov-metric"><div class="v">${fmt(dur.median,0)}s</div><div class="l">med dur</div></div>
        </div>
        <div class="ov-bars">
          ${bar("interp ok", interp)}
          ${bar("bw test", bwt)}
        </div>`;
      el.addEventListener("click", () => {
        state.sat = (state.sat === sat.name) ? "all" : sat.name;
        syncFilters(); render();
      });
      host.appendChild(el);
    });
  }
  function bar(label, pct) {
    if (pct == null) pct = 0;
    const col = pct >= 66 ? "#5fe3a1" : pct >= 33 ? "#f2c14e" : "#f2776e";
    return `<div class="ov-bar-row"><span class="lab">${label}</span>
      <span class="ov-bar"><span style="width:${pct}%;background:${col}"></span></span>
      <span class="pct">${pct == null ? "–" : pct + "%"}</span></div>`;
  }

  // ── satellite filter pills ─────────────────────────────────────────────────
  function renderFilters() {
    const host = $("#satFilters");
    const pills = [`<button class="pill" data-sat="all">All
        <span class="n">${ALL.length}</span></button>`];
    DATA.satellites.forEach(s => {
      pills.push(`<button class="pill" data-sat="${s.name}" style="--c:${colourFor(s.name)}">
        <span class="dot"></span>${s.name}<span class="n">${s.count}</span></button>`);
    });
    host.innerHTML = pills.join("");
    host.querySelectorAll(".pill").forEach(p =>
      p.addEventListener("click", () => { state.sat = p.dataset.sat; syncFilters(); render(); }));
    syncFilters();
  }
  function syncFilters() {
    document.querySelectorAll(".pill").forEach(p =>
      p.classList.toggle("active", p.dataset.sat === state.sat));
    document.querySelectorAll(".ov-card").forEach(c =>
      c.classList.toggle("active", c.dataset.sat === state.sat));
  }

  // ── filter + sort ──────────────────────────────────────────────────────────
  function compute() {
    let list = ALL.slice();
    if (state.sat !== "all") list = list.filter(p => p.satellite === state.sat);
    if (state.q) {
      const q = state.q.toLowerCase();
      list = list.filter(p => p.name.toLowerCase().includes(q));
    }
    const by = {
      date_desc:  (a, b) => (b.date || "").localeCompare(a.date || ""),
      date_asc:   (a, b) => (a.date || "").localeCompare(b.date || ""),
      peak_desc:  (a, b) => (b.peak_snr_db ?? -1e9) - (a.peak_snr_db ?? -1e9),
      peak_asc:   (a, b) => (a.peak_snr_db ??  1e9) - (b.peak_snr_db ??  1e9),
      median_desc:(a, b) => (b.median_snr_db ?? -1e9) - (a.median_snr_db ?? -1e9),
      median_asc: (a, b) => (a.median_snr_db ??  1e9) - (b.median_snr_db ??  1e9),
      dur_desc:   (a, b) => (b.duration_s ?? 0) - (a.duration_s ?? 0),
      name_asc:   (a, b) => a.name.localeCompare(b.name),
    }[state.sort];
    list.sort(by);
    return list;
  }

  // ── grid ───────────────────────────────────────────────────────────────────
  function render() {
    view = compute();
    const grid = $("#grid");
    grid.innerHTML = "";
    $("#empty").hidden = view.length > 0;
    $("#resultCount").textContent =
      `${view.length} pass${view.length === 1 ? "" : "es"}` +
      (state.sat !== "all" ? ` · ${state.sat}` : "") +
      (state.q ? ` · “${state.q}”` : "");

    const frag = document.createDocumentFragment();
    view.forEach((p, i) => {
      const c = colourFor(p.satellite);
      const badge = snrColour(p.peak_snr_db);
      const el = document.createElement("article");
      el.className = "card";
      el.innerHTML = `
        <div class="card-fig">
          <img loading="lazy" src="${p.thumb}" alt="${p.name} spectrogram and SNR">
          <span class="card-sat" style="--c:${c}"><span class="dot"></span>${p.satellite}</span>
          <span class="snr-badge" style="--badge:${badge}">${fmt(p.peak_snr_db)} dB</span>
        </div>
        <div class="card-body">
          <div class="card-name">${p.name}</div>
          <div class="card-date">${p.date || "—"}</div>
          <div class="card-stats">
            <div class="s"><span class="v" style="color:${snrColour(p.median_snr_db)}">${fmt(p.median_snr_db)}</span><span class="l">median dB</span></div>
            <div class="s"><span class="v">${fmt(p.duration_s,0)}s</span><span class="l">duration</span></div>
            <div class="s"><span class="v">${fmt(p.bw_estimated_hz,0)}</span><span class="l">bw est Hz</span></div>
          </div>
        </div>`;
      el.addEventListener("click", () => openModal(i));
      frag.appendChild(el);
    });
    grid.appendChild(frag);
  }

  // ── modal ──────────────────────────────────────────────────────────────────
  function openModal(i) {
    modalIdx = i;
    const p = view[i];
    if (!p) return;
    const c = colourFor(p.satellite);
    $("#modalImg").src = p.full;
    $("#modalImg").alt = p.name;
    $("#modalSat").innerHTML = `<span class="dot" style="background:${c}"></span>${p.satellite}`;
    $("#modalTitle").textContent = p.name;
    $("#modalDate").textContent = p.date || "—";
    $("#modalSnr").innerHTML = `
      <div class="b"><div class="v" style="color:${snrColour(p.peak_snr_db)}">${fmt(p.peak_snr_db)}</div><div class="l">peak SNR dB</div></div>
      <div class="b"><div class="v" style="color:${snrColour(p.median_snr_db)}">${fmt(p.median_snr_db)}</div><div class="l">median SNR dB</div></div>`;

    const rows = [
      ["duration", fmt(p.duration_s, 0) + " s"],
      ["frames", p.n_frames != null ? p.n_frames : "–"],
      ["bw mode", p.bw_mode || "–"],
      ["bw used", fmt(p.bw_used_hz, 0) + " Hz"],
      ["bw estimated", fmt(p.bw_estimated_hz, 0) + " Hz"],
      ["bw rel. error", p.bw_rel_error == null ? "–" : fmt(p.bw_rel_error, 3)],
      ["interp gaps", p.interp_n_gaps == null ? "–" : p.interp_n_gaps],
      ["interp ok", boolTag(p.interp_ok)],
      ["bw test ok", boolTag(p.bw_test_ok)],
    ];
    $("#modalTable").innerHTML = rows.map(
      ([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

    $("#modal").hidden = false;
    document.body.style.overflow = "hidden";
    $("#navPrev").style.visibility = i > 0 ? "visible" : "hidden";
    $("#navNext").style.visibility = i < view.length - 1 ? "visible" : "hidden";
  }
  const boolTag = b => b ? `<span class="tag-ok">✓ yes</span>` : `<span class="tag-bad">✗ no</span>`;

  function closeModal() {
    $("#modal").hidden = true;
    document.body.style.overflow = "";
    modalIdx = -1;
  }
  function step(d) {
    const n = modalIdx + d;
    if (n >= 0 && n < view.length) openModal(n);
  }

  // ── events ─────────────────────────────────────────────────────────────────
  $("#searchBox").addEventListener("input", e => { state.q = e.target.value.trim(); render(); });
  $("#sortSel").addEventListener("change", e => { state.sort = e.target.value; render(); });
  $("#navPrev").addEventListener("click", () => step(-1));
  $("#navNext").addEventListener("click", () => step(1));
  document.querySelectorAll("[data-close]").forEach(el => el.addEventListener("click", closeModal));
  document.addEventListener("keydown", e => {
    if ($("#modal").hidden) return;
    if (e.key === "Escape") closeModal();
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
  });

  // ── go ─────────────────────────────────────────────────────────────────────
  renderHead();
  renderOverview();
  renderFilters();
  render();
})();
