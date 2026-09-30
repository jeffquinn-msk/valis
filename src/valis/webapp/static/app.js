"use strict";

const SIDES = ["image", "reference"];

const state = {
  schema: null,
  sessionId: null,
  matcher: {},
  align: {},
  side: {
    // Only the moving image has a geometric pre-transform; the reference
    // defines the aligned output's frame. ``res.size`` is the working
    // resolution (longest side, px) the image is preprocessed + matched at.
    image: {
      processor: null, params: {}, geometry: {}, res: {}, native: null, preview: null,
    },
    reference: { processor: null, params: {}, res: {}, native: null, preview: null },
  },
  lastMatch: null,
  matchSeq: 0,
  aligning: false,
  osdViewer: null,
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function $(sel, root = document) { return root.querySelector(sel); }
function $all(sel, root = document) { return [...root.querySelectorAll(sel)]; }

function toast(msg, ms = 4000) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res;
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---------------------------------------------------------------------------
// Schema-driven controls
// ---------------------------------------------------------------------------

function paramDefaults(procName) {
  const spec = state.schema.processors[procName];
  const out = {};
  (spec ? spec.params : []).forEach((p) => { out[p.name] = p.default; });
  return out;
}

function buildProcessorControls(side) {
  const panel = $(`.panel[data-side="${side}"]`);
  const select = $(".processor-select", panel);
  select.innerHTML = "";
  Object.keys(state.schema.processors).forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });
  select.value = state.side[side].processor;
  select.onchange = () => {
    state.side[side].processor = select.value;
    state.side[side].params = paramDefaults(select.value);
    renderParams(side);
    clearMatchOverlay();
    schedulePreprocess(side);
  };
  renderParams(side);
}

function renderParams(side) {
  const panel = $(`.panel[data-side="${side}"]`);
  const container = $(".params", panel);
  container.innerHTML = "";
  const spec = state.schema.processors[state.side[side].processor];
  (spec ? spec.params : []).forEach((p) => {
    container.appendChild(paramRow(side, p, state.side[side].params));
  });
}

function geometryDefaults() {
  const out = {};
  (state.schema.geometry || []).forEach((p) => { out[p.name] = p.default; });
  return out;
}

function renderGeometry() {
  const container = $(`.panel[data-side="image"] .geometry`);
  container.innerHTML = "";
  (state.schema.geometry || []).forEach((p) => {
    container.appendChild(paramRow("image", p, state.side.image.geometry));
  });
}

function renderResolution(side) {
  const container = $(`.panel[data-side="${side}"] .resolution`);
  container.innerHTML = "";
  container.appendChild(paramRow(
    side, { ...state.schema.resolution, name: "size", label: "working resolution (px)" },
    state.side[side].res,
  ));
}

function updateResInfo(side) {
  const s = state.side[side];
  const info = $(`.panel[data-side="${side}"] .res-info`);
  if (!s.preview || !s.native) { info.textContent = ""; return; }
  // longest sides, so a 90° rotation of the moving image doesn't skew it
  const scale = Math.max(s.preview.iw, s.preview.ih) / Math.max(s.native.w, s.native.h);
  info.textContent =
    `working image ${s.preview.iw} × ${s.preview.ih} px ` +
    `(native ${s.native.w} × ${s.native.h}, ${(scale * 100).toPrecision(3)}%)` +
    (s.res.size > Math.max(s.native.w, s.native.h) ? " — capped at native" : "");
}

function paramRow(side, p, target) {
  const row = document.createElement("div");
  const cur = target[p.name];
  const labelText = p.label || p.name;
  if (p.type === "bool") {
    row.className = "param-row bool";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!cur;
    cb.onchange = () => {
      target[p.name] = cb.checked;
      clearMatchOverlay();
      schedulePreprocess(side);
    };
    const label = document.createElement("span");
    label.className = "param-label";
    label.textContent = labelText;
    row.append(cb, label);
  } else if (p.type === "enum") {
    row.className = "param-row";
    const lab = document.createElement("div");
    lab.className = "param-label";
    lab.innerHTML = `<span>${labelText}</span>`;
    const sel = document.createElement("select");
    p.options.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o; opt.textContent = o; sel.appendChild(opt);
    });
    sel.value = cur;
    sel.onchange = () => {
      target[p.name] = sel.value;
      clearMatchOverlay();
      schedulePreprocess(side);
    };
    row.append(lab, sel);
  } else {
    // float / int slider
    row.className = "param-row";
    const lab = document.createElement("div");
    lab.className = "param-label";
    lab.innerHTML = `<span>${labelText}</span><span class="val">${cur}</span>`;
    const input = document.createElement("input");
    input.type = "range";
    input.min = p.min; input.max = p.max; input.step = p.step;
    input.value = cur;
    input.oninput = () => {
      const v = p.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
      target[p.name] = v;
      $(".val", lab).textContent = v;
      clearMatchOverlay();
      schedulePreprocess(side);
    };
    row.append(lab, input);
  }
  return row;
}

// ``onChange(key)`` runs after a control changes ``target[key]``.
function buildControlGrid(grid, schema, target, onChange) {
  grid.innerHTML = "";
  Object.keys(schema).forEach((key) => {
    const spec = schema[key];
    target[key] = spec.default;
    const wrap = document.createElement("label");
    wrap.className = "ctrl";
    const labelText = key.replace(/_/g, " ");
    if (spec.type === "enum") {
      wrap.innerHTML = `<span>${labelText}</span>`;
      const sel = document.createElement("select");
      spec.options.forEach((o) => {
        const opt = document.createElement("option");
        opt.value = o; opt.textContent = o; sel.appendChild(opt);
      });
      sel.value = spec.default;
      sel.onchange = () => { target[key] = sel.value; onChange(key); };
      wrap.appendChild(sel);
    } else {
      wrap.innerHTML =
        `<span>${labelText}: <b class="mval">${spec.default}</b></span>`;
      const input = document.createElement("input");
      input.type = "range";
      input.min = spec.min; input.max = spec.max; input.step = spec.step;
      input.value = spec.default;
      input.oninput = () => {
        const v = spec.type === "int"
          ? parseInt(input.value, 10) : parseFloat(input.value);
        target[key] = v;
        $(".mval", wrap).textContent = v;
        onChange(key);
      };
      wrap.appendChild(input);
    }
    grid.appendChild(wrap);
  });
}

function buildMatcherControls() {
  // Matches shown must always be the ones the current settings produce.
  buildControlGrid($("#matcher-controls .matcher-grid"), state.schema.matcher,
    state.matcher, clearMatchOverlay);
}

function buildAlignControls() {
  buildControlGrid($(".align-grid"), state.schema.alignment, state.align,
    updateAlignButton);
}

// ---------------------------------------------------------------------------
// Preview rendering
// ---------------------------------------------------------------------------

// One debounce timer per side: a shared timer would let a change on one
// panel cancel the pending re-render of the other.
const preprocessTimers = {};
SIDES.forEach((side) => {
  preprocessTimers[side] = debounce(() => { preprocess(side); }, 220);
});
function schedulePreprocess(side) { preprocessTimers[side](); }

// Monotonic request counter per side so a slow, stale response can't
// overwrite the preview for newer parameters.
const preprocessSeq = {};

async function preprocess(side) {
  if (!state.sessionId) return;
  const seq = (preprocessSeq[side] = (preprocessSeq[side] || 0) + 1);
  const s = state.side[side];
  const body = JSON.stringify({
    processor: s.processor, params: s.params, geometry: s.geometry, size: s.res.size,
  });
  try {
    const res = await api(`/api/preprocess/${state.sessionId}/${side}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body,
    });
    const blob = await res.blob();
    const img = await blobToImage(blob);
    if (seq !== preprocessSeq[side]) return;
    drawPreview(side, img);
  } catch (e) {
    toast(`Preprocess (${side}) failed: ${e.message}`);
  }
}

function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = reject;
    img.src = url;
  });
}

function drawPreview(side, img) {
  const panel = $(`.panel[data-side="${side}"]`);
  const canvas = $("canvas.preview", panel);
  const cw = canvas.width, ch = canvas.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  const scale = Math.min(cw / img.width, ch / img.height);
  const dw = img.width * scale, dh = img.height * scale;
  const ox = (cw - dw) / 2, oy = (ch - dh) / 2;
  ctx.drawImage(img, ox, oy, dw, dh);
  state.side[side].preview = {
    img, dw, dh, ox, oy, iw: img.width, ih: img.height,
  };
  updateResInfo(side);
}

// ---------------------------------------------------------------------------
// Matching + overlay
// ---------------------------------------------------------------------------

function clearMatchOverlay() {
  const ov = $("#match-overlay");
  const ctx = ov.getContext("2d");
  ctx.clearRect(0, 0, ov.width, ov.height);
  $("#match-readout").textContent = "";
  state.lastMatch = null;
  state.matchSeq += 1; // an in-flight match is now stale
  updateAlignButton();
  // redraw previews to erase dots
  SIDES.forEach((side) => {
    const p = state.side[side].preview;
    if (p) drawPreview(side, p.img);
  });
}

function updateAlignButton() {
  const btn = $("#align-btn");
  if (state.aligning) return;
  const m = state.lastMatch;
  const need = state.align.min_matches;
  btn.disabled = !m || m.n_filtered < need;
  btn.title = !m
    ? "Run keypoint detection first; alignment starts from its matches"
    : m.n_filtered < need
      ? `Only ${m.n_filtered} matches; min matches is ${need}`
      : "";
  if (m) {
    $("#match-readout").textContent =
      `${m.n_filtered} filtered / ${m.n_total} total matches` +
      (m.n_filtered < need ? ` — below min matches (${need})` : "");
  }
}

function imageCfg() {
  const s = state.side.image;
  return {
    processor: s.processor, params: s.params, geometry: s.geometry, size: s.res.size,
  };
}

async function runMatch() {
  if (!state.sessionId) return;
  const btn = $("#match-btn");
  btn.disabled = true;
  btn.textContent = "Detecting…";
  clearMatchOverlay();
  const seq = state.matchSeq;
  try {
    const body = JSON.stringify({
      image: imageCfg(),
      reference: {
        processor: state.side.reference.processor,
        params: state.side.reference.params,
        size: state.side.reference.res.size,
      },
      matcher: state.matcher,
    });
    const res = await api(`/api/match/${state.sessionId}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body,
    });
    const data = await res.json();
    if (seq !== state.matchSeq) return; // settings changed while matching
    state.lastMatch = data;
    drawMatches(data);
    updateAlignButton();
  } catch (e) {
    toast(`Matching failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Keypoint Detection";
  }
}

function sideToCanvasPoint(side, xy) {
  // xy is in preprocessed-thumbnail pixel space (== preview natural dims)
  const p = state.side[side].preview;
  if (!p) return null;
  return {
    x: p.ox + (xy[0] / p.iw) * p.dw,
    y: p.oy + (xy[1] / p.ih) * p.dh,
  };
}

function drawMatches(data) {
  // Dots on each panel canvas
  SIDES.forEach((side) => {
    const p = state.side[side].preview;
    if (!p) return;
    drawPreview(side, p.img);
    const canvas = $(`canvas.preview`, $(`.panel[data-side="${side}"]`));
    const ctx = canvas.getContext("2d");
    const kps = side === "image" ? data.matches.kp1 : data.matches.kp2;
    ctx.lineWidth = 1;
    kps.forEach((xy, i) => {
      const pt = sideToCanvasPoint(side, xy);
      if (!pt) return;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = hueFor(i, kps.length);
      ctx.fill();
    });
  });
  drawConnectors(data);
}

function hueFor(i, n) {
  const h = Math.round((360 * i) / Math.max(1, n));
  return `hsl(${h}, 90%, 60%)`;
}

function positionOverlay() {
  const ov = $("#match-overlay");
  const panels = $(".panels");
  const rect = panels.getBoundingClientRect();
  ov.style.left = panels.offsetLeft + "px";
  ov.style.top = panels.offsetTop + "px";
  ov.width = rect.width;
  ov.height = rect.height;
  return { panels, rect };
}

function drawConnectors(data) {
  const { panels, rect } = positionOverlay();
  const ov = $("#match-overlay");
  const ctx = ov.getContext("2d");
  ctx.clearRect(0, 0, ov.width, ov.height);
  ctx.lineWidth = 0.6;
  ctx.globalAlpha = 0.5;

  const canvasOffset = (side) => {
    const canvas = $("canvas.preview", $(`.panel[data-side="${side}"]`));
    const cRect = canvas.getBoundingClientRect();
    // scale between canvas internal px and displayed px
    return {
      left: cRect.left - rect.left,
      top: cRect.top - rect.top,
      sx: cRect.width / canvas.width,
      sy: cRect.height / canvas.height,
    };
  };
  const offImg = canvasOffset("image");
  const offRef = canvasOffset("reference");

  const n = data.matches.kp1.length;
  for (let i = 0; i < n; i++) {
    const a = sideToCanvasPoint("image", data.matches.kp1[i]);
    const b = sideToCanvasPoint("reference", data.matches.kp2[i]);
    if (!a || !b) continue;
    ctx.strokeStyle = hueFor(i, n);
    ctx.beginPath();
    ctx.moveTo(offImg.left + a.x * offImg.sx, offImg.top + a.y * offImg.sy);
    ctx.lineTo(offRef.left + b.x * offRef.sx, offRef.top + b.y * offRef.sy);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// Alignment job
// ---------------------------------------------------------------------------

async function runAlignment() {
  if (!state.sessionId || !state.lastMatch) return;
  const btn = $("#align-btn");
  state.aligning = true;
  btn.disabled = true;
  $("#outputs").hidden = true;
  const body = JSON.stringify({
    match_id: state.lastMatch.match_id,
    min_matches: state.align.min_matches,
    valis_resolution: state.align.valis_resolution,
  });
  try {
    const res = await api(`/api/align/${state.sessionId}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body,
    });
    const { job_id } = await res.json();
    pollJob(job_id);
  } catch (e) {
    toast(`Alignment failed to start: ${e.message}`);
    state.aligning = false;
    updateAlignButton();
  }
}

async function pollJob(jobId) {
  const btn = $("#align-btn");
  try {
    const res = await api(`/api/align/${jobId}/status`);
    const s = await res.json();
    btn.textContent = `Aligning… ${s.stage} ${(s.progress * 100).toFixed(0)}%`;
    if (s.state === "done") {
      btn.textContent = "Run Alignment";
      state.aligning = false;
      updateAlignButton();
      showResult(jobId);
      showOutputs(jobId);
      return;
    }
    if (s.state === "error") {
      btn.textContent = "Run Alignment";
      state.aligning = false;
      updateAlignButton();
      toast(`Alignment error: ${s.message}`);
      // A failed run still leaves diagnostics (e.g. the failed-matches plot).
      showOutputs(jobId);
      return;
    }
    setTimeout(() => pollJob(jobId), 1000);
  } catch (e) {
    btn.textContent = "Run Alignment";
    state.aligning = false;
    updateAlignButton();
    toast(`Status poll failed: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Valis output: directory, summary metrics, plots, file listing
// ---------------------------------------------------------------------------

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function humanSize(n) {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function fmtCell(v) {
  const x = Number(v);
  return v !== "" && Number.isFinite(x) && !Number.isInteger(x) ? x.toPrecision(4) : v;
}

async function showOutputs(jobId) {
  let data;
  try {
    const res = await api(`/api/result/${jobId}/outputs`);
    data = await res.json();
  } catch (e) {
    toast(`Could not list valis output: ${e.message}`);
    return;
  }
  const fileUrl = (path) =>
    `/api/result/${jobId}/file?path=${encodeURIComponent(path)}`;

  $("#out-dir").textContent = data.out_dir;
  $("#copy-dir-btn").onclick = async () => {
    try {
      await navigator.clipboard.writeText(data.out_dir);
      toast("Output path copied", 1500);
    } catch (e) {
      toast(`Copy failed: ${e.message}`);
    }
  };

  const summary = $("#summary");
  summary.innerHTML = "";
  if (data.summary.length) {
    const cols = Object.keys(data.summary[0]);
    const table = el("table", { className: "summary-table" },
      el("thead", {}, el("tr", {}, ...cols.map((c) => el("th", { textContent: c })))),
      el("tbody", {}, ...data.summary.map((r) =>
        el("tr", {}, ...cols.map((c) => el("td", { textContent: fmtCell(r[c]) }))))));
    summary.append(el("div", { className: "table-wrap" }, table));
  }

  const groups = $("#plot-groups");
  groups.innerHTML = "";
  if (!data.plot_groups.length) {
    groups.append(el("p", { className: "muted", textContent: "No plots were written." }));
  }
  data.plot_groups.forEach(({ group, plots }) => {
    const grid = el("div", { className: "plot-grid" });
    plots.forEach((p) => {
      const url = fileUrl(p.path);
      grid.append(el("a", { href: url, target: "_blank", className: "plot" },
        el("img", { src: url, loading: "lazy", alt: p.name }),
        el("span", { textContent: p.name })));
    });
    groups.append(el("h3", { textContent: group.replace(/_/g, " ") }), grid);
  });

  const ul = $("#file-list ul");
  ul.innerHTML = "";
  data.files.forEach((f) => {
    const label = `${f.path}${f.link ? " (link)" : ""}`;
    const name = f.link
      ? el("span", { textContent: label })
      : el("a", { href: fileUrl(f.path), target: "_blank", textContent: label });
    ul.append(el("li", {}, name, el("span", { className: "muted", textContent: humanSize(f.size) })));
  });
  $("#file-list summary").textContent = `All files (${data.files.length})`;

  $("#outputs").hidden = false;
}

function ensureGeoTIFFEnabled() {
  if (window.OpenSeadragon && OpenSeadragon.GeoTIFFTileSource) return true;
  const enable = window.enableGeoTIFFTileSource
    || (window.GeoTIFFTileSource && window.GeoTIFFTileSource.enableGeoTIFFTileSource);
  if (enable) { enable(OpenSeadragon); return true; }
  return !!(window.OpenSeadragon && OpenSeadragon.GeoTIFFTileSource);
}

const TINT_MOVING = [0, 1, 0]; // green
const TINT_REFERENCE = [1, 0, 1]; // magenta

function tintContext(ctx, [r, g, b]) {
  // Map a grayscale tile's intensity onto a single color, in place.
  const { width, height } = ctx.canvas;
  const img = ctx.getImageData(0, 0, width, height);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i];
    d[i] = v * r;
    d[i + 1] = v * g;
    d[i + 2] = v * b;
  }
  ctx.putImageData(img, 0, 0);
}

async function showResult(jobId) {
  $("#tuning").hidden = true;
  const result = $("#result");
  result.hidden = false;
  try {
    if (!ensureGeoTIFFEnabled()) {
      throw new Error("GeoTIFFTileSource plugin not loaded");
    }
    // aligned.ome.tif stacks two same-size pages (0 = warped moving image,
    // 1 = reference) with SubIFD pyramids. The plugin can't read SubIFDs and
    // its fallback ignores hints.layout.planeIndex, so both layers showed
    // page 0. The server re-exposes each page as its own single-page,
    // IFD-pyramid TIFF instead. (The options arg is required: the plugin
    // reads opts.GeoTIFFOptions.)
    const plane = (i) => OpenSeadragon.GeoTIFFTileSource
      .getAllTileSources(`/api/result/${jobId}/plane/${i}.tif`, {})
      .then((srcs) => srcs[0]);
    const [moving, reference] = await Promise.all([plane(0), plane(1)]);
    if (state.osdViewer) { state.osdViewer.destroy(); }
    const viewer = OpenSeadragon({
      element: $("#osd"),
      drawer: "canvas", // the default WebGL drawer rendered nothing in testing
      showNavigator: true,
      showNavigationControl: false, // avoids needing button image assets
      gestureSettingsMouse: { clickToZoom: false },
      tileSources: [moving],
    });
    // One slider fades between the two: 0 = moving only, 1 = reference
    // only, 0.5 = both at half opacity.
    const mix = () => parseFloat($("#opacity").value);
    const applyMix = () => {
      const mov = viewer.world.getItemAt(0);
      const ref = viewer.world.getItemAt(1);
      if (mov) mov.setOpacity(1 - mix());
      if (ref) ref.setOpacity(mix());
    };
    state.osdViewer = viewer;
    // Tint the grayscale planes (moving = green, reference = magenta) and add
    // them, so aligned tissue reads white and misalignment shows colored fringes.
    // Registered before "open" so the first tiles are tinted too.
    const tintFor = (tiledImage) =>
      tiledImage.source === moving ? TINT_MOVING : TINT_REFERENCE;
    viewer.addHandler("tile-invalidated", async (e) => {
      const ctx = await e.getData("context2d");
      if (!ctx || (await e.outdated())) return;
      tintContext(ctx, tintFor(e.tiledImage));
      await e.setData(ctx, "context2d");
    });
    viewer.addHandler("open", () => {
      viewer.addTiledImage({
        tileSource: reference,
        compositeOperation: "lighter",
        index: 1,
        success: applyMix,
      });
    });
    $("#opacity").oninput = applyMix;
  } catch (e) {
    toast(`Could not open result viewer: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Directory browser
// ---------------------------------------------------------------------------

const picks = { reference: null, image: null };

async function openBrowser() {
  picks.reference = null; picks.image = null;
  updatePickLabels();
  $("#browse-modal").hidden = false;
  await loadDir("");
}

async function loadDir(path) {
  try {
    const res = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    const data = await res.json();
    renderBrowser(data);
  } catch (e) {
    toast(`Browse failed: ${e.message}`);
  }
}

function renderBrowser(data) {
  const crumbs = $("#crumbs");
  crumbs.innerHTML = "";
  const rootLink = document.createElement("a");
  rootLink.textContent = "root";
  rootLink.onclick = () => loadDir("");
  crumbs.append(rootLink);
  if (data.path) {
    const parts = data.path.split("/");
    let acc = "";
    parts.forEach((part) => {
      acc = acc ? `${acc}/${part}` : part;
      const seg = acc;
      crumbs.append(document.createTextNode(" / "));
      const a = document.createElement("a");
      a.textContent = part;
      a.onclick = () => loadDir(seg);
      crumbs.append(a);
    });
  }

  const browser = $("#browser");
  browser.innerHTML = "";
  if (data.parent !== null) {
    browser.append(entryRow("📁", "..", () => loadDir(data.parent), null));
  }
  data.dirs.forEach((d) => {
    browser.append(entryRow("📁", d.name, () => loadDir(d.path), null));
  });
  data.files.forEach((f) => {
    browser.append(entryRow("🖼", f.name, null, f.path));
  });
}

function entryRow(icon, name, onOpen, filePath) {
  const row = document.createElement("div");
  row.className = "entry" + (filePath ? " file" : " dir");
  if (filePath) row.dataset.path = filePath;
  row.innerHTML = `<span class="icon">${icon}</span><span>${name}</span>`;
  row.onclick = () => {
    if (onOpen) { onOpen(); return; }
    selectFile(filePath);
  };
  return row;
}

function selectFile(path) {
  if (!picks.reference) picks.reference = path;
  else if (!picks.image) picks.image = path;
  else { picks.reference = path; picks.image = null; } // restart
  updatePickLabels();
  $all(".entry.file").forEach((el) => {
    el.classList.toggle(
      "selected", el.dataset.path === picks.reference || el.dataset.path === picks.image
    );
  });
}

function updatePickLabels() {
  $("#sel-reference").textContent = picks.reference || "— click a file —";
  $("#sel-image").textContent = picks.image || "— click a file —";
  $("#load-btn").disabled = !(picks.reference && picks.image);
}

async function loadSession() {
  try {
    const res = await api("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_path: picks.image, reference_path: picks.reference,
      }),
    });
    const data = await res.json();
    state.sessionId = data.session_id;
    // apply suggested stains
    SIDES.forEach((side) => {
      const sug = data[side].suggested_stain;
      state.side[side].processor =
        sug && state.schema.processors[sug]
          ? sug
          : Object.keys(state.schema.processors)[0];
      state.side[side].params = paramDefaults(state.side[side].processor);
      state.side[side].native = { w: data[side].w, h: data[side].h };
      state.side[side].res = { size: state.schema.resolution.default };
      state.side[side].preview = null;
      buildProcessorControls(side);
      renderResolution(side);
    });
    state.side.image.geometry = geometryDefaults();
    renderGeometry();
    $("#browse-modal").hidden = true;
    $("#loaded-label").textContent =
      `${basename(picks.image)}  ↔  ${basename(picks.reference)}`;
    $("#match-btn").disabled = false;
    clearMatchOverlay();
    SIDES.forEach((side) => preprocess(side));
  } catch (e) {
    toast(`Load failed: ${e.message}`);
  }
}

function basename(p) { return p ? p.split("/").pop() : ""; }

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  try {
    const res = await api("/api/schema");
    state.schema = await res.json();
  } catch (e) {
    toast(`Could not load schema: ${e.message}`);
    return;
  }
  buildMatcherControls();
  buildAlignControls();

  $("#open-btn").onclick = openBrowser;
  $("#browse-close").onclick = () => { $("#browse-modal").hidden = true; };
  $("#load-btn").onclick = loadSession;
  $("#match-btn").onclick = runMatch;
  $("#align-btn").onclick = runAlignment;
  $("#back-btn").onclick = () => {
    $("#result").hidden = true;
    $("#tuning").hidden = false;
    if (state.lastMatch) drawConnectors(state.lastMatch);
  };

  window.addEventListener("resize", () => {
    if (state.lastMatch) drawConnectors(state.lastMatch);
  });
}

init();
