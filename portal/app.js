/* Ajay Mitchell shot-mechanics portal — reads portal/data.json (static, read-only). */

const PLAYER = {
  name: "Ajay Mitchell",
  team: "Oklahoma City Thunder",
  jersey: 25,
  position: "Guard",
  photo: "assets/mitchell_card.jpg",
  bio: [
    { k: "Height", v: "6'4\" (1.93m)" },
    { k: "Weight", v: "190 lb (86kg)" },
    { k: "Age", v: "24" },
    { k: "College", v: "UC Santa Barbara" },
    { k: "Draft", v: "2024, Rd 2, Pick 38" },
  ],
  // 2025-26 regular season, via NBA.com / ESPN
  stats: [
    { k: "PPG", v: "13.6" },
    { k: "RPG", v: "3.3" },
    { k: "APG", v: "3.6" },
    { k: "FG%", v: "48.5" },
    { k: "3P%", v: "34.7" },
    { k: "FT%", v: "87.0" },
    { k: "MPG", v: "25.8" },
  ],
  statsNote: "2025–26 season · 57 GP",
};

let DATA = null;
let compareDim = "result_en"; // "result_en" | "shot_type_group"
let clipSort = { key: "id", dir: 1 };
let clipFilter = { shotType: "all", result: "all" };
const chartInstances = [];

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const GROUPS = {
  result_en: {
    label: "Outcome",
    categories: ["Make", "Miss"],
    colorVar: { Make: "--series-make", Miss: "--series-miss" },
  },
  shot_type_group: {
    label: "Shot type",
    categories: ["Pull-up", "Catch-and-shoot"],
    colorVar: { "Pull-up": "--series-pullup", "Catch-and-shoot": "--series-catchshoot" },
  },
};

const METRIC_UNITS = {
  elbow_angle: "°",
  knee_angle: "°",
  wrist_height: "",
  body_lean: "°",
  guide_sep: "",
};

function fmtMetric(metric, v) {
  if (v === null || v === undefined) return "—";
  if (metric === "wrist_height") return v.toFixed(3);
  if (metric === "guide_sep") return v.toFixed(4);
  return v.toFixed(1) + METRIC_UNITS[metric];
}

function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// vertical reference line at rel_frame = 0 (release)
const releaseLinePlugin = {
  id: "releaseLine",
  afterDraw(chart) {
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    if (!xScale || !yScale) return;
    const x = xScale.getPixelForValue(0);
    if (x < xScale.left || x > xScale.right) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = cssVar("--baseline") || "#c3c2b7";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, yScale.top);
    ctx.lineTo(x, yScale.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--text-muted") || "#898781";
    ctx.font = "10.5px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("release", x + 4, yScale.top + 10);
    ctx.restore();
  },
};
Chart.register(releaseLinePlugin);

Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', sans-serif";
Chart.defaults.color = "#898781";

function destroyCharts() {
  while (chartInstances.length) chartInstances.pop().destroy();
}

function baseScales(metric) {
  const scales = {
    x: {
      type: "linear",
      title: { display: true, text: "Frame (relative to release)", font: { size: 11 } },
      grid: { color: cssVar("--gridline"), drawTicks: false },
      ticks: { color: cssVar("--text-muted"), font: { size: 10.5 }, maxTicksLimit: 8 },
      border: { color: cssVar("--baseline") },
    },
    y: {
      grid: { color: cssVar("--gridline"), drawTicks: false },
      ticks: { color: cssVar("--text-muted"), font: { size: 10.5 } },
      border: { color: cssVar("--baseline") },
    },
  };
  if (metric === "elbow_angle" || metric === "knee_angle") {
    scales.y.min = 0;
    scales.y.max = 190;
  }
  if (metric === "wrist_height") {
    scales.y.min = 0;
    scales.y.max = 1;
  }
  return scales;
}

function router() {
  const hash = location.hash;
  const m = hash.match(/^#\/clip\/(.+)$/);
  if (m) {
    renderDetail(decodeURIComponent(m[1]));
  } else {
    renderOverview();
  }
}

async function main() {
  const res = await fetch("data.json");
  DATA = await res.json();
  document.getElementById("header-subtitle").textContent =
    `${DATA.summary.total} shots analyzed · ${DATA.summary.make_pct}% make rate ` +
    `(${DATA.summary.n_make}/${DATA.summary.total}) · Pull-up n=${DATA.summary.by_shot_type["Pull-up"].n}, ` +
    `Catch-and-shoot n=${DATA.summary.by_shot_type["Catch-and-shoot"].n}`;
  window.addEventListener("hashchange", router);
  router();
}

// ---------------------------------------------------------------- overview

function renderOverview() {
  destroyCharts();
  document.getElementById("view-detail").style.display = "none";
  const el = document.getElementById("view-overview");
  el.style.display = "block";

  el.innerHTML = `
    <section id="player-card-section"></section>
    <section>
      <div class="kpi-row" id="kpi-row"></div>
    </section>
    <section>
      <h2 class="section-title">Shot type × outcome<span class="hint">mean ± std at release, small-n — read alongside the individual shots below</span></h2>
      <div class="grid-2x2" id="grid-2x2"></div>
    </section>
    <section>
      <h2 class="section-title">Mechanics trajectories<span class="hint">thin lines = individual shots, bold line = group mean</span></h2>
      <div class="segmented" id="compare-toggle"></div>
      <div class="chart-grid" id="compare-charts"></div>
    </section>
    <section>
      <h2 class="section-title">All shots</h2>
      <div class="filter-row" id="clip-filters"></div>
      <table class="clip-table" id="clip-table"></table>
    </section>
  `;

  renderPlayerCard();
  renderKPIs();
  render2x2();
  renderCompareToggle();
  renderCompareCharts();
  renderClipFilters();
  renderClipTable();
}

function renderPlayerCard() {
  const bioHtml = PLAYER.bio
    .map((b) => `<div class="item"><span class="k">${escapeHtml(b.k)}</span><span class="v">${escapeHtml(b.v)}</span></div>`)
    .join("");
  const statsHtml = PLAYER.stats
    .map((s) => `<div class="stat-pill"><div class="v">${escapeHtml(s.v)}</div><div class="k">${escapeHtml(s.k)}</div></div>`)
    .join("");

  document.getElementById("player-card-section").innerHTML = `
    <div class="player-card">
      <div class="photo-col">
        <span class="jersey-tag">#${PLAYER.jersey}</span>
        <img src="${PLAYER.photo}" alt="${escapeHtml(PLAYER.name)} shooting, from tracked broadcast footage">
      </div>
      <div class="info-col">
        <div class="name-row">
          <h2>${escapeHtml(PLAYER.name)}</h2>
          <span class="position-badge">${escapeHtml(PLAYER.position)}</span>
        </div>
        <div class="team-line">${escapeHtml(PLAYER.team)}</div>
        <div class="bio-row">${bioHtml}</div>
        <div class="stat-pill-row">
          ${statsHtml}
          <span class="stat-pill-note">${escapeHtml(PLAYER.statsNote)}</span>
        </div>
      </div>
    </div>
  `;
}

function renderKPIs() {
  const s = DATA.summary;
  const tiles = [
    { label: "Total shots analyzed", value: s.total, sub: "30 clips passed tracking + pose QC" },
    { label: "Make rate", value: `${s.make_pct}%`, sub: `${s.n_make} make / ${s.n_miss} miss` },
    { label: "Pull-up", value: s.by_shot_type["Pull-up"].n, sub: `${s.by_shot_type["Pull-up"].n_make} made` },
    { label: "Catch-and-shoot", value: s.by_shot_type["Catch-and-shoot"].n, sub: `${s.by_shot_type["Catch-and-shoot"].n_make} made` },
  ];
  document.getElementById("kpi-row").innerHTML = tiles
    .map(
      (t) => `<div class="stat-tile">
        <div class="label">${t.label}</div>
        <div class="value">${t.value}</div>
        <div class="sub">${t.sub}</div>
      </div>`
    )
    .join("");
}

function render2x2() {
  const metrics = DATA.metrics;
  const labels = DATA.metric_labels;
  const cellHtml = (key, colorVar) => {
    const cell = DATA.grid[key];
    const rows = metrics
      .map(
        (m) =>
          `<tr><td class="m-name">${labels[m]}</td><td class="m-val">${fmtMetric(m, cell.metrics[m].mean)} ${
            cell.metrics[m].std !== null ? `<span style="color:var(--text-muted)">±${fmtMetric(m, cell.metrics[m].std).replace(/[^0-9.\-]/g, "")}</span>` : ""
          }</td></tr>`
      )
      .join("");
    return `<div class="cell">
      <span class="n-badge"><span class="swatch" style="background:var(${colorVar})"></span>n = ${cell.n}</span>
      <table class="metric-mini-table">${rows}</table>
    </div>`;
  };

  const html = `
    <div class="cell corner"></div>
    <div class="cell head"><span class="swatch" style="background:var(--series-make)"></span>Make</div>
    <div class="cell head"><span class="swatch" style="background:var(--series-miss)"></span>Miss</div>

    <div class="cell head">Pull-up</div>
    ${cellHtml("Pull-up|Make", "--series-make")}
    ${cellHtml("Pull-up|Miss", "--series-miss")}

    <div class="cell head">Catch&#8209;and&#8209;shoot</div>
    ${cellHtml("Catch-and-shoot|Make", "--series-make")}
    ${cellHtml("Catch-and-shoot|Miss", "--series-miss")}
  `;
  document.getElementById("grid-2x2").innerHTML = html;
}

function renderCompareToggle() {
  const wrap = document.getElementById("compare-toggle");
  wrap.innerHTML = Object.entries(GROUPS)
    .map(([key, g]) => `<button data-dim="${key}" class="${key === compareDim ? "active" : ""}">By ${g.label.toLowerCase()}</button>`)
    .join("");
  wrap.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      compareDim = btn.dataset.dim;
      renderCompareToggle();
      renderCompareCharts();
    });
  });
}

function renderCompareCharts() {
  const container = document.getElementById("compare-charts");
  container.innerHTML = "";
  const group = GROUPS[compareDim];

  const legendHtml = group.categories
    .map((cat) => {
      const hex = cssVar(group.colorVar[cat]);
      return `<div class="item"><span class="key" style="background:${hex}"></span>${cat}</div>`;
    })
    .join("");

  DATA.metrics.forEach((metric) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `
      <div class="chart-title">${DATA.metric_labels[metric]}</div>
      <div class="legend-row">${legendHtml}</div>
      <canvas></canvas>
    `;
    container.appendChild(card);
    const canvas = card.querySelector("canvas");
    buildCompareChart(canvas, metric, group);
  });
}

function buildCompareChart(canvas, metric, group) {
  const datasets = [];

  group.categories.forEach((cat) => {
    const hex = cssVar(group.colorVar[cat]);
    const clips = DATA.clips.filter((c) => c[compareDim] === cat);

    // individual, de-emphasized
    clips.forEach((clip) => {
      const pts = clip.trajectory
        .filter((t) => t[metric] !== null)
        .map((t) => ({ x: t.rel_frame, y: t[metric] }));
      datasets.push({
        data: pts,
        borderColor: hexToRgba(hex, 0.22),
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.15,
        spanGaps: true,
        isIndividual: true,
        order: 2,
      });
    });

    // group mean, emphasized
    const byFrame = {};
    clips.forEach((clip) => {
      clip.trajectory.forEach((t) => {
        if (t[metric] === null) return;
        (byFrame[t.rel_frame] = byFrame[t.rel_frame] || []).push(t[metric]);
      });
    });
    const meanPts = Object.keys(byFrame)
      .map(Number)
      .sort((a, b) => a - b)
      .map((rf) => ({ x: rf, y: byFrame[rf].reduce((a, b) => a + b, 0) / byFrame[rf].length }));
    datasets.push({
      label: cat,
      data: meanPts,
      borderColor: hex,
      backgroundColor: hex,
      borderWidth: 3,
      pointRadius: 0,
      tension: 0.15,
      spanGaps: true,
      order: 1,
    });
  });

  const chart = new Chart(canvas, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: baseScales(metric),
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (item) => !item.dataset.isIndividual,
          callbacks: {
            title: (items) => (items.length ? `frame ${items[0].parsed.x}` : ""),
            label: (item) => `${item.dataset.label}: ${fmtMetric(metric, item.parsed.y)}`,
          },
        },
      },
    },
  });
  chart.canvas.style.height = "220px";
  chartInstances.push(chart);
}

function renderClipFilters() {
  const wrap = document.getElementById("clip-filters");
  wrap.innerHTML = `
    <select id="f-shottype">
      <option value="all">All shot types</option>
      <option value="Pull-up">Pull-up</option>
      <option value="Catch-and-shoot">Catch-and-shoot</option>
    </select>
    <select id="f-result">
      <option value="all">Make + Miss</option>
      <option value="Make">Make only</option>
      <option value="Miss">Miss only</option>
    </select>
    <span class="count" id="clip-count"></span>
  `;
  wrap.querySelector("#f-shottype").addEventListener("change", (e) => {
    clipFilter.shotType = e.target.value;
    renderClipTable();
  });
  wrap.querySelector("#f-result").addEventListener("change", (e) => {
    clipFilter.result = e.target.value;
    renderClipTable();
  });
}

function filteredClips() {
  return DATA.clips.filter((c) => {
    if (clipFilter.shotType !== "all" && c.shot_type_group !== clipFilter.shotType) return false;
    if (clipFilter.result !== "all" && c.result_en !== clipFilter.result) return false;
    return true;
  });
}

const CLIP_COLUMNS = [
  { key: "game_id", label: "Game" },
  { key: "game_clock", label: "Clock" },
  { key: "distance_ft", label: "Dist" },
  { key: "shot_type", label: "Shot type" },
  { key: "result_en", label: "Result" },
  { key: "elbow_angle", label: "Elbow", metric: true },
  { key: "knee_angle", label: "Knee", metric: true },
  { key: "wrist_height", label: "Wrist", metric: true },
  { key: "body_lean", label: "Lean", metric: true },
  { key: "guide_sep", label: "Guide", metric: true },
];

function sortValue(clip, key) {
  const col = CLIP_COLUMNS.find((c) => c.key === key);
  if (col && col.metric) return clip.release_values[key];
  return clip[key];
}

function renderClipTable() {
  const clips = filteredClips().slice();
  document.getElementById("clip-count").textContent = `${clips.length} shots`;

  clips.sort((a, b) => {
    let av = sortValue(a, clipSort.key);
    let bv = sortValue(b, clipSort.key);
    if (av === null || av === undefined) av = -Infinity;
    if (bv === null || bv === undefined) bv = -Infinity;
    if (typeof av === "string") return av.localeCompare(bv) * clipSort.dir;
    return (av - bv) * clipSort.dir;
  });

  const thead = `<thead><tr>${CLIP_COLUMNS.map(
    (c) => `<th data-key="${c.key}">${c.label}${clipSort.key === c.key ? (clipSort.dir === 1 ? " ↑" : " ↓") : ""}</th>`
  ).join("")}</tr></thead>`;

  const rows = clips
    .map((c) => {
      const cells = CLIP_COLUMNS.map((col) => {
        if (col.key === "result_en") {
          return `<td><span class="badge ${c.result_en === "Make" ? "make" : "miss"}">${c.result_en}</span></td>`;
        }
        if (col.key === "distance_ft") {
          return `<td>${c.distance_ft !== null ? c.distance_ft + " ft" : "—"}</td>`;
        }
        if (col.metric) {
          return `<td>${fmtMetric(col.key, c.release_values[col.key])}</td>`;
        }
        return `<td>${escapeHtml(String(c[col.key] ?? ""))}</td>`;
      }).join("");
      return `<tr data-id="${encodeURIComponent(c.id)}">${cells}</tr>`;
    })
    .join("");

  const table = document.getElementById("clip-table");
  table.innerHTML = thead + `<tbody>${rows || `<tr><td colspan="${CLIP_COLUMNS.length}"><div class="empty-note">No shots match these filters.</div></td></tr>`}</tbody>`;

  table.querySelectorAll("th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (clipSort.key === key) clipSort.dir *= -1;
      else {
        clipSort.key = key;
        clipSort.dir = 1;
      }
      renderClipTable();
    });
  });
  table.querySelectorAll("tbody tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", () => {
      location.hash = `#/clip/${tr.dataset.id}`;
    });
  });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// ------------------------------------------------------------------ detail

function renderDetail(clipId) {
  destroyCharts();
  document.getElementById("view-overview").style.display = "none";
  const el = document.getElementById("view-detail");
  el.style.display = "block";

  const clip = DATA.clips.find((c) => c.id === clipId);
  if (!clip) {
    el.innerHTML = `<div class="empty-note">Clip not found.</div>`;
    return;
  }

  el.innerHTML = `
    <a class="back-link" href="#/">&larr; Back to all shots</a>
    <div class="detail-header">
      <div>
        <h2>${escapeHtml(clip.game_id)} · ${escapeHtml(clip.game_clock)}</h2>
        <div class="detail-meta">
          <span class="item"><span class="k">Shot type</span>${escapeHtml(clip.shot_type)}</span>
          <span class="item"><span class="k">Distance</span>${clip.distance_ft !== null ? clip.distance_ft + " ft" : "—"}</span>
          <span class="item"><span class="k">Video time</span>${escapeHtml(clip.video_time)}</span>
          ${clip.note ? `<span class="item"><span class="k">Note</span>${escapeHtml(clip.note)}</span>` : ""}
        </div>
      </div>
      <span class="badge ${clip.result_en === "Make" ? "make" : "miss"}" style="font-size:13px;padding:5px 12px;">${clip.result_en}</span>
    </div>
    <div class="detail-layout">
      <div class="video-card">
        <video src="../${clip.video_url}" controls muted playsinline></video>
        <div class="caption">Skeleton overlay · release at frame ${clip.release_frame} (${clip.fps} fps)</div>
      </div>
      <div class="detail-chart-grid" id="detail-charts"></div>
    </div>
  `;

  const container = document.getElementById("detail-charts");
  const seriesColor = clip.result_en === "Make" ? cssVar("--series-make") : cssVar("--series-miss");

  DATA.metrics.forEach((metric) => {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `
      <div class="chart-title">${DATA.metric_labels[metric]}</div>
      <div class="chart-sub">at release: ${fmtMetric(metric, clip.release_values[metric])}</div>
      <canvas></canvas>
    `;
    container.appendChild(card);
    const pts = clip.trajectory.filter((t) => t[metric] !== null).map((t) => ({ x: t.rel_frame, y: t[metric] }));

    const chart = new Chart(card.querySelector("canvas"), {
      type: "line",
      data: {
        datasets: [
          {
            label: DATA.metric_labels[metric],
            data: pts,
            borderColor: seriesColor,
            backgroundColor: seriesColor,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
            spanGaps: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        scales: baseScales(metric),
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => (items.length ? `frame ${items[0].parsed.x}` : ""),
              label: (item) => `${DATA.metric_labels[metric]}: ${fmtMetric(metric, item.parsed.y)}`,
            },
          },
        },
      },
    });
    chart.canvas.style.height = "200px";
    chartInstances.push(chart);
  });
}

main();
