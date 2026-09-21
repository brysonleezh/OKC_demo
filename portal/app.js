/* Ajay Mitchell shot-mechanics portal — reads portal/data.json (static, read-only). */

const PLAYER = {
  name: "Ajay Mitchell",
  team: "Oklahoma City Thunder",
  jersey: 25,
  position: "Guard",
  photo: "https://a.espncdn.com/i/headshots/nba/players/full/4900671.png",
  photoFallback: "assets/mitchell_card.jpg",
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
let compareMetric = "elbow_angle";
let clipFilter = { shotType: "all", result: "all" };
let browseIndex = 0;
const chartInstances = [];
const browserCharts = [];

function destroyBrowserCharts() {
  while (browserCharts.length) browserCharts.pop().destroy();
}

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
  const clipMatch = hash.match(/^#\/clip\/(.+)$/);
  if (clipMatch) {
    renderDetail(decodeURIComponent(clipMatch[1]));
  } else if (hash === "#/portal") {
    renderOverview();
  } else {
    renderDemoFlow();
  }
}

let portalSubtitle = "";

// keeps one shared header (same navy bar + logo on every route) but lets
// each view supply its own title / subtitle / optional right-side pill
function setHeaderContent(title, subtitle, pillHtml) {
  document.getElementById("header-title").textContent = title;
  document.getElementById("header-subtitle").textContent = subtitle;
  document.getElementById("header-pill").innerHTML = pillHtml || "";
}

async function main() {
  const res = await fetch("data.json");
  DATA = await res.json();
  portalSubtitle =
    `${DATA.summary.total} shots analyzed · ${DATA.summary.make_pct}% make rate ` +
    `(${DATA.summary.n_make}/${DATA.summary.total}) · Pull-up n=${DATA.summary.by_shot_type["Pull-up"].n}, ` +
    `Catch-and-shoot n=${DATA.summary.by_shot_type["Catch-and-shoot"].n}`;
  window.addEventListener("hashchange", router);
  router();
}

// --------------------------------------------------------------- demo flow

const DEMO_VIDEO_URL = "../tracks/pull_up/OKC_MEM_1109_3_0701_pull_up_1119_pose_preview_web.mp4";

const DEMO_AGENDA = [
  { time: "0:00", label: "The video" },
  { time: "1:00", label: "What I built" },
  { time: "2:30", label: "The finding" },
  { time: "3:30", label: "Fit with OKC" },
  { time: "4:30", label: "Questions" },
];

const DEMO_DEFS = [
  { k: "Clip", v: "30 shot attempts cut from broadcast footage — 15 pull-up, 15 catch-and-shoot." },
  { k: "Track + pose", v: "Player tracking + RTMPose keypoints on every frame; clips failing QC are dropped." },
  { k: "Release detection", v: "Release frame auto-detected; metrics sampled on a ±20-frame window around it." },
  { k: "Mechanics", v: "Elbow angle, knee angle, wrist height, body lean, guide-hand separation — per frame." },
  { k: "Dashboard", v: "Per-shot skeleton video + trajectory charts, and a shot type × outcome comparison." },
];

const DEMO_BULLETS = [
  {
    strong: "Works from broadcast video.",
    rest: " No wearables, no lab session — every player OKC cares about already has this footage.",
  },
  {
    strong: "Turns coach-speak into a number.",
    rest: ' "His elbow drops on misses" becomes 142° vs 164°, a target a development coach can track week over week.',
  },
  {
    strong: "Fits the charting loop.",
    rest: " Clip in the morning, skeleton video and mechanics report by practice.",
  },
];

function renderDemoFlow() {
  destroyCharts();
  document.getElementById("portal-header").style.display = "";
  setHeaderContent(
    "Shot Mechanics from Broadcast Video — Project Brief",
    "Ajay Mitchell · 30 shots · RTMPose pipeline · built solo",
    `<div class="header-pill-badge">Video Analysis &amp; Charting · Player Development</div>`
  );
  document.getElementById("view-overview").style.display = "none";
  document.getElementById("view-detail").style.display = "none";

  const talkTrack = new URLSearchParams(location.search).get("talk") === "1";

  const el = document.getElementById("view-demo");
  el.style.display = "block";
  el.className = "demo-flow";

  const agendaHtml = DEMO_AGENDA.map(
    (a) => `<div class="df-agenda-row"><span class="df-time">${a.time}</span><span class="df-label">${a.label}</span></div>`
  ).join("");

  const cueHtml = talkTrack
    ? `<div class="df-cue-box"><strong>Cue:</strong> keep each section under 75 seconds; land on the caveat yourself before they ask.</div>`
    : "";

  const defsHtml = DEMO_DEFS.map(
    (d) => `<div class="df-def-row"><dt>${escapeHtml(d.k)}</dt><dd>${d.v}</dd></div>`
  ).join("");

  const bulletsHtml = DEMO_BULLETS.map(
    (b) => `<div class="df-bullet"><span class="df-dot"></span><span><strong>${escapeHtml(b.strong)}</strong>${b.rest}</span></div>`
  ).join("");

  el.innerHTML = `
    <div class="df-wrap">
      <div class="df-body-grid">
        <div class="df-rail">
          <div class="df-rail-label">5-Minute Walkthrough</div>
          ${agendaHtml}
          ${cueHtml}
        </div>

        <div class="df-sections">
          <section>
            <video class="df-video" id="demo-video" src="${DEMO_VIDEO_URL}" controls autoplay muted loop playsinline></video>
            <div class="df-chip-row">
              <span class="df-chip">RTMPose 2D keypoints, every frame</span>
              <span class="df-chip">Release frame auto-detected</span>
              <span class="df-chip">No wearables — broadcast footage only</span>
            </div>
          </section>

          <section>
            <h2 class="df-section-title">1 &middot; What I Built</h2>
            <div class="df-def-list">${defsHtml}</div>
          </section>

          <section>
            <h2 class="df-section-title">2 &middot; The Finding</h2>
            <div class="df-card-grid">
              <div class="df-card df-card-a">
                <div class="df-card-label">Elbow angle at release</div>
                <div class="df-stat-row"><span class="df-stat-num make">163.9&deg;</span><span class="df-stat-tag">makes</span></div>
                <div class="df-stat-row"><span class="df-stat-num miss">142.0&deg;</span><span class="df-stat-tag">misses</span></div>
                <div class="df-footnote">d = 0.60 — the largest gap of the five metrics.</div>
              </div>
              <div class="df-card df-card-b">
                <div class="df-card-label">Read it honestly</div>
                <div class="df-card-body">Suggestive, not proven: p = 0.10 at n = 30. Flagged as a hypothesis to test, not a conclusion to coach on.</div>
              </div>
              <div class="df-card df-card-c">
                <div class="df-card-label">Why it still matters</div>
                <div class="df-card-body">The pipeline is reusable. At 300 shots — or any other player — the same code gives a real answer in hours, not weeks.</div>
              </div>
            </div>
          </section>

          <section>
            <h2 class="df-section-title">3 &middot; Fit with Player Development</h2>
            <div class="df-bullets">${bulletsHtml}</div>
          </section>

          <section class="df-footer-strip">
            <p><strong>Then live:</strong> the full portal — filter any of the 30 shots, open its skeleton clip and per-frame charts.</p>
            <button class="df-open-portal-btn" id="open-portal-btn">Open Portal &rarr;</button>
          </section>
        </div>
      </div>
    </div>
  `;

  document.getElementById("open-portal-btn").addEventListener("click", () => {
    location.hash = "#/portal";
  });

  const video = document.getElementById("demo-video");
  video.addEventListener("error", () => {
    video.outerHTML = `<div class="df-video-placeholder">Skeleton-overlay clip not found.<br>Expected at <code>${DEMO_VIDEO_URL.replace("../", "")}</code></div>`;
  });
}

// ---------------------------------------------------------------- overview

function renderOverview() {
  destroyCharts();
  document.getElementById("view-demo").style.display = "none";
  document.getElementById("view-demo").innerHTML = "";
  document.getElementById("portal-header").style.display = "";
  setHeaderContent("Ajay Mitchell — Shot Mechanics Portal", portalSubtitle, "");
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
      <h2 class="section-title">Mechanics trajectories<span class="hint">thin lines = individual shots, bold line = group mean, shaded band = &plusmn;1 SEM &mdash; overlapping bands mean the gap isn't reliable at this n</span></h2>
      <div class="filter-row" id="metric-select-row"></div>
      <div class="chart-grid" id="compare-charts"></div>
    </section>
    <section>
      <h2 class="section-title">Browse shots<span class="hint">skeleton video + mechanics charts for every shot — use &larr; &rarr; to move between them</span></h2>
      <div class="filter-row" id="clip-filters"></div>
      <div id="clip-browser"></div>
    </section>
  `;

  browseIndex = 0;
  renderPlayerCard();
  renderKPIs();
  render2x2();
  renderMetricSelect();
  renderCompareCharts();
  renderClipFilters();
  renderClipBrowser();
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
        <img src="${PLAYER.photo}" alt="${escapeHtml(PLAYER.name)} — ESPN profile photo" referrerpolicy="no-referrer"
             onerror="this.onerror=null; this.src='${PLAYER.photoFallback}';">
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
    <div class="cell corner col-head"></div>
    <div class="cell head col-head"><span class="swatch" style="background:var(--series-make)"></span>Make</div>
    <div class="cell head col-head"><span class="swatch" style="background:var(--series-miss)"></span>Miss</div>

    <div class="cell head row-label">Pull-up</div>
    ${cellHtml("Pull-up|Make", "--series-make")}
    ${cellHtml("Pull-up|Miss", "--series-miss")}

    <div class="cell head row-label">Catch&#8209;and&#8209;shoot</div>
    ${cellHtml("Catch-and-shoot|Make", "--series-make")}
    ${cellHtml("Catch-and-shoot|Miss", "--series-miss")}
  `;
  document.getElementById("grid-2x2").innerHTML = html;
}

function renderMetricSelect() {
  const wrap = document.getElementById("metric-select-row");
  wrap.innerHTML = `
    <label for="metric-select">Metric</label>
    <select id="metric-select">
      ${DATA.metrics.map((m) => `<option value="${m}">${DATA.metric_labels[m]}</option>`).join("")}
    </select>
  `;
  wrap.querySelector("#metric-select").value = compareMetric;
  wrap.querySelector("#metric-select").addEventListener("change", (e) => {
    compareMetric = e.target.value;
    renderCompareCharts();
  });
}

function renderCompareCharts() {
  const container = document.getElementById("compare-charts");
  container.innerHTML = "";

  Object.entries(GROUPS).forEach(([groupKey, group]) => {
    const legendHtml = group.categories
      .map((cat) => {
        const hex = cssVar(group.colorVar[cat]);
        const n = DATA.clips.filter((c) => c[groupKey] === cat).length;
        return `<div class="item"><span class="key" style="background:${hex}"></span>${cat} (n=${n})</div>`;
      })
      .join("");

    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `
      <div class="chart-title">By ${group.label.toLowerCase()}</div>
      <div class="legend-row">${legendHtml}</div>
      <canvas></canvas>
    `;
    container.appendChild(card);
    const canvas = card.querySelector("canvas");
    buildCompareChart(canvas, compareMetric, group, groupKey);
  });
}

// mean and standard error of the mean (sample std, n-1) for a list of values
function meanAndSem(values) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, sem: 0 };
  const variance = values.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1);
  return { mean, sem: Math.sqrt(variance) / Math.sqrt(n) };
}

function buildCompareChart(canvas, metric, group, groupKey) {
  const datasets = [];

  group.categories.forEach((cat) => {
    const hex = cssVar(group.colorVar[cat]);
    const clips = DATA.clips.filter((c) => c[groupKey] === cat);

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
        order: 3,
      });
    });

    // per-frame mean +/- 1 SEM
    const byFrame = {};
    clips.forEach((clip) => {
      clip.trajectory.forEach((t) => {
        if (t[metric] === null) return;
        (byFrame[t.rel_frame] = byFrame[t.rel_frame] || []).push(t[metric]);
      });
    });
    const frames = Object.keys(byFrame).map(Number).sort((a, b) => a - b);
    const meanPts = [];
    const upperPts = [];
    const lowerPts = [];
    frames.forEach((rf) => {
      const { mean, sem } = meanAndSem(byFrame[rf]);
      meanPts.push({ x: rf, y: mean });
      upperPts.push({ x: rf, y: mean + sem });
      lowerPts.push({ x: rf, y: mean - sem });
    });

    // uncertainty band (+/-1 SEM) — a wide, overlapping band between two
    // groups' bands is the honest visual answer to "is this gap real?"
    const lowerIndex = datasets.length;
    datasets.push({
      data: lowerPts,
      borderColor: "transparent",
      pointRadius: 0,
      tension: 0.15,
      spanGaps: true,
      fill: false,
      isIndividual: true,
      order: 4,
    });
    datasets.push({
      data: upperPts,
      borderColor: "transparent",
      backgroundColor: hexToRgba(hex, 0.14),
      pointRadius: 0,
      tension: 0.15,
      spanGaps: true,
      fill: lowerIndex,
      isIndividual: true,
      order: 4,
    });

    // group mean, emphasized
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
  chart.canvas.style.height = "320px";
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
    browseIndex = 0;
    renderClipBrowser();
  });
  wrap.querySelector("#f-result").addEventListener("change", (e) => {
    clipFilter.result = e.target.value;
    browseIndex = 0;
    renderClipBrowser();
  });
}

function filteredClips() {
  return DATA.clips.filter((c) => {
    if (clipFilter.shotType !== "all" && c.shot_type_group !== clipFilter.shotType) return false;
    if (clipFilter.result !== "all" && c.result_en !== clipFilter.result) return false;
    return true;
  });
}

function renderClipBrowser() {
  destroyBrowserCharts();
  const clips = filteredClips();
  const container = document.getElementById("clip-browser");
  const countEl = document.getElementById("clip-count");
  if (countEl) countEl.textContent = `${clips.length} shots`;

  if (!clips.length) {
    container.innerHTML = `<div class="empty-note">No shots match these filters.</div>`;
    return;
  }
  browseIndex = Math.max(0, Math.min(browseIndex, clips.length - 1));
  const clip = clips[browseIndex];
  const { opponent, date, quarter } = parseGameId(clip.game_id);

  container.innerHTML = `
    <div class="browse-nav">
      <button class="browse-btn" id="browse-prev" ${browseIndex === 0 ? "disabled" : ""}>&larr; Prev</button>
      <span class="browse-pos">${browseIndex + 1} / ${clips.length}</span>
      <button class="browse-btn" id="browse-next" ${browseIndex === clips.length - 1 ? "disabled" : ""}>Next &rarr;</button>
      <span class="browse-hint">use &larr; &rarr; keys to browse</span>
    </div>
    <div class="detail-layout">
      <div class="video-card">
        <video src="../${clip.video_url}" controls autoplay muted loop playsinline></video>
        <div class="game-info-panel">
          <div class="gi-row"><span class="gi-k">Opponent</span><span class="gi-v">${escapeHtml(opponent)}</span></div>
          <div class="gi-row"><span class="gi-k">Date</span><span class="gi-v">${escapeHtml(date)}</span></div>
          <div class="gi-row"><span class="gi-k">Quarter</span><span class="gi-v">${escapeHtml(quarter)}</span></div>
          <div class="gi-row"><span class="gi-k">Game clock</span><span class="gi-v">${escapeHtml(clip.game_clock)}</span></div>
          <div class="gi-row"><span class="gi-k">Distance</span><span class="gi-v">${clip.distance_ft !== null ? clip.distance_ft + " ft" : "—"}</span></div>
          <div class="gi-row"><span class="gi-k">Shot type</span><span class="gi-v">${escapeHtml(clip.shot_type)}</span></div>
          <div class="gi-row"><span class="gi-k">Result</span><span class="badge ${clip.result_en === "Make" ? "make" : "miss"}">${clip.result_en}</span></div>
          ${clip.note ? `<div class="gi-row"><span class="gi-k">Note</span><span class="gi-v">${escapeHtml(clip.note)}</span></div>` : ""}
        </div>
      </div>
      <div class="detail-chart-grid" id="browser-charts"></div>
    </div>
  `;

  buildMetricCharts(document.getElementById("browser-charts"), clip, browserCharts);

  const prevBtn = document.getElementById("browse-prev");
  const nextBtn = document.getElementById("browse-next");
  if (prevBtn) prevBtn.addEventListener("click", () => { browseIndex--; renderClipBrowser(); });
  if (nextBtn) nextBtn.addEventListener("click", () => { browseIndex++; renderClipBrowser(); });
}

function handleBrowseKeydown(e) {
  const overviewVisible = document.getElementById("view-overview").style.display !== "none";
  if (!overviewVisible) return;
  const tag = ((document.activeElement && document.activeElement.tagName) || "").toLowerCase();
  if (tag === "input" || tag === "select" || tag === "textarea") return;

  const clips = filteredClips();
  if (!clips.length) return;
  if (e.key === "ArrowLeft" && browseIndex > 0) {
    e.preventDefault();
    browseIndex--;
    renderClipBrowser();
  } else if (e.key === "ArrowRight" && browseIndex < clips.length - 1) {
    e.preventDefault();
    browseIndex++;
    renderClipBrowser();
  }
}
document.addEventListener("keydown", handleBrowseKeydown);

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// game_id looks like "DEN_0309_1" -> opponent DEN, date 03/09, quarter Q1
function parseGameId(gameId) {
  const parts = gameId.split("_");
  const opponent = parts[0] || gameId;
  const mmdd = parts[1] || "";
  const qtrRaw = parts[2] || "";
  const date = mmdd.length === 4 ? `${mmdd.slice(0, 2)}/${mmdd.slice(2)}` : mmdd;
  const quarter = /^\d+$/.test(qtrRaw) ? `Q${qtrRaw}` : qtrRaw;
  return { opponent, date, quarter };
}

// builds the 5 metric line charts for one clip into `container`, tracking
// the Chart.js instances in `targetArray` so the caller can destroy them later
function buildMetricCharts(container, clip, targetArray) {
  container.innerHTML = "";
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
    targetArray.push(chart);
  });
}

// ------------------------------------------------------------------ detail

function renderDetail(clipId) {
  destroyCharts();
  document.getElementById("view-demo").style.display = "none";
  document.getElementById("view-demo").innerHTML = "";
  document.getElementById("portal-header").style.display = "";
  setHeaderContent("Ajay Mitchell — Shot Mechanics Portal", portalSubtitle, "");
  document.getElementById("view-overview").style.display = "none";
  const el = document.getElementById("view-detail");
  el.style.display = "block";

  const clip = DATA.clips.find((c) => c.id === clipId);
  if (!clip) {
    el.innerHTML = `<div class="empty-note">Clip not found.</div>`;
    return;
  }

  el.innerHTML = `
    <a class="back-link" href="#/portal">&larr; Back to all shots</a>
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

  buildMetricCharts(document.getElementById("detail-charts"), clip, chartInstances);
}

main();
