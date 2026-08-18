const API = "";

const state = {
  page: 1,
  pageSize: 25,
  sortBy: "shield_score",
  sortDir: "desc",
  filters: { search: "", occupation: "", account_type: "", segment: "", flagged_only: false, min_score: 0 },
  currentAccount: null,
};

const tierColor = { critical: "var(--risk-critical)", high: "var(--risk-high)", medium: "var(--risk-medium)", low: "var(--risk-low)" };

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

async function loadMetrics() {
  const m = await fetch(`${API}/api/metrics`).then(r => r.json());
  const kpiRow = document.getElementById("kpiRow");
  kpiRow.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-value">${m.n_accounts.toLocaleString()}</div>
      <div class="kpi-label">Accounts scored</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value" style="color:var(--risk-critical)">${m.flagged_count.toLocaleString()}</div>
      <div class="kpi-label">Flagged for review</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">${m.score_buckets["critical (750-1000)"].toLocaleString()}</div>
      <div class="kpi-label">Critical tier (≥750)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">${fmtPct(m.oof_auc)}</div>
      <div class="kpi-label">Model AUC (proxy labels)</div>
      <div class="kpi-caveat">⚠ No confirmed fraud ground-truth exists in this dataset. AUC reflects fit to rule-derived proxy labels only.</div>
    </div>
  `;
  document.getElementById("modalText").innerText = m.disclaimer;
}

async function loadFilters() {
  const f = await fetch(`${API}/api/filters`).then(r => r.json());
  const occSel = document.getElementById("occupationFilter");
  occSel.innerHTML = `<option value="">All occupations</option>` + f.occupations.map(o => `<option value="${o}">${o}</option>`).join("");
  const typeSel = document.getElementById("accountTypeFilter");
  typeSel.innerHTML = `<option value="">All account types</option>` + f.account_types.map(o => `<option value="${o}">${o}</option>`).join("");
  const segSel = document.getElementById("segmentFilter");
  segSel.innerHTML = `<option value="">All segments</option>` + f.segments.map(o => `<option value="${o}">${o}</option>`).join("");
}

function scoreBarColor(tier) { return tierColor[tier]; }

async function loadQueue() {
  const params = new URLSearchParams({
    page: state.page, page_size: state.pageSize,
    sort_by: state.sortBy, sort_dir: state.sortDir,
    min_score: state.filters.min_score, max_score: 1000,
    flagged_only: state.filters.flagged_only,
  });
  if (state.filters.search) params.set("search", state.filters.search);
  if (state.filters.occupation) params.set("occupation", state.filters.occupation);
  if (state.filters.account_type) params.set("account_type", state.filters.account_type);
  if (state.filters.segment) params.set("segment", state.filters.segment);

  const data = await fetch(`${API}/api/accounts?${params}`).then(r => r.json());

  document.getElementById("resultCount").innerText = `${data.total.toLocaleString()} accounts`;

  const tbody = document.getElementById("queueBody");
  tbody.innerHTML = data.results.map(a => `
    <tr class="${a.risk_tier === 'critical' ? 'row-critical' : ''}" data-id="${a.account_id}">
      <td class="mono">ACC-${a.account_id}</td>
      <td>${a.account_type}</td>
      <td>${a.occupation}</td>
      <td class="mono">${a.age ?? "—"}</td>
      <td class="mono">${a.iso_score.toFixed(2)}</td>
      <td class="mono">${a.xgb_score.toFixed(3)}</td>
      <td>
        <div class="score-bar-wrap">
          <span class="mono">${a.shield_score.toFixed(0)}</span>
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:${a.shield_score/10}%; background:${scoreBarColor(a.risk_tier)}"></div>
          </div>
        </div>
      </td>
      <td><span class="tier-badge tier-${a.risk_tier}">${a.risk_tier}</span></td>
    </tr>
  `).join("");

  tbody.querySelectorAll("tr").forEach(row => {
    row.addEventListener("click", () => openDrawer(parseInt(row.dataset.id)));
  });

  renderPagination(data.total);
}

function renderPagination(total) {
  const pages = Math.ceil(total / state.pageSize);
  const el = document.getElementById("pagination");
  const cur = state.page;
  let html = "";
  const range = [1, 2, cur - 1, cur, cur + 1, pages - 1, pages].filter(p => p >= 1 && p <= pages);
  const uniq = [...new Set(range)].sort((a, b) => a - b);
  let last = 0;
  uniq.forEach(p => {
    if (p - last > 1) html += `<span style="color:var(--text-tertiary)">…</span>`;
    html += `<button class="page-btn ${p === cur ? 'active' : ''}" data-p="${p}">${p}</button>`;
    last = p;
  });
  el.innerHTML = html;
  el.querySelectorAll(".page-btn").forEach(b => b.addEventListener("click", () => {
    state.page = parseInt(b.dataset.p);
    loadQueue();
  }));
}

// ---------------- Drawer ----------------

function humanizeFeature(name) {
  if (name.includes("_")) {
    const parts = name.split("_");
    return parts.join(" ");
  }
  return `Engineered signal ${name}`;
}

async function openDrawer(id) {
  state.currentAccount = id;
  const a = await fetch(`${API}/api/accounts/${id}`).then(r => r.json());

  document.getElementById("drawerId").innerText = `ACC-${a.account_id}`;
  document.getElementById("drawerTier").innerHTML = `<span class="tier-badge tier-${a.risk_tier}">${a.risk_tier}</span>`;
  document.getElementById("drawerScore").innerText = a.shield_score.toFixed(0);
  document.getElementById("drawerScore").style.color = tierColor[a.risk_tier];
  document.getElementById("drawerNarrative").innerText = a.narrative;

  document.getElementById("drawerMeta").innerHTML = `
    <div class="meta-item"><span>Account type</span><span>${a.account_type}</span></div>
    <div class="meta-item"><span>Occupation</span><span>${a.occupation}</span></div>
    <div class="meta-item"><span>Segment</span><span>${a.segment}</span></div>
    <div class="meta-item"><span>Age</span><span>${a.age ?? "—"}</span></div>
    <div class="meta-item"><span>Gender</span><span>${a.gender}</span></div>
    <div class="meta-item"><span>Business type</span><span>${a.business_type}</span></div>
    <div class="meta-item"><span>Tenure</span><span>${a.tenure_bucket}</span></div>
    <div class="meta-item"><span>Opened</span><span>${a.account_open_date}</span></div>
    <div class="meta-item"><span>Branch</span><span>${a.branch_code}</span></div>
    <div class="meta-item"><span>Anomaly score</span><span>${a.iso_score.toFixed(2)}</span></div>
  `;

  const maxAbs = Math.max(...a.top_factors.map(f => Math.abs(f.impact)), 0.01);
  document.getElementById("shapChart").innerHTML = a.top_factors.map(f => {
    const pct = (Math.abs(f.impact) / maxAbs) * 100;
    const color = f.direction === "increases_risk" ? "var(--risk-critical)" : "var(--signal)";
    const justify = f.direction === "increases_risk" ? "left:50%;" : `right:50%;`;
    return `
      <div class="shap-row">
        <div class="shap-feature">${humanizeFeature(f.feature)}</div>
        <div class="shap-bar-area">
          <div class="shap-bar" style="${justify} width:${pct/2}%; background:${color}"></div>
        </div>
        <div class="shap-val">${f.impact > 0 ? "+" : ""}${f.impact.toFixed(2)}</div>
      </div>`;
  }).join("");

  const net = await fetch(`${API}/api/accounts/${id}/network`).then(r => r.json());
  document.getElementById("networkNote").innerText = net.note;
  renderNetwork(net);

  loadAuditTrail(id);

  document.getElementById("drawer").classList.add("open");
  document.getElementById("drawerBackdrop").classList.add("open");
}

function closeDrawer() {
  document.getElementById("drawer").classList.remove("open");
  document.getElementById("drawerBackdrop").classList.remove("open");
}

function renderNetwork(net) {
  const svg = d3.select("#networkGraph");
  svg.selectAll("*").remove();
  const width = document.getElementById("networkGraph").clientWidth || 400;
  const height = 260;

  // radar sweep signature behind the graph
  const defs = svg.append("defs");
  const grad = defs.append("radialGradient").attr("id", "radarGrad");
  grad.append("stop").attr("offset", "0%").attr("stop-color", "#3ed9a0").attr("stop-opacity", 0.12);
  grad.append("stop").attr("offset", "100%").attr("stop-color", "#3ed9a0").attr("stop-opacity", 0);
  svg.append("circle").attr("cx", width/2).attr("cy", height/2).attr("r", Math.min(width,height)/2)
    .attr("fill", "url(#radarGrad)");
  [0.33, 0.66, 1].forEach(f => {
    svg.append("circle").attr("cx", width/2).attr("cy", height/2)
      .attr("r", (Math.min(width,height)/2) * f)
      .attr("fill", "none").attr("stroke", "#232938").attr("stroke-width", 1);
  });

  const sim = d3.forceSimulation(net.nodes)
    .force("link", d3.forceLink(net.edges).id(d => d.id).distance(70).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-120))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collide", d3.forceCollide(18));

  const link = svg.append("g").selectAll("line")
    .data(net.edges).enter().append("line")
    .attr("stroke", "#3ed9a0").attr("stroke-opacity", d => 0.15 + d.weight * 0.25)
    .attr("stroke-width", d => 1 + d.weight * 1.5);

  const node = svg.append("g").selectAll("g")
    .data(net.nodes).enter().append("g");

  node.append("circle")
    .attr("r", d => d.is_focus ? 12 : 8)
    .attr("fill", d => tierColor[d.risk_tier].replace("var(", "").replace(")", ""))
    .attr("fill", d => ({critical:"#ff5c5c",high:"#ff9f43",medium:"#f5c542",low:"#3ed9a0"}[d.risk_tier]))
    .attr("stroke", d => d.is_focus ? "#fff" : "none")
    .attr("stroke-width", d => d.is_focus ? 1.5 : 0);

  node.append("text")
    .text(d => d.label)
    .attr("font-size", 9)
    .attr("font-family", "IBM Plex Mono, monospace")
    .attr("fill", "#8890a0")
    .attr("dy", -14)
    .attr("text-anchor", "middle");

  sim.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });
}

async function loadAuditTrail(id) {
  const log = await fetch(`${API}/api/audit-trail?account_id=${id}`).then(r => r.json());
  const el = document.getElementById("auditList");
  if (!log.length) { el.innerHTML = ""; return; }
  el.innerHTML = log.map(r => {
    const t = new Date(r.timestamp * 1000).toLocaleString();
    return `<div class="audit-item">[${t}] ${r.action.toUpperCase()} by ${r.investigator}${r.note ? " — " + r.note : ""}</div>`;
  }).join("");
}

function showToast(msg) {
  const t = document.getElementById("toast");
  t.innerText = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2200);
}

// ---------------- Event wiring ----------------

document.getElementById("drawerClose").addEventListener("click", closeDrawer);
document.getElementById("drawerBackdrop").addEventListener("click", closeDrawer);

document.getElementById("disclaimerBtn").addEventListener("click", () => {
  document.getElementById("modalBackdrop").classList.add("open");
});
document.getElementById("modalClose").addEventListener("click", () => {
  document.getElementById("modalBackdrop").classList.remove("open");
});

document.querySelectorAll(".action-row .btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    if (!state.currentAccount) return;
    const note = document.getElementById("actionNote").value;
    const action = btn.dataset.action;
    await fetch(`${API}/api/accounts/${state.currentAccount}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, note }),
    });
    document.getElementById("actionNote").value = "";
    showToast(`Account ${state.currentAccount} — ${action} logged`);
    loadAuditTrail(state.currentAccount);
  });
});

let searchTimer;
document.getElementById("searchInput").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.filters.search = e.target.value;
    state.page = 1;
    loadQueue();
  }, 300);
});
["occupationFilter", "accountTypeFilter", "segmentFilter"].forEach(id => {
  document.getElementById(id).addEventListener("change", (e) => {
    const key = { occupationFilter: "occupation", accountTypeFilter: "account_type", segmentFilter: "segment" }[id];
    state.filters[key] = e.target.value;
    state.page = 1;
    loadQueue();
  });
});
document.getElementById("flaggedOnly").addEventListener("change", (e) => {
  state.filters.flagged_only = e.target.checked;
  state.page = 1;
  loadQueue();
});
document.getElementById("minScore").addEventListener("input", (e) => {
  document.getElementById("minScoreVal").innerText = e.target.value;
});
document.getElementById("minScore").addEventListener("change", (e) => {
  state.filters.min_score = parseInt(e.target.value);
  state.page = 1;
  loadQueue();
});

document.querySelectorAll(".queue-table thead th[data-sort]").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (state.sortBy === key) state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
    else { state.sortBy = key; state.sortDir = "desc"; }
    loadQueue();
  });
});

// live "scanning" label cycling to sell the real-time feel
const scanMsgs = ["SCANNING PORTFOLIO", "RE-SCORING QUEUE", "MONITORING LIVE", "SHIELD ACTIVE"];
let scanI = 0;
setInterval(() => {
  scanI = (scanI + 1) % scanMsgs.length;
  document.getElementById("statusLabel").innerText = scanMsgs[scanI];
}, 4000);

// ---------------- Init ----------------

loadMetrics();
loadFilters();
loadQueue();
