/* kgirl console — one page, one live stream, every module. */
"use strict";
const $ = (id) => document.getElementById(id);
const S = { snap: null, convId: null, chatJob: null, jobs: {}, feedPaused: false,
            matrix: null, mcpTool: null, approvals: [], applyConv: null, drawerOpen: false,
            rescueBusy: false };
const STUCK_S = 120;   // a running job with no events for this long is flagged stuck

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function toast(msg, err) {
  const t = $("toast"); t.textContent = msg; t.className = err ? "err" : "";
  clearTimeout(t._h); t._h = setTimeout(() => t.className = "hidden", err ? 6000 : 3200);
}

// A persistent error banner: stays until dismissed, unlike a toast. One banner per distinct message.
function banner(msg) {
  const box = $("banners");
  for (const el of box.children) if (el.dataset.msg === msg) return;
  const div = document.createElement("div");
  div.className = "banner"; div.dataset.msg = msg;
  div.setAttribute("role", "alert");
  const span = document.createElement("span");
  span.textContent = msg;
  const btn = document.createElement("button");
  btn.className = "btn ghost"; btn.textContent = "dismiss";
  btn.onclick = () => div.remove();
  div.append(span, btn);
  box.prepend(div);
  while (box.children.length > 6) box.lastChild.remove();
}

const TOKEN = document.querySelector('meta[name="kgirl-token"]')?.content || "";

async function api(method, path, body) {
  const opt = { method, headers: { "X-Kgirl-Token": TOKEN } };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  let data = null;
  try { data = await r.json(); } catch { /* empty */ }
  if (!r.ok) {
    const msg = data?.error || `${r.status} ${r.statusText}`;
    if (method === "POST") banner(`${path}: ${msg}`);
    throw new Error(msg);
  }
  return data;
}
const get = (p) => api("GET", p);
const post = (p, b) => api("POST", p, b ?? {});

function renderContent(text) {
  const parts = String(text ?? "").split(/```/);
  let html = "";
  parts.forEach((p, i) => {
    if (i % 2 === 1) {
      const nl = p.indexOf("\n");
      const code = nl >= 0 && /^[\w+-]*$/.test(p.slice(0, nl).trim()) ? p.slice(nl + 1) : p;
      html += `<pre>${esc(code)}</pre>`;
    } else {
      html += esc(p).replace(/`([^`\n]+)`/g, "<code>$1</code>");
    }
  });
  return html;
}

function fmtTime(ts) { return ts ? new Date(ts * 1000).toLocaleTimeString() : ""; }

/* ---------------------------------------------------------------- live stream */

function connect() {
  const es = new EventSource("/api/events?token=" + encodeURIComponent(TOKEN));
  es.onopen = () => { $("conn").className = "conn on"; $("conn-text").textContent = "live"; };
  es.onerror = () => { $("conn").className = "conn off"; $("conn-text").textContent = "reconnecting"; };
  es.onmessage = (m) => { let ev; try { ev = JSON.parse(m.data); } catch { return; } dispatch(ev); };
}

function dispatch(ev) {
  feedEvent(ev);
  switch (ev.type) {
    case "hello":
      S.approvals = ev.approvals || []; showApproval();
      (ev.jobs || []).forEach(j => trackJob(j));
      break;
    case "job_start": trackJob({ id: ev.job, kind: ev.kind, label: ev.label, status: "running" }); break;
    case "job": jobEvent(ev); break;
    case "job_end": jobEnd(ev); break;
    case "approval": S.approvals.push(ev); showApproval(); break;
    case "approval_done":
      S.approvals = S.approvals.filter(a => a.id !== ev.id); showApproval(); break;
    case "workspace": toast(ev.text); break;
    case "bus": if (String(ev.topic || "").startsWith("holo.")) holoEvent(ev); break;
  }
}

function trackJob(j) {
  S.jobs[j.id] = { ...j, events: [], lastEvent: j.last_event || j.created || Date.now() / 1000 };
  const mini = $("jobs-mini");
  const pill = document.createElement("div");
  pill.className = "job-pill"; pill.id = "pill-" + j.id;
  pill.innerHTML = `<span class="spin">●</span> ${esc(j.kind)} ${esc(j.label || "").slice(0, 24)}`;
  pill.onclick = () => showJob(j.id);
  mini.prepend(pill);
  while (mini.children.length > 5) mini.lastChild.remove();
  renderTray();
}

function jobEvent(ev) {
  const inner = ev.event || {};
  const j = S.jobs[ev.job];
  if (j) { j.events.push(inner); j.lastEvent = inner.ts || Date.now() / 1000; }
  if (ev.job === S.chatJob) {
    if (inner.type === "delta") streamDelta(inner.text);
    else if (inner.type === "status" || inner.type === "queued") streamStatus(inner.text);
  }
  const card = document.querySelector(`#job-${ev.job} .events`);
  if (card && inner.type !== "delta") {
    card.insertAdjacentHTML("beforeend", `<div>${esc(inner.text || inner.type)}</div>`);
    card.scrollTop = card.scrollHeight;
  }
}

async function jobEnd(ev) {
  const j = S.jobs[ev.job];
  const pill = $("pill-" + ev.job);
  if (pill) { pill.className = "job-pill " + (ev.status === "done" ? "done" : "error");
    pill.innerHTML = `${ev.status === "done" ? "✓" : "✗"} ${esc(ev.kind)} ${esc(j?.label || "").slice(0, 24)}`; }
  let full = null;
  try { full = await get("/api/jobs/" + ev.job); } catch { /* gone */ }
  if (j && full) { Object.assign(j, full); j.lastEvent = full.last_event || j.lastEvent; }
  if (ev.status === "error") {
    toast(`${ev.kind} failed: ${ev.error}`, true);
    banner(`${ev.kind} failed: ${ev.error}`);
  }
  renderTray();
  if (S.drawerOpen) renderDrawer();
  if (ev.job === S.chatJob) { S.chatJob = null; finalizeChat(full, ev); return; }
  if (!full) return;
  const r = full.result;
  switch (ev.kind) {
    case "task": renderTask(r); break;
    case "build": renderBuild(r); break;
    case "builder": renderBuilder(r); break;
    case "write": renderWrite(r); break;
    case "matrix": renderMatrixHits(r); break;
    case "helix": renderHelixRun(r); break;
    case "holo-build": toast("holo store built"); loadHolo(); break;
    case "holo-query": renderHoloQuery(r); break;
    case "evolve": renderEvolve(r); break;
    case "mcp": renderMcpResult(r); break;
    case "index": toast("index finished"); loadRepos(); break;
  }
}

function showJob(id) {
  const j = S.jobs[id];
  if (!j) return;
  switchView("specialists");
  const box = $("task-result");
  const div = document.createElement("div");
  div.className = "job-card"; div.id = "job-" + id;
  div.innerHTML = `<div class="head"><span class="kind">${esc(j.kind)}</span><span>${esc(j.label)}</span>
    <span class="status">${esc(j.status)}</span></div><div class="events"></div>`;
  box.prepend(div);
  const evs = div.querySelector(".events");
  (j.events || []).filter(e => e.type !== "delta").forEach(e =>
    evs.insertAdjacentHTML("beforeend", `<div>${esc(e.text || e.type)}</div>`));
}

/* ---------------------------------------------------------------- job tray & drawer */

function jobStuck(j) {
  return j.status === "running" && Date.now() / 1000 - (j.lastEvent || j.created || 0) > STUCK_S;
}

function renderTray() {
  const jobs = Object.values(S.jobs);
  const running = jobs.filter(j => j.status === "running").length;
  const failed = jobs.filter(j => j.status === "error").length;
  const stuck = jobs.filter(jobStuck).length;
  $("jobs-running").textContent = running;
  $("jobs-failed").textContent = failed;
  const tray = $("jobs-tray");
  tray.classList.toggle("busy", running > 0);
  tray.classList.toggle("bad", failed > 0 || stuck > 0);
  tray.title = `${running} running · ${failed} failed` + (stuck ? ` · ${stuck} stuck` : "");
  tray.setAttribute("aria-expanded", String(S.drawerOpen));
}

// Job labels and errors can carry model-written text: everything here goes in through textContent.
function renderDrawer() {
  const list = $("jobs-drawer-list");
  list.innerHTML = "";
  const jobs = Object.values(S.jobs).sort((a, b) => (b.created || 0) - (a.created || 0)).slice(0, 50);
  if (!jobs.length) {
    const d = document.createElement("div");
    d.className = "hint"; d.textContent = "no jobs yet this session";
    list.append(d);
    return;
  }
  const now = Date.now() / 1000;
  for (const j of jobs) {
    const stuck = jobStuck(j);
    const row = document.createElement("div");
    row.className = "drawer-row " + (stuck ? "stuck" : j.status);
    const head = document.createElement("div");
    head.className = "drawer-head-row";
    const kind = document.createElement("span");
    kind.className = "kind"; kind.textContent = j.kind;
    const label = document.createElement("span");
    label.className = "label"; label.textContent = j.label || "";
    const status = document.createElement("span");
    status.className = "status"; status.textContent = stuck ? "stuck" : j.status;
    const elapsed = document.createElement("span");
    elapsed.className = "elapsed";
    elapsed.textContent = fmtAge(Math.max(0, Math.round((j.ended || now) - (j.created || now))));
    head.append(kind, label, status, elapsed);
    row.append(head);
    const last = [...(j.events || [])].reverse().find(e => e.type !== "delta" && e.text);
    if (last) {
      const le = document.createElement("div");
      le.className = "drawer-last";
      le.textContent = "last: " + String(last.text).slice(0, 120);
      row.append(le);
    }
    if (j.error) {
      const err = document.createElement("div");
      err.className = "drawer-error"; err.textContent = j.error;
      row.append(err);
    }
    row.onclick = () => { toggleDrawer(false); showJob(j.id); };
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); row.onclick(); }
    };
    list.append(row);
  }
}

function toggleDrawer(open) {
  S.drawerOpen = open === undefined ? !S.drawerOpen : open;
  $("jobs-drawer").classList.toggle("hidden", !S.drawerOpen);
  if (S.drawerOpen) {
    renderDrawer();
    get("/api/jobs").then(jobs => {          // pick up anything missed while the stream was down
      jobs.forEach(j => {
        if (!S.jobs[j.id]) trackJob(j);
        else Object.assign(S.jobs[j.id], j, { lastEvent: j.last_event || S.jobs[j.id].lastEvent });
      });
      renderTray();
      if (S.drawerOpen) renderDrawer();
    }).catch(() => { /* the stream snapshot is enough */ });
  }
  renderTray();
}

function checkStuckJobs() {
  for (const j of Object.values(S.jobs)) {
    if (jobStuck(j) && !j.stuckNotified) {
      j.stuckNotified = true;
      banner(`${j.kind} job looks stuck: no events for ${STUCK_S} s (${j.label || j.id})`);
    }
  }
  renderTray();
  if (S.drawerOpen) renderDrawer();
}

// The server compares KGIRL_CODE's startup fingerprint with the disk at most once a minute; when it
// drifts, the running code is stale and only a restart loads the new one.
function checkCodeChanged() {
  get("/api/code").then(d => {
    const el = $("code-restart");
    el.classList.toggle("hidden", !(d.available && d.changed));
    if (d.available && d.changed) el.title = `${d.root}\nstartup ${d.startup} → now ${d.current}`;
  }).catch(() => { /* the hint is optional */ });
}

/* ---------------------------------------------------------------- approvals */

function showApproval() {
  const a = S.approvals[0];
  if (!a) { $("approval").classList.add("hidden"); return; }
  $("approval-q").textContent = a.question;
  $("approval-detail").textContent = a.detail;
  $("approval").classList.remove("hidden");
  $("approval-yes").onclick = () => resolveApproval(a.id, true);
  $("approval-no").onclick = () => resolveApproval(a.id, false);
}
async function resolveApproval(id, yes) {
  $("approval").classList.add("hidden");
  try { await post("/api/approve", { id, yes }); } catch (e) { toast(e.message, true); }
}

/* ---------------------------------------------------------------- live feed */

function feedEvent(ev) {
  if (S.feedPaused) return;
  if (ev.type === "job" && ev.event?.type === "delta") return;
  const list = $("feed-list");
  const div = document.createElement("div");
  const filter = $("feed-filter").value.trim();
  let topic = ev.type, body = "";
  if (ev.type === "bus") { topic = ev.topic; body = JSON.stringify(ev.payload ?? "").slice(0, 160); }
  else if (ev.type === "job") { topic = `job:${ev.job?.slice(0, 6)}`; body = ev.event?.text || ev.event?.type || ""; }
  else { body = ev.label || ev.question || ev.text || ev.status || ""; }
  const line = `${topic} ${body}`;
  if (filter && !line.includes(filter)) return;
  div.className = "ev" + (ev.type === "bus" ? "" : " job-ev");
  div.innerHTML = `<span class="topic">${esc(topic)}</span> ${esc(body)}`;
  list.prepend(div);
  while (list.children.length > 300) list.lastChild.remove();
}

/* ---------------------------------------------------------------- navigation */

function switchView(name) {
  document.querySelectorAll(".nav[data-view]").forEach(b => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + name));
  const loaders = { chat: loadChat, specialists: loadSpecialists, memory: loadSoup, atlas: loadAtlas,
                    forge: loadForge, vitals: loadVitals, matrix: loadMatrix, helix: loadHelix,
                    holo: loadHolo, lanes: loadLanes, pocket: loadPocket,
                    rescue: () => loadRescue().catch(e => toast(e.message, true)), services: loadServices };
  loaders[name]?.();   // builder has nothing to preload; its result arrives as a job
}

/* ---------------------------------------------------------------- chat */

async function loadChat() {
  const convs = await get("/api/conversations");
  const sel = $("conv-select");
  sel.innerHTML = convs.map(c =>
    `<option value="${c.id}">#${c.id} ${esc(c.title || "(untitled)").slice(0, 44)}</option>`).join("");
  if (!S.convId || !convs.some(c => c.id === S.convId)) S.convId = convs[0]?.id ?? null;
  if (S.convId) { sel.value = S.convId; await openConv(S.convId); }
  else $("transcript").innerHTML = `<div class="hint">no conversations yet — start one</div>`;
  loadWsBadge();
}

async function openConv(id) {
  S.convId = id;
  const data = await get("/api/conversations/" + id);
  const box = $("transcript");
  box.innerHTML = "";
  const turns = data.turns || [];
  turns.forEach((t, i) => box.appendChild(turnEl(t, i === turns.length - 1)));
  box.scrollTop = box.scrollHeight;
}

function turnEl(t, isLast) {
  const div = document.createElement("div");
  div.className = "turn " + (t.role === "user" ? "user" : t.role === "tool" ? "tool" : "assistant");
  const mem = (t.recalled || []).map(id => `<span class="memchip">mem #${id}</span>`).join("");
  div.innerHTML = `<div class="meta">${esc(t.role)} · ${fmtTime(t.ts)} ${mem}</div>
    <div class="content">${renderContent(t.content)}</div>`;
  // Rules-check meta arrives with the mind-rules lane; older turns have none of it and show nothing.
  const rmeta = t.meta || {};
  const hits = Array.isArray(rmeta.rules_hits) ? rmeta.rules_hits : [];
  const left = Array.isArray(rmeta.rules_left) ? rmeta.rules_left : [];
  if (t.role === "assistant" && (hits.length || left.length)) {
    const chips = document.createElement("div");
    chips.className = "chips rules-chips";
    for (const [names, bad] of [[hits, false], [left, true]]) {
      if (!names.length) continue;
      const chip = document.createElement("span");
      chip.className = "chip" + (bad ? " verdict-no" : "");
      chip.textContent = bad ? `rules check: still breaks ${names.join(", ")}`
                             : `rules check: revised (${names.join(", ")})`;
      chips.append(chip);
    }
    div.appendChild(chips);
  }
  if (isLast && t.role === "assistant") {
    const fb = document.createElement("div");
    fb.className = "feedback";
    fb.innerHTML = `<button data-v="positive">good</button><button data-v="negative">bad</button>`;
    fb.querySelectorAll("button").forEach(b => b.onclick = async () => {
      try {
        const r = await post("/api/feedback", { conv_id: S.convId, verdict: b.dataset.v });
        fb.querySelectorAll("button").forEach(x => x.className = "");
        b.className = b.dataset.v === "positive" ? "picked-good" : "picked-bad";
        toast(r.text);
      } catch (e) { toast(e.message, true); }
    });
    div.appendChild(fb);
  }
  return div;
}

async function sendChat() {
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text || S.chatJob) return;
  if (!S.convId) {
    const r = await post("/api/conversations", { title: text.slice(0, 40) });
    S.convId = r.id; await loadChat();
  }
  input.value = "";
  $("transcript").appendChild(Object.assign(document.createElement("div"),
    { className: "turn user", innerHTML: `<div class="content">${esc(text)}</div>` }));
  const stream = document.createElement("div");
  stream.className = "turn assistant streaming"; stream.id = "stream-turn";
  stream.innerHTML = `<div class="meta">kgirl</div><div class="content"></div><div class="chips"></div>`;
  $("transcript").appendChild(stream);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  S.streamRaw = "";
  $("chat-send").disabled = true;
  try {
    const r = await post("/api/chat", { conv_id: S.convId, text });
    S.chatJob = r.job;
  } catch (e) { toast(e.message, true); stream.remove(); $("chat-send").disabled = false; }
}

function streamDelta(text) {
  S.streamRaw = (S.streamRaw || "") + text;
  const el = document.querySelector("#stream-turn .content");
  if (el) { el.innerHTML = renderContent(S.streamRaw); $("transcript").scrollTop = $("transcript").scrollHeight; }
}

function streamStatus(text) {
  const chips = document.querySelector("#stream-turn .chips");
  if (chips) chips.insertAdjacentHTML("beforeend", `<span class="toolchip">${esc(text).slice(0, 90)}</span>`);
}

async function finalizeChat(full, ev) {
  $("chat-send").disabled = false;
  $("stream-turn")?.remove();
  if (ev.status === "done" && full?.result?.tools?.some(t => (t[0] || t.name) === "task")) {
    S.applyPending = { conv_id: S.convId }; $("apply-bar").classList.remove("hidden");
  }
  if (S.convId) await openConv(S.convId);
}

async function loadWsBadge() {
  try {
    const ws = await get("/api/workspace");
    $("ws-badge").textContent = ws.attached ? `workspace: ${ws.root}` : "no workspace";
    $("ws-auto").textContent = "auto: " + (ws.auto_edit ? "on" : "off");
  } catch { /* state not ready */ }
}

function initChatControls() {
  $("conv-select").onchange = (e) => openConv(Number(e.target.value));
  $("conv-new").onclick = async () => {
    const r = await post("/api/conversations", { title: "" });
    S.convId = r.id; await loadChat(); $("chat-input").focus();
  };
  $("chat-send").onclick = sendChat;
  $("chat-input").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  };
  $("apply-btn").onclick = async () => {
    try {
      const r = await post("/api/apply", S.applyPending || {});
      toast(r.text); $("apply-bar").classList.add("hidden"); S.applyPending = null;
    } catch (e) { toast(e.message, true); }
  };
  $("apply-dismiss").onclick = () => { $("apply-bar").classList.add("hidden"); S.applyPending = null; };
  $("ws-diff").onclick = async () => {
    const ws = await get("/api/workspace");
    showSource("workspace diff", ws.diff || "kgirl has changed nothing in this session");
  };
  $("ws-undo").onclick = async () => { toast((await post("/api/workspace/undo")).text); };
  $("ws-auto").onclick = async () => {
    const cur = $("ws-auto").textContent.endsWith("on");
    const r = await post("/api/workspace/auto", { on: !cur });
    $("ws-auto").textContent = "auto: " + (r.auto_edit ? "on" : "off");
  };
  $("ws-verify").onclick = async () => {
    const cmd = prompt("verifier command (proves a swarm's change):", "python3 -m pytest -q");
    if (cmd === null) return;
    const r = await post("/api/workspace/verify", { command: cmd });
    toast(r.verify ? `verifier: ${r.verify}` : "verifier cleared");
  };
}

function showSource(title, text) {
  $("source-title").textContent = title;
  $("source-text").textContent = text;
  $("source-modal").classList.remove("hidden");
}

/* ---------------------------------------------------------------- specialists */

async function loadSpecialists() {
  const repos = await get("/api/repos");
  $("task-repo").innerHTML = `<option value="">workspace / default</option>` +
    repos.map(r => `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("");
  const jobs = await get("/api/jobs");
  $("jobs-list").innerHTML = jobs.length ? jobs.map(j =>
    `<div class="job-card"><div class="head"><span class="kind">${esc(j.kind)}</span>
     <span>${esc(j.label)}</span><span class="status ${j.status === "done" ? "verdict-yes" : j.status === "error" ? "verdict-no" : "verdict-run"}">${esc(j.status)}</span>
     <button class="btn ghost" onclick="showJob('${j.id}')">open</button></div></div>`).join("")
    : `<div class="hint">no operations yet this session</div>`;
}

async function dispatchTask() {
  const goal = $("task-goal").value.trim();
  if (!goal) return toast("a goal is required", true);
  const body = { goal };
  if ($("task-repo").value) body.repo = $("task-repo").value;
  if ($("task-verify").value.trim()) body.verify = $("task-verify").value.trim();
  if ($("task-size").value) body.size = Number($("task-size").value);
  try {
    const r = await post("/api/task", body);
    toast("swarm dispatched"); showJob(r.job);
  } catch (e) { toast(e.message, true); }
}

function renderTask(r) {
  switchView("specialists");
  const box = $("task-result");
  if (!r) return;
  let html = `<div class="panel"><h2>swarm result</h2>`;
  const trajs = r.trajectories || [];
  if (trajs.length) {
    html += `<table class="traj-table"><tr><th>specialist</th><th>outcome</th><th>verified</th><th>steps</th>
      <th>llm calls</th><th>tokens</th><th>seconds</th></tr>`;
    for (const t of trajs) {
      const win = r.winner && t.jev_id === r.winner.jev_id;
      const v = t.verified === true ? `<span class="verdict-yes">yes</span>`
        : t.verified === false ? `<span class="verdict-no">no</span>` : `<span class="verdict-run">—</span>`;
      html += `<tr class="${win ? "winner" : ""}"><td>${esc(t.jev_id)}${win ? " ★" : ""}</td>
        <td>${esc(t.outcome)}</td><td>${v}</td><td>${(t.steps || []).length}</td>
        <td>${t.llm_calls ?? "—"}</td><td>${t.prompt_tokens ?? "—"}</td><td>${Math.round(t.elapsed || 0)}</td></tr>`;
    }
    html += `</table>`;
  }
  if (r.files?.length) {
    html += `<p class="hint">verified diff against ${esc(r.root)} — read it, then apply</p>`;
    for (const f of r.files) html += diffHtml(f.path, f.diff);
    html += `<div class="row"><button class="btn accent" id="task-apply">apply to repo</button></div>`;
  } else {
    html += `<pre class="render">${esc(r.text || "no agent produced a verified change")}</pre>`;
  }
  html += `</div>`;
  box.insertAdjacentHTML("afterbegin", html);
  $("task-apply")?.addEventListener("click", async () => {
    try { toast((await post("/api/apply", {})).text); } catch (e) { toast(e.message, true); }
  });
}

function diffHtml(path, diff) {
  const lines = String(diff || "").split("\n").map(l => {
    const cls = l.startsWith("+") && !l.startsWith("+++") ? "add"
      : l.startsWith("-") && !l.startsWith("---") ? "del"
      : l.startsWith("@@") ? "hunk" : "ctx";
    return `<span class="${cls}">${esc(l)}</span>`;
  }).join("");
  return `<div class="diff-file"><div class="path">${esc(path)}</div><pre>${lines}</pre></div>`;
}

/* ---------------------------------------------------------------- memory */

async function loadSoup() {
  const p = new URLSearchParams();
  if ($("soup-q").value.trim()) p.set("q", $("soup-q").value.trim());
  if ($("soup-kind").value) p.set("kind", $("soup-kind").value);
  if ($("soup-status").value) p.set("status", $("soup-status").value);
  p.set("limit", "200");
  const frags = await get("/api/soup?" + p);
  $("soup-list").innerHTML = frags.length ? frags.map(f => `
    <div class="frag ${esc(f.status)}">
      <span class="id">#${f.id}</span><span class="kind ${esc(f.kind)}">${esc(f.kind)}</span>
      <span class="text">${esc(f.text)}<span class="src">${esc(f.source || "")} · support ${f.support ?? 0}
        · ${f.wins ?? 0}w/${f.losses ?? 0}l</span></span>
      <span class="nums"><span class="u">u=${f.utility}</span><br>lcb ${f.bound}</span>
      <span class="acts">
        ${f.status !== "active" ? `<button onclick="soupStatus(${f.id},'active')">promote</button>` : ""}
        ${f.status !== "retired" ? `<button onclick="soupStatus(${f.id},'retired')">retire</button>` : ""}
        <button onclick="soupCredit(${f.id},true)">+</button><button onclick="soupCredit(${f.id},false)">−</button>
      </span>
    </div>`).join("") : `<div class="hint">no fragments match</div>`;
  loadRules();
}

async function soupStatus(id, status) {
  try { await post("/api/soup/status", { id, status, reason: "console" }); toast(`#${id} → ${status}`); loadSoup(); }
  catch (e) { toast(e.message, true); }
}
async function soupCredit(id, success) {
  try { await post("/api/soup/credit", { ids: [id], success }); loadSoup(); }
  catch (e) { toast(e.message, true); }
}

async function loadLedger() {
  const rows = await get("/api/soup/ledger?limit=150");
  $("ledger").innerHTML = rows.map(r =>
    `<div class="ledger-row"><b>${esc(r.op)}</b> #${r.fragment_id ?? "—"} · ${esc(r.detail || "")}
     · ${fmtTime(r.ts)}</div>`).join("") || `<div class="hint">ledger is empty</div>`;
}

// The rules panel lights up only when the mind-rules lane is in this codebase; preference text goes
// in through textContent.
async function loadRules() {
  const panel = $("rules-panel"), badge = $("rules-badge");
  let d;
  try { d = await get("/api/rules"); } catch { panel.classList.add("hidden"); return; }
  if (!d.available) { panel.classList.add("hidden"); badge.classList.add("hidden"); return; }
  badge.title = `${(d.rules || []).length} standing rules active`;
  badge.classList.remove("hidden");
  const list = $("rules-list");
  list.innerHTML = "";
  if (!(d.rules || []).length) {
    const div = document.createElement("div");
    div.className = "hint"; div.textContent = "no active preference is phrased as a rule yet";
    list.append(div);
  }
  for (const r of d.rules || []) {
    const row = document.createElement("div");
    row.className = "ledger-row";
    const id = document.createElement("b");
    id.textContent = "#" + r.id;
    const scope = document.createElement("span");
    scope.className = "memchip"; scope.textContent = r.scope || "*";
    const txt = document.createElement("span");
    txt.textContent = " " + r.text;
    row.append(id, scope, txt);
    list.append(row);
  }
  panel.classList.remove("hidden");
}

/* ---------------------------------------------------------------- atlas */

async function loadAtlas() { await loadRepos(); }

async function loadRepos() {
  const repos = await get("/api/repos");
  $("repos-grid").innerHTML = repos.map(r => `
    <div class="repo-card"><div class="name">${esc(r.name)}</div>
      <div class="stats">${r.nfiles ?? 0} files · ${r.nsymbols ?? 0} symbols</div>
      <div class="head">${esc((r.head || "").slice(0, 10))}</div></div>`).join("")
    || `<div class="hint">no repos indexed — add one above</div>`;
}

async function atlasSearch() {
  const q = $("atlas-q").value.trim();
  if (!q) return;
  const box = $("atlas-results");
  box.innerHTML = `<div class="hint">searching…</div>`;
  try {
    const r = await get("/api/search?q=" + encodeURIComponent(q));
    let html = "";
    if (r.chunks?.length) {
      html += `<h2>hybrid chunks (keyword + matrix)</h2>` + r.chunks.map(h => `
        <div class="hit" onclick="openSource('${esc(h.repo)}','${esc(h.path)}',${h.start_line || 1})">
          <span class="score">${(h.score ?? 0).toFixed(3)}</span>
          <div class="cite">${esc(h.cite || `${h.repo}:${h.path}`)}</div>
          <div class="sym">${esc(h.symbol || h.kind || "")}</div>
          <div class="snippet">${esc(h.text || "")}</div></div>`).join("");
    }
    if (r.symbols?.length) {
      html += `<h2>symbols</h2>` + r.symbols.map(h => `
        <div class="hit" onclick="openSource('${esc(h.repo)}','${esc(h.path)}',${h.line || 1})">
          <span class="score">${(h.score ?? 0).toFixed(2)}</span>
          <div class="cite">${esc(h.repo)}:${esc(h.path)}:${h.line ?? "?"}</div>
          <div class="sym">${esc(h.qualname || h.signature || "")}</div>
          <div class="snippet">${esc(h.doc || "")}</div></div>`).join("");
    }
    box.innerHTML = html || `<div class="hint">no matches</div>`;
  } catch (e) { box.innerHTML = ""; toast(e.message, true); }
}

async function openSource(repo, path, start) {
  try {
    const r = await get(`/api/source?repo=${encodeURIComponent(repo)}&path=${encodeURIComponent(path)}&start=${start}&end=${start + 119}`);
    showSource(`${repo}:${path}`, r.text);
  } catch (e) { toast(e.message, true); }
}

/* ---------------------------------------------------------------- forge */

async function loadForge() { await loadSkills(); }

function renderBuild(r) {
  switchView("forge");
  if (!r) return;
  const box = $("build-result");
  box.innerHTML = `
    <div class="row"><span class="confidence">confidence ${(r.confidence ?? 0).toFixed(2)}</span>
      <span class="hint">group ${r.group_size ?? 0} · passed ${r.passed ?? 0}/${r.credible ?? 0} credible tests</span></div>
    <div class="code-tabs"><button class="active" data-t="code">module</button><button data-t="tests">tests</button>
      <button data-t="notes">report</button></div>
    <pre class="render" id="build-code">${esc(r.code || "(no module)")}</pre>
    <pre class="render hidden" id="build-tests">${esc(r.tests || "")}</pre>
    <pre class="render hidden" id="build-notes">${esc((r.notes || []).join("\n"))}</pre>`;
  box.querySelectorAll(".code-tabs button").forEach(b => b.onclick = () => {
    box.querySelectorAll(".code-tabs button").forEach(x => x.classList.toggle("active", x === b));
    ["code", "tests", "notes"].forEach(t => $("build-" + t).classList.toggle("hidden", t !== b.dataset.t));
  });
  $("build-accept").classList.toggle("hidden", !r.code);
}

function renderWrite(r) {
  switchView("forge");
  $("write-result").innerHTML = r?.markdown ? `<pre class="render">${esc(r.markdown)}</pre>` : "";
}

function renderEvolve(r) {
  $("evolve-result").innerHTML = r?.render ? `<pre class="render">${esc(r.render)}</pre>` : "";
  loadSkills();
}

async function loadSkills() {
  const skills = await get("/api/skills");
  $("skills-list").innerHTML = skills.length ? skills.map(f => `
    <div class="frag ${esc(f.status)}"><span class="id">#${f.id}</span><span class="kind skill">skill</span>
      <span class="text">${esc(f.text)}</span>
      <span class="nums"><span class="u">u=${f.utility}</span></span></div>`).join("")
    : `<div class="hint">no skills forged yet — verified trajectories become skills at support ≥ 2</div>`;
}

/* ---------------------------------------------------------------- builder (read-only candidate matrix) */

// explain/summary are feature-detected server-side: absent on older builders, the panel then stays hidden.
// Everything here is model-written text, so it goes in through textContent only.
function explainPanel(r) {
  const panel = document.createElement("div");
  panel.className = "panel";
  if (r.explain) {
    const det = document.createElement("details");
    det.className = "explain";
    const sum = document.createElement("summary");
    sum.textContent = "Why this winner";
    const pre = document.createElement("pre");
    pre.className = "render";
    pre.textContent = r.explain;
    det.append(sum, pre);
    panel.append(det);
  }
  const cands = r.summary?.candidates || [];
  const suiteOf = t => String(t).split(":")[0];
  const suites = [...new Set(cands.flatMap(c =>
    [...(c.passed || []), ...Object.keys(c.failed || {})].map(suiteOf)))].sort();
  if (cands.length && suites.length) {
    const h = document.createElement("h2");
    h.textContent = "candidate × suite votes (passed / failed)";
    const table = document.createElement("table");
    table.className = "traj-table";
    const head = document.createElement("tr");
    const th0 = document.createElement("th");
    th0.textContent = "#";
    head.append(th0);
    for (const s of suites) {
      const th = document.createElement("th");
      th.textContent = s;
      head.append(th);
    }
    table.append(head);
    for (const c of cands) {
      const win = c.idx === r.chosen && r.code;
      const tr = document.createElement("tr");
      if (win) tr.className = "winner";
      const label = document.createElement("td");
      label.textContent = (win ? "★ " : "") + "#" + c.idx;
      tr.append(label);
      for (const s of suites) {
        const p = (c.passed || []).filter(t => suiteOf(t) === s).length;
        const f = Object.keys(c.failed || {}).filter(t => suiteOf(t) === s).length;
        const td = document.createElement("td");
        td.textContent = f ? `${p} / ${f}` : String(p);
        tr.append(td);
      }
      table.append(tr);
    }
    panel.append(h, table);
  }
  return panel.childNodes.length ? panel : null;
}

function renderBuilder(r) {
  switchView("builder");
  if (!r) return;
  const box = $("builder-result");
  const cands = r.candidates || [];
  const rows = cands.map(c => {
    const ok = !!c.code && !c.dropped;
    const win = c.idx === r.chosen && r.code;
    const tail = c.dropped ? `dropped: ${esc(c.dropped)}` : (c.repaired ? "repaired" : "");
    return `<tr class="${win ? "winner" : ""}" data-idx="${c.idx}">
      <td>${win ? "★ " : ""}#${c.idx}</td><td>${esc(c.style || "")}</td><td>${(c.temperature ?? 0).toFixed(1)}</td>
      <td>${(c.passed || []).length}</td><td>${Object.keys(c.failed || {}).length}</td>
      <td>${ok ? '<span class="verdict-yes">ok</span>' : '<span class="verdict-no">✗</span>'}</td><td>${tail}</td></tr>`;
  }).join("");
  const disputed = Object.entries(r.disputed || {});
  box.innerHTML = `
    <div class="panel">
      <div class="row"><span class="confidence">confidence ${(r.confidence ?? 0).toFixed(2)}</span>
        <span class="hint">winner #${r.chosen ?? "—"} · group ${r.group_size ?? 0} · ${r.passed ?? 0}/${r.credible ?? 0} credible tests · sandbox ${esc(r.sandbox || "none")}</span></div>
      <h2>candidates — click a row for its code</h2>
      <table class="traj-table"><tr><th>#</th><th>style</th><th>T</th><th>passed</th><th>failed</th><th>ok</th><th>note</th></tr>${rows}</table>
      ${(r.exemplars || []).length ? `<p class="hint">recalled exemplars: ${r.exemplars.map(i => "#" + i).join(", ")}</p>` : ""}
      ${(r.notes || []).length ? `<p class="hint">${r.notes.map(esc).join(" · ")}</p>` : ""}
    </div>
    <div class="panel"><h2>winner module</h2><pre class="render">${esc(r.code || "(no usable module)")}</pre></div>
    <div class="panel"><h2>agreed tests</h2><pre class="render">${esc((r.agreed || []).join("\n\n") || r.tests || "")}</pre></div>
    ${disputed.length ? `<div class="panel"><h2>disputed — the module fails these</h2>${disputed.map(([id, out]) =>
      `<div class="diff-file"><div class="path">${esc(id)}</div><pre>${esc(out)}</pre></div>`).join("")}</div>` : ""}
    <div class="panel"><h2>suites as generated</h2><pre class="render">${esc((r.suites || []).join("\n\n"))}</pre></div>`;
  const ep = explainPanel(r);
  if (ep) box.insertBefore(ep, box.children[1] || null);
  box.querySelectorAll("tr[data-idx]").forEach(tr => tr.onclick = () => {
    const c = cands.find(x => String(x.idx) === tr.dataset.idx);
    if (c) showSource(`candidate #${c.idx} (${c.style || ""})`, c.code || `(dropped: ${c.dropped || "no code"})`);
  });
}

/* ---------------------------------------------------------------- vitals */

async function loadVitals() {
  const v = await get("/api/vitals");
  const grid = $("vitals-grid");
  if (!v.enabled) { grid.innerHTML = `<div class="hint">metabolism disabled (KGIRL_LIFE)</div>`; return; }
  const d = v.vitals || {};
  const cards = [
    ["episodes", d.episodes, ""], ["attention A", d.A, "acc"], ["cost C", d.C, ""],
    ["order O", d.O, "acc2"], ["fatigue F", d.F, ""], ["rigidity R", d.R, ""],
    ["p(agent)", d.p_agent, "acc"], ["affordable swarm", d.affordable_swarm, "acc2"],
    ["predicted success", d.predicted_next?.success, "acc"], ["predicted latency", d.predicted_next?.latency, ""],
  ];
  grid.innerHTML = cards.map(([label, val, cls]) => `
    <div class="vital"><div class="label">${label}</div>
      <div class="value ${cls}">${typeof val === "number" ? val.toFixed(3) : val ?? "—"}</div></div>`).join("");
  $("vitals-render").textContent = v.render || "";
  try {
    const plan = await get("/api/plan");
    $("plan-box").innerHTML = plan.n !== undefined ? `<pre class="render">${esc(JSON.stringify(plan, null, 2))}</pre>`
      : `<div class="hint">metabolism disabled</div>`;
  } catch { /* plan optional */ }
  loadGpu().catch(e => toast(e.message, true));
}

async function loadGpu() {
  const d = await get("/api/gpu");
  const g = d.gpu;
  let chips = "";
  if (g) {
    const heat = g.temperature >= 95;
    chips += `<span class="chip">VRAM <b>${Math.round(g.memory_used)}</b>/${Math.round(g.memory_total)} MiB</span>
      <span class="chip">util <b>${Math.round(g.utilization)}%</b></span>
      <span class="chip ${heat ? "verdict-no" : ""}">temp <b>${g.temperature.toFixed(0)}°C</b>${heat ? " HEAT" : ""}</span>`;
  } else { chips += `<span class="chip">no nvidia-smi</span>`; }
  if (d.mind_ctx != null) chips += `<span class="chip">KGIRL_MIND_CTX <b>${d.mind_ctx}</b></span>`;
  $("gpu-chips").innerHTML = chips;
  $("gpu-warnings").innerHTML = (d.warnings || []).map(w => `<div class="conflict">${esc(w)}</div>`).join("");
  const models = d.ollama?.models || [];
  $("gpu-models").innerHTML = d.ollama?.error ? `<div class="hint">${esc(d.ollama.error)}</div>`
    : models.length ? `<table class="traj-table"><tr><th>model</th><th>size</th><th>VRAM</th><th>GPU%</th><th>params</th><th>quant</th><th>ctx</th></tr>` +
      models.map(m => `<tr><td>${esc(m.name)}</td><td>${fmtBytes(m.size)}</td><td>${fmtBytes(m.size_vram)}</td>
        <td class="${m.gpu_pct < 100 ? "verdict-no" : "verdict-yes"}">${m.gpu_pct}%</td>
        <td>${esc(m.parameter_size || "—")}</td><td>${esc(m.quantization || "—")}</td><td>${m.context_length ?? "—"}</td></tr>`).join("") + "</table>"
    : `<div class="hint">no models loaded in Ollama right now</div>`;
}

/* ---------------------------------------------------------------- matrix */

function repoHue(name) {
  let h = 0;
  for (const ch of String(name)) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return h;
}

async function loadMatrix() {
  const m = await get("/api/matrix");
  S.matrix = m;
  const stats = $("matrix-stats");
  if (!m.available) { stats.innerHTML = `<span class="chip">no KGIRL_MATRIX configured</span>`; return; }
  stats.innerHTML = `<span class="chip"><b>${m.count.toLocaleString()}</b> vectors</span>
    <span class="chip"><b>${m.dim}</b> dims</span><span class="chip">${esc(m.model)}</span>
    <span class="chip">sample <b>${m.sample}</b></span>`;
  drawMatrix(m);
}

function drawMatrix(m) {
  const cv = $("matrix-canvas"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, pad = 40;
  ctx.clearRect(0, 0, W, H);
  const repos = [...new Set(m.points.map(p => p.repo))];
  for (const p of m.points) {
    const x = pad + (p.x + 1) / 2 * (W - 2 * pad);
    const y = pad + (p.y + 1) / 2 * (H - 2 * pad);
    const hue = repoHue(p.repo);
    ctx.beginPath();
    ctx.fillStyle = `hsla(${hue}, 85%, 68%, .8)`;
    ctx.shadowColor = `hsla(${hue}, 90%, 60%, .9)`;
    ctx.shadowBlur = 9;
    ctx.arc(x, y, 2.1, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
  $("matrix-legend").innerHTML = repos.slice(0, 24).map(r =>
    `<span class="lg"><span class="sw" style="background:hsl(${repoHue(r)},85%,68%)"></span>${esc(r)}</span>`).join("");
}

function renderMatrixHits(r) {
  switchView("matrix");
  if (!r) return;
  $("matrix-hits").innerHTML = `<p class="hint">coarse pass: ${r.count.toLocaleString()} vectors → shortlist ${r.shortlist} → fine int8 dot</p>` +
    (r.hits || []).map(h => `
      <div class="matrix-hit"><span class="score">${h.score}</span>
        <span><span class="where">${esc(h.repo || "?")}:${esc(h.path || "")} ${esc(h.symbol || "")}</span>
        <div class="txt">${esc(h.text || "")}</div></span></div>`).join("");
}

/* ---------------------------------------------------------------- services (MCP) */

async function loadServices() {
  const tools = await get("/api/mcp/tools");
  $("mcp-list").innerHTML = tools.map(t => `
    <div class="mcp-tool" data-name="${esc(t.name)}"><div class="name">${esc(t.name)}</div>
      <div class="desc">${esc(t.description || "")}</div></div>`).join("");
  document.querySelectorAll(".mcp-tool").forEach(el => el.onclick = () => {
    const t = tools.find(x => x.name === el.dataset.name);
    S.mcpTool = t;
    $("mcp-call-name").textContent = t.name;
    $("mcp-call-desc").textContent = t.description || "";
    const props = t.inputSchema?.properties || {};
    const skeleton = {};
    for (const k of Object.keys(props)) skeleton[k] = "";
    $("mcp-args").value = JSON.stringify(skeleton, null, 2);
    $("mcp-result").textContent = "";
    $("mcp-call").classList.remove("hidden");
  });
}

function renderMcpResult(r) {
  const text = r?.content?.map(c => c.text).join("\n") ?? JSON.stringify(r, null, 2);
  $("mcp-result").textContent = text;
  if (r?.isError) toast("MCP tool reported an error", true);
}

/* ---------------------------------------------------------------- helix */

async function loadHelix() {
  const sel = $("helix-session");
  const want = sel.value || "main";
  const d = await get("/api/helix/state?session=" + encodeURIComponent(want));
  const sessions = d.sessions?.length ? d.sessions : ["main"];
  sel.innerHTML = sessions.map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  sel.value = sessions.includes(want) ? want : sessions[0];
  const f = d.floor || {};
  $("helix-floor").innerHTML = `<span class="chip">floor <b>${f.tokens ?? "—"}</b> ${esc(f.source || "")}` +
    `${f.runs ? " · " + f.runs + " runs" : ""}</span>`;
  renderHelixState(d.state);
  loadHelixHistory(sel.value);
}

function renderHelixState(s) {
  const box = $("helix-state");
  if (!s) { box.innerHTML = `<div class="hint">no STATE yet — run a task, or ingest a handoff with the CLI</div>`; return; }
  const axes = Object.entries(s.vector || {}).map(([axis, v]) => `
    <div class="axis-row"><span class="axis-name">${esc(axis)}</span>
      <span class="axis-bar"><span style="width:${Math.round((v ?? 0) * 100)}%"></span></span>
      <span class="axis-val">${(v ?? 0).toFixed(2)}</span></div>`).join("");
  const led = s.ledger || {};
  const spent = led.spent_ext ?? (led.spent_est != null ? led.spent_est + " EST" : "—");
  const delta = s.last_delta
    ? `${esc(s.last_delta.axis)} ${s.last_delta.d > 0 ? "+" : ""}${s.last_delta.d} — “${esc(s.last_delta.evidence)}”`
    : "—";
  const conflicts = (s.open_conflicts || []).length
    ? `<div class="conflicts">${s.open_conflicts.map(c => `<div class="conflict">open: ${esc(c)}</div>`).join("")}</div>`
    : `<p class="hint">no open conflicts</p>`;
  box.innerHTML = `
    <div class="row"><span class="chip">v<b>${s.v}</b></span>
      <span class="chip">seed: ${esc(s.task_seed || "—")}</span>
      <span class="chip">B ${led.alloc ?? "—"} · spent ${spent}</span></div>
    <div class="axes">${axes}</div>
    <p class="hint">last delta: ${delta}</p>
    ${conflicts}
    <p class="hint src">${esc(s.invariants_hash || "")}</p>`;
}

async function loadHelixHistory(session) {
  const rows = await get("/api/helix/history?session=" + encodeURIComponent(session));
  const box = $("helix-history");
  box.innerHTML = rows.length ? rows.map(h => `
    <div class="ledger-row"><b>v${h.v}</b> ${esc(h.origin)}
      <span class="src">${esc((h.hash || "").slice(0, 12))}</span>
      <button class="btn ghost" data-v="${h.v}">diff</button></div>`).join("")
    : `<div class="hint">no versions yet</div>`;
  box.querySelectorAll("button[data-v]").forEach(b => b.onclick = () => {
    const h = rows.find(x => String(x.v) === b.dataset.v);
    showSource(`v${h.v} — ${h.origin}`, JSON.stringify(h.diff, null, 2));
  });
}

function renderHelixRun(r) {
  switchView("helix");
  if (!r) return;
  const res = r.result || {};
  let rec = r.render || "";
  if (res.answer && rec.startsWith(res.answer)) rec = rec.slice(res.answer.length).replace(/^\s*-{72}\s*\n?/, "");
  $("helix-run-result").innerHTML = `<div class="panel"><h2>answer</h2>
    <div class="content">${renderContent(res.answer || "(no answer)")}</div>
    <h2>flight recorder</h2><pre class="render">${esc(rec)}</pre></div>`;
  loadHelix();
}

function renderHelixAudit(r) {
  const box = $("helix-audit-result");
  if (!r) { box.innerHTML = ""; return; }
  const vcls = r.verdict === "PASS" ? "verdict-yes" : "verdict-no";
  const findings = (r.findings || []).map(f =>
    `<div class="ledger-row"><b class="${f.severity === "High" ? "verdict-no" : "verdict-run"}">${esc(f.severity)}</b> ${esc(f.detector)} · ${esc(f.evidence || "")}</div>`).join("");
  const m = r.metrics || {};
  box.innerHTML = `<div class="row"><span class="confidence ${vcls}">verdict ${esc(r.verdict)}</span>
    <span class="hint">evidence anchor rate ${m.evidence_anchor_rate ?? "—"} · delta predicate rate ${m.delta_predicate_rate ?? "—"}</span></div>
    ${findings || `<p class="hint">no findings — the log is honest by these detectors</p>`}`;
}

/* ---------------------------------------------------------------- holo */

async function loadHolo() {
  const s = await get("/api/holo/status");
  const st = s.storage || {};
  $("holo-stats").innerHTML = `
    <span class="chip"><b>${s.entities ?? 0}</b> entities</span>
    <span class="chip"><b>${s.overlaps ?? 0}</b> shared digests</span>
    <span class="chip">${((st.stored_bytes ?? 0) / 1024).toFixed(0)} KiB stored</span>
    <span class="chip ${s.chains_ok ? "" : "verdict-no"}">${s.chains_ok ? "chains ok" : "BROKEN: " + esc((s.broken || []).join(", "))}</span>
    ${s.memory ? `<span class="chip">HoloMemory prior present</span>` : ""}`;
  const types = Object.entries(s.by_type || {}).map(([t, n]) => `${esc(t)} ${n}`).join(" · ");
  $("holo-status").innerHTML = s.built
    ? `<pre class="render">${esc(JSON.stringify(s.built, null, 2))}</pre>
       <p class="hint">${types || "no entities"} · ${st.matrices ?? 0} matrices · raw ${((st.raw_bytes ?? 0) / 1024).toFixed(0)} KiB</p>`
    : `<div class="hint">not built yet — build derives the store from the chunk index (offline, no model). ${types}</div>`;
}

function renderHoloQuery(r) {
  switchView("holo");
  if (!r) return;
  const box = $("holo-hits");
  const ents = r.entities || [];
  const hasMem = ents.some(e => e.p_memory != null);
  const guidance = ents.map(e => `
    <tr><td>${esc(e.entity)}</td><td>${e.p.toFixed(3)}</td><td>${e.share.toFixed(3)}</td>
      <td>${e.cosine.toFixed(3)}</td><td>${e.hits}</td>${hasMem ? `<td>${(e.p_memory ?? 0).toFixed(3)}</td>` : ""}</tr>`).join("");
  const prior = r.prior?.used ? `prior used: w=${r.prior.weight.toFixed(3)}` : `prior off: ${esc(r.prior?.reason || "")}`;
  box.innerHTML = `
    <p class="hint">${r.ms} ms · ${(r.matrix?.count ?? 0).toLocaleString()} vectors · ${prior}</p>
    <h2>guidance — where the query landed</h2>
    <table class="traj-table"><tr><th>entity</th><th>p</th><th>share</th><th>cosine</th><th>hits</th>${hasMem ? "<th>p_mem</th>" : ""}</tr>${guidance}</table>
    <h2>hits with provenance</h2>` +
    (r.hits || []).map(h => `
      <div class="matrix-hit"><span class="score">${h.score}</span>
        <span><span class="where" ${h.repo && h.path ? `onclick="openSource('${esc(h.repo)}','${esc(h.path)}',${h.line || 1})"` : ""}>${esc(h.cite || "chunk " + h.id)}</span>
        <div class="txt">${(h.entities || []).map(e => `<span class="memchip">${esc(e)}</span>`).join("")}
          ${h.stored ? `<span class="memchip">stored</span>` : ""}
          <span class="src" data-digest="${esc(h.digest || "")}">${esc((h.digest || "").slice(0, 12))}</span></div>
        <div class="txt src">${(h.provenance || []).map(p => `${esc(p.entity)} head ${esc((p.head || "").slice(0, 10))}`).join(" · ")}</div></span></div>`).join("");
  box.querySelectorAll("[data-digest]").forEach(el => el.onclick = async () => {
    if (!el.dataset.digest) return;
    try {
      const o = await get("/api/holo/owners?digest=" + encodeURIComponent(el.dataset.digest));
      showSource("owners of " + o.digest.slice(0, 12), (o.owners || []).join("\n") || "(no owners)");
    } catch (e) { toast(e.message, true); }
  });
}

async function loadHoloOverlaps() {
  const d = await get("/api/holo/overlaps?limit=100");
  $("holo-overlap-stats").innerHTML = `<span class="chip"><b>${d.count ?? 0}</b> shared digests</span>` +
    `<span class="chip">showing ${d.shown ?? 0}</span>`;
  $("holo-overlaps").innerHTML = (d.overlaps || []).length ? d.overlaps.map(o => `
    <div class="ledger-row"><span class="src">${esc(o.digest.slice(0, 16))}…</span>
      ${(o.owners || []).map(w => `<span class="memchip">${esc(w)}</span>`).join("")}</div>`).join("")
    : `<div class="hint">no digest has two owners yet — build the store from an index with overlapping repo/kind cells</div>`;
}

async function loadHoloVerify() {
  const v = await get("/api/holo/verify");
  const rows = Object.entries(v);
  const broken = rows.filter(([, c]) => !c.ok);
  $("holo-verify-stats").innerHTML = rows.length
    ? `<span class="chip"><b>${rows.length}</b> entities</span>
       <span class="chip ${broken.length ? "verdict-no" : "verdict-yes"}">${broken.length ? broken.length + " broken" : "all chains ok"}</span>`
    : "";
  $("holo-verify").innerHTML = rows.length ? `
    <table class="traj-table"><tr><th>entity</th><th>ok</th><th>events</th><th>head</th><th>unproven</th></tr>` +
    rows.map(([name, c]) => `<tr><td>${esc(name)}</td>
      <td class="${c.ok ? "verdict-yes" : "verdict-no"}">${c.ok ? "ok" : "BROKEN"}</td>
      <td>${c.length ?? 0}</td><td class="src">${esc((c.head || "").slice(0, 12))}</td>
      <td>${(c.unproven || []).length ? `<span class="verdict-no">${(c.unproven || []).length}</span>` : "0"}</td></tr>`).join("") +
    `</table>`
    : `<div class="hint">not built yet — verification recomputes each entity's hash chain and lists matrices recorded without a ledger entry</div>`;
}

function holoEvent(ev) {
  const box = $("holo-events");
  const p = ev.payload || {};
  let body;
  if (ev.topic === "holo.query") body = `q=${JSON.stringify(p.q)} k=${p.k} ${p.ms}ms top=${p.top} prior=${p.prior}`;
  else if (ev.topic === "holo.regulate") body = `F=${p.F} O=${p.O} C=${p.C} R=${p.R}`;
  else body = JSON.stringify(p).slice(0, 160);
  const div = document.createElement("div");
  div.className = "ledger-row";
  div.innerHTML = `<b>${esc(ev.topic)}</b> ${esc(body)} · ${fmtTime(ev.ts)}`;
  box.prepend(div);
  while (box.children.length > 60) box.lastChild.remove();
}

/* ---------------------------------------------------------------- lanes (porter, read-only) */

function fmtAge(s) {
  if (s == null) return "—";
  if (s < 60) return s + "s";
  if (s < 3600) return Math.round(s / 60) + "m";
  if (s < 86400) return (s / 3600).toFixed(1) + "h";
  return (s / 86400).toFixed(1) + "d";
}

async function loadLanes() {
  const proj = $("lanes-project").value;
  const d = await get("/api/lanes" + (proj ? "?project=" + encodeURIComponent(proj) : ""));
  const sel = $("lanes-project");
  const keep = sel.value;
  sel.innerHTML = `<option value="">all projects</option>` +
    (d.projects || []).map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join("");
  sel.value = (d.projects || []).includes(keep) ? keep : "";
  const lg = d.landing;
  $("lanes-landing").innerHTML = lg ? `
    <div class="lane-claims">${(lg.worktrees || []).map(w => `
      <span class="memchip" title="${esc(w.path)}">${esc(w.name)}${w.role === "stack" ? " (stack)" : ""}:
        ${w.error ? `<span class="verdict-no">${esc(w.error)}</span>`
                  : `<b>${w.uncommitted}</b> uncommitted · ${esc(w.branch || "")}`}</span>`).join("")}</div>
    <pre class="render">${esc(lg.command)}</pre>
    <div class="hint">read-only git status per worktree — run the plan yourself, with the user's go; the console never runs git writes</div>` : "";
  if (!d.available) {
    $("lanes-sessions").innerHTML = `<div class="hint">no porter ledger at ${esc(d.db)}</div>`;
    $("lanes-events").innerHTML = "";
    return;
  }
  const sessions = d.sessions || [];
  $("lanes-sessions").innerHTML = sessions.length ? sessions.map(s => `
    <div class="lane-card ${s.alive ? "live" : "stale"}">
      <div class="lane-head"><span class="lane-agent">${esc(s.agent)}</span>
        <span class="lane-id">${esc(s.id)}</span>
        <span class="lane-state ${s.alive ? "verdict-yes" : "verdict-no"}">${s.alive ? "live" : "stale"}</span>
        <span class="lane-age">seen ${fmtAge(s.age_s)} ago</span></div>
      <div class="lane-task">${esc(s.task || "")}</div>
      <div class="lane-claims">${(s.claims || []).map(c =>
        `<span class="memchip" title="${esc(c.intent || "")}">${esc(c.path)}</span>`).join("") ||
        `<span class="hint">no claims</span>`}</div>
      <div class="hint src">${esc(s.project || "")}</div>
    </div>`).join("") : `<div class="hint">no active sessions</div>`;
  $("lanes-events").innerHTML = (d.events || []).map(e => `
    <div class="ledger-row"><b>${esc(e.kind)}</b> ${esc(e.actor)}${e.recipient ? " → " + esc(e.recipient) : ""}
      · ${esc((e.message || "").slice(0, 160))} · ${fmtTime(e.created ? Date.parse(e.created) / 1000 : null)}</div>`).join("")
    || `<div class="hint">no events</div>`;
}

/* ---------------------------------------------------------------- pocket (read-only node status) */

function fmtBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KiB";
  if (n < 1073741824) return (n / 1048576).toFixed(1) + " MiB";
  return (n / 1073741824).toFixed(2) + " GiB";
}

function pocketKV(obj, keys) {
  if (!obj) return `<div class="hint">no record yet</div>`;
  return keys.filter(([k]) => obj[k] != null).map(([k, label, fmt]) =>
    `<div class="ledger-row"><b>${esc(label)}</b> ${esc(fmt ? fmt(obj[k]) : obj[k])}</div>`).join("");
}

async function loadPocket() {
  const d = await get("/api/pocket");
  const n = d.node || {};
  $("pocket-node").innerHTML = `
    <span class="chip ${n.listening ? "" : "verdict-no"}">node ${n.listening ? "listening" : "down"}</span>
    <span class="chip">${esc(n.host || "127.0.0.1")}:${n.port ?? "—"} → phone :${n.phone_port ?? "—"}</span>`;
  $("pocket-backup").innerHTML = pocketKV(d.backup, [
    ["name", "archive"], ["snapshot", "snapshot"], ["bytes", "size", fmtBytes],
    ["encrypted", "encrypted", v => v ? "yes (gpg)" : "no"], ["written", "written"],
    ["sha256", "sha256", v => String(v).slice(0, 16) + "…"],
    ["pack_s", "pack", v => v + "s"], ["send_s", "send", v => v + "s"]]);
  $("pocket-drill").innerHTML = pocketKV(d.drill, [
    ["name", "archive"], ["passed", "restores", v => v ? "PASSED" : "FAILED"], ["when", "when"]]);
  $("pocket-bench").innerHTML = pocketKV(d.bench, [
    ["model", "model"], ["prompt_tok_s", "read", v => v + " tok/s"], ["gen_tok_s", "write", v => v + " tok/s"],
    ["prompt_tokens", "prompt tokens"], ["gen_tokens", "gen tokens"], ["wall_s", "wall", v => v + "s"],
    ["batt_c_before", "batt before", v => v + "°C"], ["batt_c_after", "batt after", v => v + "°C"], ["when", "when"]]);
}

/* ---------------------------------------------------------------- rescue (read-only scan) */

// Session tasks and paths come from other tools' logs: everything goes in through textContent.
async function loadRescue() {
  if (S.rescueBusy) return;          // a scan can take 30 s; never run two at once
  S.rescueBusy = true;
  try { await rescueScan(); } finally { S.rescueBusy = false; }
}

async function rescueScan() {
  const d = await get("/api/rescue");
  const chips = $("rescue-chips"), box = $("rescue-table");
  chips.innerHTML = ""; box.innerHTML = "";
  const hint = (text) => {
    const div = document.createElement("div");
    div.className = "hint"; div.textContent = text;
    box.append(div);
  };
  if (!d.available) {
    hint(`rescue lane not found at ${d.script} — this panel lights up once the lane lands`);
    return;
  }
  if (d.error) { hint(d.error); banner(d.error); return; }   // a failed scan is loud; a missing script is not
  const rows = [...(d.sessions || [])].sort((a, b) =>
    ((b.is_stale ? 1 : 0) - (a.is_stale ? 1 : 0)) ||
    ((b.uncommitted_count || 0) - (a.uncommitted_count || 0)));   // STALE first, then most uncommitted
  const stale = rows.filter(r => r.is_stale).length;
  const live = rows.filter(r => r.is_live).length;
  for (const [label, n, cls] of [["sessions", rows.length, ""], ["live", live, "verdict-yes"],
                                  ["stale", stale, stale ? "verdict-no" : ""]]) {
    const chip = document.createElement("span");
    chip.className = "chip" + (cls ? " " + cls : "");
    const b = document.createElement("b");
    b.textContent = n;
    chip.append(b, " " + label);
    chips.append(chip);
  }
  if (!rows.length) { hint("no sessions discovered"); return; }
  const table = document.createElement("table");
  table.className = "traj-table";
  const head = document.createElement("tr");
  for (const h of ["platform", "session", "worktree", "uncommitted", "also", "state", "seen", "last task"]) {
    const th = document.createElement("th");
    th.textContent = h;
    head.append(th);
  }
  table.append(head);
  const cell = (tr, text, cls, title) => {
    const td = document.createElement("td");
    if (cls) td.className = cls;
    if (title) td.title = title;
    td.textContent = text ?? "—";
    tr.append(td);
  };
  for (const r of rows) {
    const tr = document.createElement("tr");
    cell(tr, r.platform);
    cell(tr, String(r.session_id || "").slice(0, 12), "src", r.session_id);
    const wtTitle = r.worktree ? (r.branch ? `${r.worktree} (${r.branch})` : r.worktree) : "";
    cell(tr, r.worktree ? String(r.worktree).split("/").pop() : "—", "", wtTitle);
    const skipped = r.skipped_count || 0;
    const split = r.eligible_count != null
      ? `${r.eligible_count} safe to commit · ${skipped} private skipped` : "";
    cell(tr, r.uncommitted_count ?? 0, r.uncommitted_count ? "verdict-run" : "", split);
    const also = document.createElement("td");
    if (skipped > 0) {
      const chip = document.createElement("span");
      chip.className = "memchip";
      chip.textContent = `${skipped} private skipped`;
      also.append(chip);
    }
    const older = r.history_count || 0;
    if (older > 0) {
      const chip = document.createElement("span");
      chip.className = "memchip";
      chip.textContent = `+${older} older session${older === 1 ? "" : "s"}`;
      if ((r.history_sessions || []).length) chip.title = r.history_sessions.join("\n");
      also.append(chip);
    }
    const unmerged = r.unmerged_commits || 0;
    if (unmerged > 0 || r.has_unmerged_branch) {
      const chip = document.createElement("span");
      chip.className = "memchip verdict-run";
      chip.textContent = unmerged > 0 ? `${unmerged} unmerged commit${unmerged === 1 ? "" : "s"}`
                                      : "unmerged branch";
      also.append(chip);
    }
    if (!also.childNodes.length) also.textContent = "—";
    tr.append(also);
    cell(tr, r.state || (r.is_stale ? "STALE" : r.is_live ? "LIVE" : "—"),
         r.is_stale ? "verdict-no" : r.is_live ? "verdict-yes" : "");
    const seen = r.last_activity
      ? `${fmtAge(Math.max(0, Math.round(Date.now() / 1000 - r.last_activity)))} ago (${fmtTime(r.last_activity)})`
      : "—";
    cell(tr, seen);
    cell(tr, r.last_task || "—", "rescue-task", r.last_task || "");
    table.append(tr);
  }
  box.append(table);
}

/* ---------------------------------------------------------------- init */

function init() {
  document.querySelectorAll(".nav[data-view]").forEach(b => b.onclick = () => switchView(b.dataset.view));
  $("jobs-tray").onclick = () => toggleDrawer();
  $("jobs-drawer-close").onclick = () => toggleDrawer(false);
  $("feed-toggle").onclick = () => $("feed").classList.toggle("hidden");
  $("feed-pause").onclick = () => {
    S.feedPaused = !S.feedPaused;
    $("feed-pause").textContent = S.feedPaused ? "resume" : "pause";
  };
  $("source-close").onclick = () => $("source-modal").classList.add("hidden");
  // Escape closes the drawer, else the source modal. Never the approval modal: an approval must be
  // answered explicitly, not hidden (an unanswered one times out to "no").
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (S.drawerOpen) toggleDrawer(false);
    else if (!$("source-modal").classList.contains("hidden")) $("source-modal").classList.add("hidden");
  });
  $("task-run").onclick = dispatchTask;
  $("soup-refresh").onclick = loadSoup;
  $("soup-q").onkeydown = (e) => { if (e.key === "Enter") loadSoup(); };
  $("soup-kind").onchange = loadSoup; $("soup-status").onchange = loadSoup;
  $("remember-btn").onclick = async () => {
    const text = $("remember-text").value.trim();
    if (!text) return;
    try {
      const r = await post("/api/soup/remember", { text, kind: $("remember-kind").value });
      toast(`stored #${r.id}`); $("remember-text").value = ""; loadSoup();
    } catch (e) { toast(e.message, true); }
  };
  $("ledger-toggle").onclick = () => {
    const el = $("ledger");
    el.classList.toggle("hidden");
    if (!el.classList.contains("hidden")) loadLedger();
  };
  $("atlas-search").onclick = atlasSearch;
  $("atlas-q").onkeydown = (e) => { if (e.key === "Enter") atlasSearch(); };
  $("index-btn").onclick = async () => {
    const spec = $("index-spec").value.trim();
    if (!spec) return;
    try { toast("indexing…"); await post("/api/index", { spec, refresh: true }); }
    catch (e) { toast(e.message, true); }
  };
  $("build-run").onclick = async () => {
    try { await post("/api/build", { spec: $("build-spec").value, candidates: Number($("build-n").value) || 6 }); }
    catch (e) { toast(e.message, true); }
  };
  $("build-accept").onclick = async () => {
    try { const r = await post("/api/build/accept", {}); toast(`exemplar #${r.exemplar} active`); }
    catch (e) { toast(e.message, true); }
  };
  $("write-run").onclick = async () => {
    try { await post("/api/write", { brief: $("write-brief").value, words: Number($("write-words").value) || 800 }); }
    catch (e) { toast(e.message, true); }
  };
  $("evolve-run").onclick = async () => {
    try { toast("evolving…"); await post("/api/evolve"); } catch (e) { toast(e.message, true); }
  };
  $("skills-forge").onclick = async () => {
    try { const r = await post("/api/skills/forge"); toast(`${r.skills.length} skills forged`); loadSkills(); }
    catch (e) { toast(e.message, true); }
  };
  $("skills-export").onclick = async () => {
    const dir = $("skills-dir").value.trim();
    if (!dir) return toast("export needs a directory", true);
    try { const r = await post("/api/skills/export", { dir }); toast(`exported ${r.paths.length} SKILL.md`); }
    catch (e) { toast(e.message, true); }
  };
  $("vitals-refresh").onclick = loadVitals;
  $("gpu-refresh").onclick = () => loadGpu().catch(e => toast(e.message, true));
  $("matrix-refresh").onclick = loadMatrix;
  $("matrix-search").onclick = async () => {
    const q = $("matrix-q").value.trim();
    if (!q) return;
    try { await post("/api/matrix/search", { query: q }); } catch (e) { toast(e.message, true); }
  };
  $("matrix-q").onkeydown = (e) => { if (e.key === "Enter") $("matrix-search").click(); };
  $("mcp-run").onclick = async () => {
    if (!S.mcpTool) return;
    let args = {};
    try { args = JSON.parse($("mcp-args").value || "{}"); } catch { return toast("arguments must be JSON", true); }
    try {
      const r = await post("/api/mcp/call", { name: S.mcpTool.name, arguments: args });
      if (r.job) { toast("running as job " + r.job); showJob(r.job); }
      else renderMcpResult(r);
    } catch (e) { toast(e.message, true); }
  };
  $("mcp-cancel").onclick = () => $("mcp-call").classList.add("hidden");
  $("helix-refresh").onclick = loadHelix;
  $("helix-session").onchange = loadHelix;
  $("helix-run").onclick = async () => {
    const task = $("helix-task").value.trim();
    if (!task) return toast("a task is required", true);
    const body = { task, budget: Number($("helix-budget").value) || 4000,
                   session: $("helix-session").value || "main", lattice: $("helix-lattice").checked };
    if ($("helix-critic").value.trim()) body.critic_model = $("helix-critic").value.trim();
    try { const r = await post("/api/helix/run", body); toast("helix running"); showJob(r.job); }
    catch (e) { toast(e.message, true); }
  };
  $("helix-verify").onclick = async () => {
    try {
      const v = await get("/api/helix/verify");
      $("helix-verify-out").textContent = JSON.stringify(v, null, 2);
      $("helix-verify-panel").classList.remove("hidden");
    } catch (e) { toast(e.message, true); }
  };
  $("helix-diff").onclick = async () => {
    const a = $("helix-diff-a").value, b = $("helix-diff-b").value;
    if (!a || !b) return toast("two version numbers are required", true);
    const session = $("helix-session").value || "main";
    try {
      const d = await get(`/api/helix/diff?session=${encodeURIComponent(session)}&a=${a}&b=${b}`);
      showSource(`diff v${a} → v${b}`, JSON.stringify(d, null, 2));
    } catch (e) { toast(e.message, true); }
  };
  $("helix-ingest").onclick = async () => {
    const text = $("helix-ingest-state").value.trim();
    if (!text) return toast("paste a STATE handoff first", true);
    let state;
    try { state = JSON.parse(text); } catch { return toast("the handoff must be JSON", true); }
    try {
      const r = await post("/api/helix/ingest", { session: $("helix-session").value || "main", state,
                                                  source: $("helix-ingest-source").value.trim() || "console" });
      const cls = r.accepted ? "verdict-yes" : "verdict-no";
      $("helix-ingest-result").innerHTML = `<p class="hint ${cls}">${r.accepted ? "accepted" : "rejected"}` +
        `${r.violations?.length ? " — " + r.violations.map(esc).join("; ") : ""}` +
        `${r.clamped?.length ? " · clamped: " + r.clamped.map(esc).join("; ") : ""}</p>`;
      toast(r.accepted ? "handoff adopted" : "handoff rejected (nothing adopted)", !r.accepted);
      loadHelix();
    } catch (e) { toast(e.message, true); }
  };
  $("helix-audit").onclick = async () => {
    const text = $("helix-audit-text").value;
    if (!text.trim()) return toast("paste a transcript first", true);
    const body = { text };
    if ($("helix-audit-external").value) body.external = Number($("helix-audit-external").value);
    try { renderHelixAudit(await post("/api/helix/audit", body)); }
    catch (e) { toast(e.message, true); }
  };
  $("holo-refresh").onclick = loadHolo;
  $("holo-build").onclick = async () => {
    try { toast("building holo store…"); await post("/api/holo/build", {}); }
    catch (e) { toast(e.message, true); }
  };
  $("holo-search").onclick = async () => {
    const q = $("holo-q").value.trim();
    if (!q) return;
    try { await post("/api/holo/query", { query: q, k: Number($("holo-k").value) || 8 }); }
    catch (e) { toast(e.message, true); }
  };
  $("holo-q").onkeydown = (e) => { if (e.key === "Enter") $("holo-search").click(); };
  $("holo-overlaps-btn").onclick = () => loadHoloOverlaps().catch(e => toast(e.message, true));
  $("holo-verify-btn").onclick = () => loadHoloVerify().catch(e => toast(e.message, true));
  $("builder-run").onclick = async () => {
    const spec = $("builder-spec").value.trim();
    if (!spec) return toast("a spec is required", true);
    try {
      const r = await post("/api/builder/run", { spec, candidates: Number($("builder-n").value) || 6,
                                                 allow_io: $("builder-io").checked });
      toast("builder running"); showJob(r.job);
    } catch (e) { toast(e.message, true); }
  };
  $("lanes-refresh").onclick = loadLanes;
  $("lanes-project").onchange = loadLanes;
  $("pocket-refresh").onclick = loadPocket;
  $("rescue-refresh").onclick = () => loadRescue().catch(e => toast(e.message, true));
  setInterval(() => {
    if ($("lanes-auto").checked && $("view-lanes").classList.contains("active")) loadLanes();
  }, 12000);
  setInterval(checkStuckJobs, 5000);   // elapsed times, stuck flags and the tray counts stay fresh
  checkCodeChanged();
  setInterval(checkCodeChanged, 60000);
  setInterval(() => {                // opt-in: each refresh runs the scan subprocess (up to 30 s)
    if ($("rescue-auto").checked && $("view-rescue").classList.contains("active"))
      loadRescue().catch(e => toast(e.message, true));
  }, 60000);
  initChatControls();

  get("/api/state").then(snap => {
    S.snap = snap;
    $("route-chips").innerHTML = Object.entries(snap.route || {})
      .filter(([, m]) => m).map(([role, m]) => `<span class="chip">${role} <b>${esc(m)}</b></span>`).join("");
    $("ws-badge").textContent = snap.workspace?.attached ? `workspace: ${snap.workspace.root}` : "no workspace";
  }).catch(e => toast("state: " + e.message, true));
  loadRules();   // one path for the header badge and the Memory panel
  connect();
  switchView("chat");
}

document.addEventListener("DOMContentLoaded", init);