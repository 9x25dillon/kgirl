/* kgirl console — one page, one live stream, every module. */
"use strict";
const $ = (id) => document.getElementById(id);
const S = { snap: null, convId: null, chatJob: null, jobs: {}, feedPaused: false,
            matrix: null, mcpTool: null, approvals: [], applyConv: null };

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function toast(msg, err) {
  const t = $("toast"); t.textContent = msg; t.className = err ? "err" : "";
  clearTimeout(t._h); t._h = setTimeout(() => t.className = "hidden", err ? 6000 : 3200);
}

async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  let data = null;
  try { data = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw new Error(data?.error || `${r.status} ${r.statusText}`);
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
  const es = new EventSource("/api/events");
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
    case "bus": break;   // feed only
  }
}

function trackJob(j) {
  S.jobs[j.id] = { ...j, events: [] };
  const mini = $("jobs-mini");
  const pill = document.createElement("div");
  pill.className = "job-pill"; pill.id = "pill-" + j.id;
  pill.innerHTML = `<span class="spin">●</span> ${esc(j.kind)} ${esc(j.label || "").slice(0, 24)}`;
  pill.onclick = () => showJob(j.id);
  mini.prepend(pill);
  while (mini.children.length > 5) mini.lastChild.remove();
}

function jobEvent(ev) {
  const inner = ev.event || {};
  const j = S.jobs[ev.job];
  if (j) j.events.push(inner);
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
  if (j && full) Object.assign(j, full);
  if (ev.status === "error") toast(`${ev.kind} failed: ${ev.error}`, true);
  if (ev.job === S.chatJob) { S.chatJob = null; finalizeChat(full, ev); return; }
  if (!full) return;
  const r = full.result;
  switch (ev.kind) {
    case "task": renderTask(r); break;
    case "build": renderBuild(r); break;
    case "write": renderWrite(r); break;
    case "matrix": renderMatrixHits(r); break;
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
                    forge: loadForge, vitals: loadVitals, matrix: loadMatrix, services: loadServices };
  loaders[name]?.();
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

/* ---------------------------------------------------------------- init */

function init() {
  document.querySelectorAll(".nav[data-view]").forEach(b => b.onclick = () => switchView(b.dataset.view));
  $("feed-toggle").onclick = () => $("feed").classList.toggle("hidden");
  $("feed-pause").onclick = () => {
    S.feedPaused = !S.feedPaused;
    $("feed-pause").textContent = S.feedPaused ? "resume" : "pause";
  };
  $("source-close").onclick = () => $("source-modal").classList.add("hidden");
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
  initChatControls();

  get("/api/state").then(snap => {
    S.snap = snap;
    $("route-chips").innerHTML = Object.entries(snap.route || {})
      .filter(([, m]) => m).map(([role, m]) => `<span class="chip">${role} <b>${esc(m)}</b></span>`).join("");
    $("ws-badge").textContent = snap.workspace?.attached ? `workspace: ${snap.workspace.root}` : "no workspace";
  }).catch(e => toast("state: " + e.message, true));
  connect();
  switchView("chat");
}

document.addEventListener("DOMContentLoaded", init);