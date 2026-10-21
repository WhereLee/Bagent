"""P2-4：最小 Agent 工作台（无构建、同源 SSE）。

页面只做四件事：发起 run、显示 step/tool/approval 事件、审批/拒绝/取消、重放。
不引入前端构建链，也不替代正式产品 UI。
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_WORKBENCH = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bagent Agent Workbench</title>
<style>
  :root { color-scheme: light dark; --bg:#f6f7f9; --panel:#fff; --ink:#17202a; --muted:#68707a; --line:#dde2e8; --accent:#2f6fed; --ok:#1b7f4b; --bad:#b42318; }
  @media (prefers-color-scheme: dark) { :root { --bg:#11151a; --panel:#171d24; --ink:#e8edf3; --muted:#9aa6b2; --line:#2b3540; --accent:#79a5ff; --ok:#69d39a; --bad:#ff8a80; } }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:1000px; margin:0 auto; padding:28px 20px 60px; }
  h1 { font-size:24px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; margin:12px 0; }
  label { display:block; font-size:12px; color:var(--muted); margin:8px 0 4px; }
  input, textarea { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:7px; background:transparent; color:var(--ink); font:inherit; }
  textarea { min-height:72px; resize:vertical; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:end; }
  .row > div { flex:1 1 180px; }
  button { border:1px solid var(--line); background:transparent; color:var(--ink); padding:8px 12px; border-radius:7px; cursor:pointer; font:inherit; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.ok { color:var(--ok); border-color:var(--ok); }
  button.bad { color:var(--bad); border-color:var(--bad); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  #status { font-size:13px; color:var(--muted); }
  #events { display:flex; flex-direction:column; gap:8px; max-height:52vh; overflow:auto; }
  .event { border-left:3px solid var(--line); padding:7px 10px; background:var(--bg); border-radius:0 6px 6px 0; }
  .event.tool { border-left-color:var(--accent); }
  .event.error { border-left-color:var(--bad); }
  .event.approval { border-left-color:#d18b00; }
  .event strong { font-size:12px; letter-spacing:.02em; }
  .event code { color:var(--muted); font-size:12px; }
  .actions { display:flex; gap:8px; margin-top:10px; }
  .hint { color:var(--muted); font-size:12px; }
  pre#final { white-space:pre-wrap; margin:0; font:inherit; }
</style>
</head>
<body>
<main>
  <h1>Bagent Agent Workbench</h1>
  <p class="sub">P2 最小工作台：发起 run、观察 step/tool 事件、审批后继续。</p>

  <section class="panel">
    <div class="row">
      <div style="flex:3 1 420px">
        <label for="goal">目标</label>
        <textarea id="goal" placeholder="例如：查一下 Bagent 的工具审批流程，并指出风险"></textarea>
      </div>
      <div>
        <label for="steps">max steps</label>
        <input id="steps" type="number" min="1" max="20" value="6">
      </div>
    </div>
    <div class="row" style="margin-top:8px">
      <div><label for="bearer">Bearer token（AUTH 开启时填写）</label><input id="bearer" type="password" autocomplete="off"></div>
      <div><label for="apikey">X-API-Key（如服务配置）</label><input id="apikey" type="password" autocomplete="off"></div>
    </div>
    <div class="actions">
      <button id="start" class="primary">开始运行</button>
      <button id="refresh" disabled>重放事件</button>
      <span id="status">idle</span>
    </div>
    <p class="hint">事件来自 <code>/agent/runs/stream</code>；审批后页面调用 approve/reject，再重放 <code>/agent/runs/{id}/events</code>。页面不保存 token。</p>
  </section>

  <section class="panel">
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
      <strong>最近运行</strong><button id="historyRefresh">刷新</button>
    </div>
    <div id="history" class="hint" style="margin-top:8px">尚未加载</div>
  </section>

  <section class="panel" id="finalPanel" hidden>
    <strong>最终答案</strong>
    <pre id="final" style="margin-top:8px"></pre>
  </section>

  <section class="panel">
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:center">
      <strong>运行事件</strong><span id="runmeta" class="hint"></span>
    </div>
    <div id="events" aria-live="polite"></div>
    <div id="approval" class="actions" hidden>
      <button id="approve" class="ok">批准并继续</button>
      <button id="reject" class="bad">拒绝</button>
      <button id="cancel">取消运行</button>
    </div>
  </section>
</main>
<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { runId: null, token: null, pendingStep: null, busy: false };

  function headers() {
    const h = { "Content-Type": "application/json" };
    const bearer = $("bearer").value.trim();
    const key = $("apikey").value.trim();
    if (bearer) h.Authorization = "Bearer " + bearer;
    if (key) h["X-API-Key"] = key;
    return h;
  }
  function setStatus(text) { $("status").textContent = text; }
  function addEvent(name, data) {
    if (name === "heartbeat") return;
    const row = document.createElement("div");
    row.className = "event" + (name.includes("tool") ? " tool" : "") +
      (name.includes("approval") ? " approval" : "") +
      (data.status === "error" ? " error" : "");
    const time = new Date().toLocaleTimeString();
    const action = data.data && (data.data.action || data.data.tool) ? " · " + (data.data.action || data.data.tool) : "";
    row.innerHTML = "<strong>" + escapeHtml(name) + "</strong> <code>" + time + escapeHtml(action) + "</code>" +
      (data.status ? "<div class='hint'>status: " + escapeHtml(String(data.status)) + "</div>" : "");
    $("events").appendChild(row);
    $("events").scrollTop = $("events").scrollHeight;
    if (data.status) $("runmeta").textContent = "run " + (state.runId || "") + " · " + data.status;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c])); }

  async function consume(res) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\\n\\n")) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        let name = "message", payload = "";
        for (const line of frame.split("\\n")) {
          if (line.startsWith("event:")) name = line.slice(6).trim();
          if (line.startsWith("data:")) payload += line.slice(5).trim();
        }
        if (!payload) continue;
        let data;
        try { data = JSON.parse(payload); } catch (_) { continue; }
        addEvent(name, data);
        if (data.event_id != null) state.lastEventId = data.event_id;
        if (name === "approval_required") {
          state.pendingStep = data.step_index;
          $("approval").hidden = false;
          setStatus("等待审批");
        }
        if (name === "run_completed" || name === "run_failed" ||
            name === "run_cancelled" || name === "run_budget_exceeded") {
          $("approval").hidden = true;
          setStatus(data.status || name);
          await showFinal();
        }
      }
    }
  }

  async function loadHistory() {
    try {
      const res = await fetch("/agent/runs?limit=20", { headers: headers() });
      if (!res.ok) { $("history").textContent = "历史加载失败 HTTP " + res.status; return; }
      const data = await res.json();
      $("history").innerHTML = "";
      if (!data.runs || !data.runs.length) { $("history").textContent = "暂无运行"; return; }
      for (const run of data.runs) {
        const btn = document.createElement("button");
        btn.style.margin = "3px 6px 3px 0";
        btn.textContent = run.status + " · " + (run.goal || run.run_id).slice(0, 42);
        btn.title = run.run_id;
        btn.addEventListener("click", () => openRun(run.run_id));
        $("history").appendChild(btn);
      }
    } catch (e) { $("history").textContent = "历史加载失败: " + e.message; }
  }

  async function openRun(runId) {
    state.runId = runId; state.token = null; state.pendingStep = null;
    $("events").innerHTML = ""; $("approval").hidden = true; $("finalPanel").hidden = true;
    $("refresh").disabled = false; setStatus("history run");
    await replay();
  }

  async function showFinal() {
    if (!state.runId) return;
    try {
      const res = await fetch("/agent/runs/" + encodeURIComponent(state.runId), { headers: headers() });
      if (!res.ok) return;
      const run = await res.json();
      const answer = (run.answer && run.answer.text) || run.final || "";
      $("final").textContent = answer || ("（无最终文本，status=" + run.status + "）");
      $("finalPanel").hidden = false;
    } catch (_) { /* 事件流已成功时，最终答案拉取失败不阻塞页面 */ }
  }

  async function start() {
    if (state.busy) return;
    state.busy = true; state.runId = null; state.token = null; state.pendingStep = null;
    $("events").innerHTML = ""; $("approval").hidden = true; $("refresh").disabled = true;
    setStatus("starting");
    try {
      const res = await fetch("/agent/runs/stream", {
        method: "POST", headers: headers(),
        body: JSON.stringify({ goal: $("goal").value.trim(), max_steps: Number($("steps").value) || 6 })
      });
      if (!res.ok) throw new Error("HTTP " + res.status + " " + await res.text());
      state.runId = res.headers.get("X-Run-ID");
      state.token = res.headers.get("X-Approval-Token");
      $("runmeta").textContent = "run " + state.runId;
      $("refresh").disabled = false;
      await consume(res);
      await loadHistory();
    } catch (e) { setStatus("error: " + e.message); }
    finally { state.busy = false; }
  }

  async function replay() {
    if (!state.runId || state.busy) return;
    state.busy = true; setStatus("replaying");
    try {
      const res = await fetch("/agent/runs/" + encodeURIComponent(state.runId) + "/events", { headers: headers() });
      if (!res.ok) throw new Error("HTTP " + res.status);
      await consume(res);
      await loadHistory();
    } catch (e) { setStatus("error: " + e.message); }
    finally { state.busy = false; }
  }

  async function decide(decision) {
    if (!state.runId || !state.token || state.pendingStep == null) return;
    const body = { step_index: state.pendingStep, approval_token: state.token };
    if (decision === "reject") body.reason = "workbench reject";
    try {
      const res = await fetch("/agent/runs/" + encodeURIComponent(state.runId) + "/" + decision,
        { method: "POST", headers: headers(), body: JSON.stringify(body) });
      if (!res.ok) throw new Error("HTTP " + res.status + " " + await res.text());
      $("approval").hidden = true;
      await replay();
    } catch (e) { setStatus("error: " + e.message); }
  }

  $("start").addEventListener("click", start);
  $("refresh").addEventListener("click", replay);
  $("historyRefresh").addEventListener("click", loadHistory);
  const initialRun = new URLSearchParams(location.search).get("run_id");
  if (initialRun) { openRun(initialRun); } else { loadHistory(); }
  $("approve").addEventListener("click", () => decide("approve"));
  $("reject").addEventListener("click", () => decide("reject"));
  $("cancel").addEventListener("click", () => decide("cancel"));
})();
</script>
</body>
</html>"""


@router.get("/workbench", response_class=HTMLResponse)
def workbench() -> HTMLResponse:
    return HTMLResponse(_WORKBENCH)
