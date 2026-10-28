// Cyber Advisor UI. Streams POST /api/ask (Server-Sent Events over a POST, so EventSource - GET only - won't do).
// Everything that comes from the model or a tool is rendered as text nodes, never as HTML: it is untrusted text.
// (innerHTML is used only for the constant icon strings below.)
const $ = (id) => document.getElementById(id);
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

// ---- theme ---------------------------------------------------------------------------------------------
const ICON_SUN = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M18.7 5.3l-1.4 1.4M6.7 17.3l-1.4 1.4"/></svg>';
const ICON_MOON = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.2A8 8 0 0 1 9.8 4 8 8 0 1 0 20 14.2Z"/></svg>';

function currentTheme() {
  return document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function paintThemeButton() {
  const dark = currentTheme() === "dark";
  $("theme").innerHTML = dark ? ICON_SUN : ICON_MOON;
  $("theme").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
}
$("theme").onclick = () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch { /* private mode: the choice just isn't remembered */ }
  paintThemeButton();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paintThemeButton);
paintThemeButton();

// ---- server status -------------------------------------------------------------------------------------
function pill(label, value) {
  const p = el("span", "pill detail");
  p.append(document.createTextNode(label + " "), el("b", null, String(value)));
  return p;
}
function statePill(ok, text, hint) {
  const p = el("span", "pill");
  p.append(el("span", "dot " + (ok ? "dot-ok" : "dot-warn")), document.createTextNode(text));
  if (hint) p.title = hint;
  return p;
}
fetch("/api/health").then((r) => r.json()).then((h) => {
  const box = $("status");
  box.replaceChildren();
  if (!h.llm_key_set) box.append(statePill(false, "API key not set", "Put OPENAI_API_KEY in .env, then restart the server"));
  else if (!h.index_ready) box.append(statePill(false, "Index missing", "Run: uv run python -m app.ingest"));
  else box.append(statePill(true, "Ready", "The server is configured"));
  box.append(pill("Model", h.chat_model), pill("Retrieval", h.retrieval), pill("Floor", h.min_score));
}).catch(() => {
  const box = $("status");
  box.replaceChildren(statePill(false, "Server not reachable"));
});

// ---- composer ------------------------------------------------------------------------------------------
const q = $("q");
const EXAMPLES = [
  { kind: "sql", label: "Microsoft entries added in 2025", tip: "Table question", q: "How many Microsoft vulnerabilities were added to KEV in 2025?" },
  { kind: "sql", label: "Top ransomware vendors", tip: "Table question", q: "Which 5 vendors have the most KEV entries with known ransomware use?" },
  { kind: "docs", label: "Incident response phases", tip: "Document question", q: "What are the phases of the incident response life cycle in NIST SP 800-61 Rev. 3?" },
  { kind: "docs", label: "Emergency patching", tip: "Document question", q: "What does NIST recommend for emergency patching?" },
  { kind: "decline", label: "Out of scope", tip: "Should be declined", q: "What is the capital of France?" },
  { kind: "weak", label: "Known weak spot", tip: "A question the model gets wrong (answers 220, the true value is 112)", q: "How many KEV entries are classified as CWE-78? Count CWE-78 only, not ids that merely start with 78." },
];
for (const ex of EXAMPLES) {
  const b = el("button", "chip");
  b.type = "button";
  b.dataset.kind = ex.kind;
  b.title = `${ex.tip}: ${ex.q}`;
  b.append(el("span", "dot"), document.createTextNode(ex.label));
  b.onclick = () => { q.value = ex.q; sync(); q.focus(); };
  $("examples").append(b);
}

function sync() {
  const n = q.value.length;
  $("count").textContent = `${n} / 1000`;
  $("count").classList.toggle("near", n > 900);
  q.style.height = "auto";
  q.style.height = Math.min(q.scrollHeight, 260) + "px";
}
q.addEventListener("input", sync);
q.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); $("composer").requestSubmit(); }
});
sync();

// ---- rendering -----------------------------------------------------------------------------------------
let lastStep = null;
let busy = false;

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}
function hideError() { $("error").hidden = true; }

function setBusy(on) {
  busy = on;
  $("ask").disabled = on;
  $("ask").querySelector(".btn-label").textContent = on ? "Working…" : "Ask";
  $("answer-card").setAttribute("aria-busy", String(on));
  const state = $("activity-state");
  if (on) state.replaceChildren(el("span", "spinner"), document.createTextNode("Working…"));
  else if (state.querySelector(".spinner")) state.replaceChildren();   // keep a "Finished in N steps" note if one was written
}

function resetResults() {
  for (const id of ["trace", "badges", "sources", "sql"]) $(id).replaceChildren();
  $("sources-wrap").hidden = true;
  $("sql-wrap").hidden = true;
  $("trace-empty").hidden = true;
  lastStep = null;
  const a = $("answer");
  a.replaceChildren();
  a.classList.add("pending");
  const sk = el("div", "skeleton");
  sk.append(el("i"), el("i"), el("i"));
  a.append(sk);
}

function argBlock(raw) {
  const box = el("div", "arg");
  let obj = null;
  try { obj = JSON.parse(raw); } catch { /* show the raw text */ }
  if (obj && typeof obj === "object" && !Array.isArray(obj) && Object.keys(obj).length) {
    Object.entries(obj).forEach(([k, v], i) => {
      if (i) box.append(document.createTextNode("\n"));
      if ((k === "sql" || k === "query") && typeof v === "string") box.append(document.createTextNode(v));
      else box.append(el("span", "k", k + ": "), document.createTextNode(typeof v === "string" ? v : JSON.stringify(v)));
    });
  } else {
    box.textContent = raw || "(no arguments)";
  }
  return box;
}

function addAct(ev) {
  const li = el("li", "step");
  li.dataset.tool = ev.tool;
  li.append(el("span", "node"));
  const title = el("div", "step-title");
  title.append(el("span", "tool", ev.tool), el("span", "step-n", `step ${ev.step}`));
  li.append(title, argBlock(ev.args));
  $("trace").append(li);
  lastStep = li;
}

function addObs(ev) {
  if (!lastStep) return;
  // the server fences tool output as <tool_result ...>; that wrapper is for the model, not for people
  const body = (ev.preview || "").replace(/^<tool_result[^>]*>\s*/, "").replace(/\s*<\/tool_result>\s*$/, "").trim();
  const first = body.split("\n").find((l) => l.trim()) || "(empty result)";
  const d = el("details", "obs");
  if (/^error:/i.test(body)) d.classList.add("err");
  const s = el("summary");
  s.append(el("span", null, first.length > 90 ? first.slice(0, 89) + "…" : first));
  d.append(s, el("pre", null, body));
  lastStep.append(d);
}

function badge(cls, text) { return el("span", "badge badge-" + cls, text); }

function jumpTo(n) {
  const target = document.getElementById("src-" + n);
  if (!target) return;
  target.scrollIntoView({ block: "center", behavior: reduceMotion() ? "auto" : "smooth" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1400);
}

function citeChip(n, known, invalid) {
  if (known.has(n)) {
    const a = el("a", "cite", String(n));
    a.href = "#src-" + n;
    a.title = `${known.get(n).title}, page ${known.get(n).page}`;
    a.addEventListener("click", (e) => { e.preventDefault(); jumpTo(n); });
    return a;
  }
  if (invalid.has(n)) {
    const s = el("span", "cite cite-bad", String(n));
    s.title = "Invalid: this number does not match anything the server retrieved";
    return s;
  }
  const s = el("span", "cite cite-loose", String(n));
  s.title = "Not verified: the server only checks single [n] markers";
  return s;
}

// A small, safe Markdown subset for the model's answer: **bold**, `code`, bullet / numbered lists, paragraphs, and
// [n] citation chips. It only ever creates DOM nodes from text, so nothing the model writes can become markup.
function inline(parent, text, f) {
  const known = new Map(f.citations.map((c) => [c.n, c]));
  const invalid = new Set(f.invalid_citations);
  for (const part of text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`|\[\d{1,3}(?:\s*,\s*\d{1,3})*\])/)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) parent.append(el("strong", null, part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) parent.append(el("code", null, part.slice(1, -1)));
    else if (/^\[\d/.test(part) && part.endsWith("]")) {
      for (const n of part.slice(1, -1).split(",").map((x) => Number(x.trim()))) parent.append(citeChip(n, known, invalid));
    } else parent.append(document.createTextNode(part));
  }
}

function renderAnswer(container, text, f) {
  container.replaceChildren();
  let list = null;   // the <ul>/<ol> currently being filled
  let para = null;   // the <p> currently being filled
  for (const raw of text.replace(/\r\n/g, "\n").split("\n")) {
    const line = raw.trimEnd();
    if (!line.trim()) { list = null; para = null; continue; }
    const item = /^\s*(?:[*•-]|\d+[.)])\s+(.*)$/.exec(line);
    if (item) {
      const ordered = /^\s*\d/.test(line);
      if (!list || list.dataset.ordered !== String(ordered)) {
        list = el(ordered ? "ol" : "ul");
        list.dataset.ordered = String(ordered);
        container.append(list);
        para = null;
      }
      const li = el("li");
      inline(li, item[1], f);
      list.append(li);
    } else {
      list = null;
      if (!para) { para = el("p"); container.append(para); }
      else para.append(document.createElement("br"));
      inline(para, line.trim(), f);
    }
  }
}

function sqlBox(s) {
  const box = el("div", "sqlbox");
  const head = el("div", "sqlbox-head");
  head.append(s.ok ? badge("ok", `${s.n_rows} row${s.n_rows === 1 ? "" : "s"}`) : badge("bad", "refused"));
  if (!s.ok && s.error) head.append(el("span", null, s.error));
  const copy = el("button", "copy", "Copy");
  copy.type = "button";
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(s.sql); copy.textContent = "Copied"; } catch { copy.textContent = "Copy failed"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1400);
  };
  head.append(copy);
  const pre = el("pre");
  pre.append(el("code", null, s.sql));
  box.append(head, pre);
  return box;
}

function renderFinal(f) {
  const a = $("answer");
  a.classList.remove("pending");
  renderAnswer(a, f.answer || "(the model returned an empty answer)", f);

  const badges = $("badges");
  badges.replaceChildren();
  if (f.refused) badges.append(badge("warn", "Declined: the sources can't answer this"));
  else badges.append(f.grounded ? badge("ok", "Grounded in tool results") : badge("warn", "No tool result was read: treat with caution"));
  if (f.stopped === "max_steps") badges.append(badge("warn", "Hit the step limit"));
  if (f.invalid_citations.length) badges.append(badge("bad", "Invalid citation " + f.invalid_citations.map((n) => `[${n}]`).join(" ")));
  if (f.tools_used.length) badges.append(badge("info", "Used " + f.tools_used.join(" + ")));

  if (f.citations.length) {
    const box = $("sources");
    for (const c of [...f.citations].sort((x, y) => x.n - y.n)) {
      const row = el("div", "source");
      row.id = "src-" + c.n;
      row.append(el("span", "cite", String(c.n)), el("span", "source-title", c.title), el("span", "source-meta", `page ${c.page}`));
      box.append(row);
    }
    $("sources-wrap").hidden = false;
  }
  if (f.sql.length) {
    for (const s of f.sql) $("sql").append(sqlBox(s));
    $("sql-wrap").hidden = false;
  }
  $("trace-empty").hidden = $("trace").children.length > 0;
  const state = $("activity-state");
  state.replaceChildren(document.createTextNode(`Finished in ${f.steps} step${f.steps === 1 ? "" : "s"}`));
}

function dropPending() {
  const a = $("answer");
  if (!a.classList.contains("pending")) return;
  a.classList.remove("pending");
  if ($("trace").children.length) {
    a.replaceChildren(el("span", "note", "The agent stopped before it could write an answer."));
  } else {
    $("results").hidden = true;
  }
}

// ---- ask -----------------------------------------------------------------------------------------------
$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  hideError();
  const question = q.value.trim();
  if (question.length < 3) { showError("Please type a question of at least 3 characters."); q.focus(); return; }
  const mode = document.querySelector('input[name="mode"]:checked').value;

  resetResults();
  $("results").hidden = false;
  setBusy(true);
  $("results").scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode }),
    });
    if (!resp.ok || !resp.body) {
      let detail = "";  // FastAPI sends {"detail": "..."}; a validation error sends a list, so only show strings
      try { const d = (await resp.json()).detail; if (typeof d === "string") detail = d; } catch { /* not JSON */ }
      throw new Error(detail || `request failed (${resp.status})`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");   // network chunks don't line up with SSE frames: stitch
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line.slice(6)); } catch { continue; }
        if (ev.type === "act") addAct(ev);
        else if (ev.type === "obs") addObs(ev);
        else if (ev.type === "final") renderFinal(ev);
        else if (ev.type === "error") showError(ev.message);
      }
    }
  } catch (err) {
    showError(String(err.message || err));
  } finally {
    setBusy(false);
    dropPending();
  }
});
