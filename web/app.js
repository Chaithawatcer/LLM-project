// Minimal client for POST /api/ask (Server-Sent Events over a POST, so EventSource - GET only - won't do).
// Everything from the model or a tool is rendered with textContent, never innerHTML: it is untrusted text.
const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  "How many Microsoft vulnerabilities were added to KEV in 2025?",
  "Which 5 vendors have the most KEV entries with known ransomware use?",
  "What are the phases of the incident response life cycle in NIST SP 800-61 Rev. 3?",
  "What does NIST recommend for emergency patching?",
  "What is the capital of France?",
];
for (const text of EXAMPLES) {
  const b = document.createElement("button");
  b.className = "chip";
  b.type = "button";
  b.textContent = text;
  b.onclick = () => { $("q").value = text; $("q").focus(); };
  $("examples").appendChild(b);
}

fetch("/api/health").then((r) => r.json()).then((h) => {
  const ok = h.llm_key_set && h.index_ready;
  $("health").textContent =
    `model ${h.chat_model} · retrieval ${h.retrieval} · chunking ${h.chunking} · min_score ${h.min_score}` +
    (h.llm_key_set ? "" : " · ⚠ OPENAI_API_KEY not set") +
    (h.index_ready ? "" : " · ⚠ no vector index (run: uv run python -m app.ingest)");
  $("health").style.color = ok ? "" : "var(--warn)";
}).catch(() => { $("health").textContent = "server not reachable"; });

function step(kind, text) {
  const d = document.createElement("div");
  d.className = "step";
  const b = document.createElement("b");
  b.className = kind;
  b.textContent = kind.toUpperCase() + " ";
  const body = document.createElement(kind === "obs" ? "pre" : "code");
  body.textContent = text;
  d.append(b, body);
  $("trace").appendChild(d);
}

function tag(cls, text) {
  const s = document.createElement("span");
  s.className = "tag " + cls;
  s.textContent = text;
  return s;
}

function list(title, items) {
  const wrap = document.createElement("div");
  const h = document.createElement("div");
  h.className = "muted";
  h.textContent = title;
  const ul = document.createElement("ul");
  for (const text of items) {
    const li = document.createElement("li");
    li.textContent = text;
    ul.appendChild(li);
  }
  wrap.append(h, ul);
  return wrap;
}

function showFinal(f) {
  $("answer").textContent = f.answer;
  const meta = $("meta");
  meta.replaceChildren();
  const flags = document.createElement("div");
  if (f.refused) flags.append(tag("warn", "declined: the sources can't answer this"));
  else flags.append(f.grounded ? tag("ok", "grounded in tool results") : tag("warn", "no tool result was read - treat with caution"));
  if (f.stopped === "max_steps") flags.append(tag("warn", "hit the step limit"));
  if (f.invalid_citations.length) flags.append(tag("bad", "invalid citation " + f.invalid_citations.map((n) => `[${n}]`).join(" ")));
  meta.append(flags);
  if (f.citations.length) {
    meta.append(list("Sources (recorded by the server when it retrieved them)",
      f.citations.map((c) => `[${c.n}] ${c.title}, p.${c.page}`)));
  }
  if (f.sql.length) {
    meta.append(list("SQL that ran",
      f.sql.map((s) => (s.ok ? `${s.sql}  → ${s.n_rows} row(s)` : `${s.sql}  → refused: ${s.error}`))));
  }
  $("answer-panel").hidden = false;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
  if (!$("trace").children.length) $("trace-panel").hidden = true;  // nothing happened: don't show an empty box
}

$("ask").onclick = async () => {
  const question = $("q").value.trim();
  if (question.length < 3) return;
  $("ask").disabled = true;
  $("error").hidden = true;
  $("answer-panel").hidden = true;
  $("trace").replaceChildren();
  $("trace-panel").hidden = false;
  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: $("mode").value }),
    });
    if (!resp.ok || !resp.body) throw new Error(`request failed (${resp.status})`);
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
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "act") step("act", `${ev.tool}(${ev.args})`);
        else if (ev.type === "obs") step("obs", ev.preview);
        else if (ev.type === "final") showFinal(ev);
        else if (ev.type === "error") { showError(ev.message); }
      }
    }
  } catch (e) {
    showError(String(e));
  } finally {
    $("ask").disabled = false;
  }
};
