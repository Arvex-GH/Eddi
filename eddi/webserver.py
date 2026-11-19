"""Local web chat UI for Eddi.

A small, dependency-free server that serves one page and a JSON API. It reuses
the project's design tokens so the UI looks like the rest of Eddi rather than a
separate artefact.

    python -m eddi.webserver --port 8000
    python -m eddi.webserver --port 8000 --brain path/to/brain.pkl

Endpoints:

* ``GET  /``        - the chat page
* ``GET  /health``  - liveness + version
* ``GET  /stats``   - what the brain currently knows
* ``POST /ask``     - ``{"text": "..."}`` -> ``{"text", "kind", "confidence"}``
* ``POST /ingest``  - ``{"text": "..."}`` -> teach Eddi a paragraph
* ``POST /teach``   - ``{"subject","relation","object"}`` -> write one fact
* ``POST /reset``   - clear the conversation transcript (not the brain)

Conversation turns are kept in the page, not on the server: the server is
stateless, so two browser tabs cannot corrupt each other's history.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import __version__
from .agent import EddiAgent
from .html_library import DESIGN_TOKENS
from .paths import default_brain_path
from .training import brain_exists, load_brain, save_brain

MAX_BODY = 1 << 20  # 1 MiB: a chat message, not a file upload


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _tokens() -> str:
    """Emit the design tokens as CSS custom properties.

    Underscores become hyphens: the token names are Python identifiers
    (``accent_ink``) while CSS custom properties are conventionally hyphenated
    (``--accent-ink``), and a mismatch here is silent - the variable simply does
    not resolve, so text renders in whatever colour it inherited.
    """
    return "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in DESIGN_TOKENS.items())


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eddi</title>
<style>
:root {
__TOKENS__
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  background: var(--bg); color: var(--ink);
  font-family: var(--font); font-size: 16px; line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  display: flex; flex-direction: column;
}

/* ---------- header ---------- */
header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; padding: 14px 22px; background: var(--surface);
  border-bottom: 1px solid var(--line); flex: 0 0 auto;
}
.brand { display: flex; align-items: center; gap: 11px; font-weight: 800;
         letter-spacing: -.02em; font-size: 1.15rem; }
.dot {
  width: 10px; height: 10px; border-radius: 50%; background: #1fb46b;
  box-shadow: 0 0 0 3px rgba(31,180,107,.16);
}
.dot.off { background: #d14343; box-shadow: 0 0 0 3px rgba(209,67,67,.16); }
.meta { display: flex; align-items: center; gap: 14px; font-size: .86rem;
        color: var(--muted); }
.meta b { color: var(--ink); font-weight: 650; }
.pill {
  background: var(--accent-soft); color: var(--accent);
  border-radius: 999px; padding: 4px 12px; font-size: .78rem; font-weight: 650;
}

/* ---------- transcript ---------- */
main {
  flex: 1 1 auto; overflow-y: auto; padding: 26px 22px 8px;
  scroll-behavior: smooth;
}
.thread { max-width: 780px; margin: 0 auto; display: flex;
          flex-direction: column; gap: 18px; }

.turn { display: flex; gap: 12px; align-items: flex-start; }
.turn.user { flex-direction: row-reverse; }
.avatar {
  flex: 0 0 34px; width: 34px; height: 34px; border-radius: 10px;
  display: grid; place-items: center; font-size: .82rem; font-weight: 800;
  background: var(--accent); color: var(--accent-ink);
}
.turn.user .avatar { background: var(--line); color: var(--muted); }
.bubble {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 13px 16px; box-shadow: var(--shadow);
  max-width: min(620px, 82%); white-space: pre-wrap; overflow-wrap: anywhere;
}
.turn.user .bubble { background: var(--accent); color: var(--accent-ink);
                     border-color: transparent; }
.turn.user .bubble a { color: #fff; }

.foot { margin-top: 8px; font-size: .74rem; color: var(--muted);
        display: flex; gap: 8px; align-items: center; }
.turn.user .foot { justify-content: flex-end; }
.tag { background: var(--accent-soft); color: var(--accent); border-radius: 6px;
       padding: 2px 7px; font-weight: 650; letter-spacing: .01em; }
.tag.warn { background: #fff3e0; color: #a35b00; }

/* ---------- empty state ---------- */
.empty { max-width: 620px; margin: 8vh auto 0; text-align: center; }
.empty h1 { font-size: clamp(1.6rem, 4vw, 2.1rem); font-weight: 800;
            letter-spacing: -.02em; margin: 0 0 .35em; }
.empty p { color: var(--muted); margin: 0 0 1.6em; }
.chips { display: flex; flex-wrap: wrap; gap: 9px; justify-content: center; }
.chip {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 999px; padding: 8px 15px; font: inherit; font-size: .88rem;
  color: var(--ink); cursor: pointer; transition: all .12s ease;
}
.chip:hover { border-color: var(--accent); color: var(--accent);
              transform: translateY(-1px); box-shadow: var(--shadow); }

/* ---------- typing ---------- */
.typing { display: inline-flex; gap: 4px; padding: 4px 0; }
.typing i {
  width: 6px; height: 6px; border-radius: 50%; background: var(--muted);
  animation: bob 1.2s infinite ease-in-out;
}
.typing i:nth-child(2) { animation-delay: .16s; }
.typing i:nth-child(3) { animation-delay: .32s; }
@keyframes bob { 0%,60%,100% { opacity:.28; transform: translateY(0); }
                 30% { opacity:1; transform: translateY(-4px); } }

/* ---------- composer ---------- */
footer {
  flex: 0 0 auto; background: var(--surface); border-top: 1px solid var(--line);
  padding: 14px 22px 18px;
}
.composer { max-width: 780px; margin: 0 auto; display: flex; gap: 10px;
            align-items: flex-end; }
textarea {
  flex: 1 1 auto; resize: none; font: inherit; color: var(--ink);
  background: var(--bg); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 12px 15px; line-height: 1.5;
  max-height: 38vh; min-height: 48px; outline: none;
  transition: border-color .12s ease, box-shadow .12s ease;
}
textarea:focus { border-color: var(--accent);
                 box-shadow: 0 0 0 3px var(--accent-soft); }
button.send {
  flex: 0 0 auto; background: var(--accent); color: var(--accent-ink);
  border: 0; border-radius: var(--radius); padding: 13px 20px;
  font: inherit; font-weight: 650; cursor: pointer; min-height: 48px;
  transition: all .12s ease;
}
button.send:hover:not(:disabled) { transform: translateY(-1px);
                                   box-shadow: var(--shadow); }
button.send:disabled { opacity: .45; cursor: not-allowed; }
.hint { max-width: 780px; margin: 8px auto 0; font-size: .75rem;
        color: var(--muted); display: flex; justify-content: space-between; }
.hint kbd {
  background: var(--bg); border: 1px solid var(--line); border-bottom-width: 2px;
  border-radius: 5px; padding: 1px 5px; font-family: inherit; font-size: .72rem;
}
.hint a { color: var(--muted); }
.hint a:hover { color: var(--accent); }
/* ---------- inline teach form ---------- */
.teach { margin-top: 10px; padding-top: 11px; border-top: 1px dashed var(--line);
         display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }
.teach .lbl { font-size: .8rem; color: var(--muted); width: 100%; }
.teach input {
  font: inherit; font-size: .86rem; padding: 7px 10px; min-width: 0;
  background: var(--bg); border: 1px solid var(--line);
  border-radius: 8px; color: var(--ink); outline: none;
}
.teach input:focus { border-color: var(--accent); }
.teach input.rel { flex: 0 1 130px; }
.teach input.obj { flex: 1 1 160px; }
.teach button {
  font: inherit; font-size: .84rem; font-weight: 650; cursor: pointer;
  background: var(--accent); color: var(--accent-ink); border: 0;
  border-radius: 8px; padding: 8px 13px;
}
.teach button:disabled { opacity: .5; cursor: default; }
.teach .done { font-size: .82rem; color: #1a7f4b; font-weight: 600; }

.sr { position: absolute; width: 1px; height: 1px; overflow: hidden;
      clip: rect(0 0 0 0); }
@media (max-width: 620px) {
  header, footer, main { padding-left: 14px; padding-right: 14px; }
  .meta .hide-sm { display: none; }
  .bubble { max-width: 88%; }
}
</style>
</head>
<body>

<header>
  <div class="brand"><span class="dot" id="dot"></span> Eddi
    <span class="pill" id="ver">__VERSION__</span></div>
  <div class="meta">
    <span class="hide-sm"><b id="facts">0</b> facts</span>
    <span class="hide-sm"><b id="toks">0</b> tokens</span>
    <span id="state" class="hide-sm">connecting</span>
  </div>
</header>

<main id="main">
  <div class="thread" id="thread"></div>
  <div class="empty" id="empty">
    <h1>Talk to Eddi</h1>
    <p>Writable memory, no training needed for the facts below.
       Ask something, or start from one of these.</p>
    <div class="chips">
      <button class="chip">what is a hypervector?</button>
      <button class="chip">what is python?</button>
      <button class="chip">tell me about water</button>
      <button class="chip">what is the capital of france?</button>
      <button class="chip">what is 12 * 37?</button>
      <button class="chip">what is the speed of light?</button>
    </div>
  </div>
</main>

<footer>
  <form class="composer" id="form">
    <label class="sr" for="box">Message</label>
    <textarea id="box" rows="1" placeholder="Ask Eddi something&hellip;"
              autocomplete="off" spellcheck="false"></textarea>
    <button class="send" id="send" type="submit">Send</button>
  </form>
  <div class="hint">
    <span><kbd>Enter</kbd> to send &middot; <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line</span>
    <span><a href="/stats" target="_blank" rel="noopener">brain stats</a> &middot;
          <a href="#" id="clear">clear chat</a></span>
  </div>
</footer>

<script>
(() => {
  const thread = document.getElementById('thread');
  const empty  = document.getElementById('empty');
  const form   = document.getElementById('form');
  const box    = document.getElementById('box');
  const send   = document.getElementById('send');
  const dot    = document.getElementById('dot');
  const state  = document.getElementById('state');
  const main   = document.getElementById('main');

  const el = (tag, cls) => { const n = document.createElement(tag);
                             if (cls) n.className = cls; return n; };

  function scrollDown() { main.scrollTop = main.scrollHeight; }

  function turn(role, text, meta) {
    empty.style.display = 'none';
    const wrap = el('div', 'turn ' + role);
    const av   = el('div', 'avatar');
    av.textContent = role === 'user' ? 'you' : 'E';
    const body = el('div');
    const bub  = el('div', 'bubble');
    bub.textContent = text;
    body.appendChild(bub);
    if (meta) {
      const foot = el('div', 'foot');
      if (meta.kind) {
        const k = el('span', 'tag');
        k.textContent = meta.kind;
        if (meta.kind === 'unknown') k.classList.add('warn');
        foot.appendChild(k);
      }
      if (meta.confidence !== undefined && meta.confidence > 0) {
        const c = el('span', 'tag');
        c.textContent = Math.round(meta.confidence * 100) + '%';
        foot.appendChild(c);
      }
      body.appendChild(foot);
    }
    wrap.appendChild(av);
    wrap.appendChild(body);
    thread.appendChild(wrap);
    scrollDown();
    if (meta && meta.kind === 'unknown') teachForm(body, text);
    return body;
  }

  // When Eddi says it does not know a subject, the CLI-style advice it
  // returns ("try 'teach X <relation> <value>'") is not something a browser
  // user can do. Offer the actual form instead, so the answer is actionable.
  function teachForm(body, text) {
    const m = /don't know (?:anything about )?'([^']+)'/i.exec(text);
    const subject = m ? m[1] : '';
    const box = el('div', 'teach');
    const lbl = el('span', 'lbl');
    lbl.textContent = 'Teach Eddi a fact about ' + (subject || 'this') + ':';
    const rel = el('input', 'rel');
    rel.placeholder = 'relation';
    rel.setAttribute('aria-label', 'relation');
    const obj = el('input', 'obj');
    obj.placeholder = 'value';
    obj.setAttribute('aria-label', 'value');
    const btn = el('button');
    btn.type = 'button';
    btn.textContent = 'Teach';
    box.appendChild(lbl); box.appendChild(rel); box.appendChild(obj); box.appendChild(btn);
    body.appendChild(box);
    rel.focus();

    const submit = async () => {
      const r = rel.value.trim(), o = obj.value.trim();
      if (!r || !o) { (r ? obj : rel).focus(); return; }
      btn.disabled = true;
      try {
        await post('/teach', { subject, relation: r, object: o });
        const done = el('span', 'done');
        done.textContent = 'Learned. Ask again.';
        box.innerHTML = '';
        box.appendChild(done);
      } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Failed';
      }
    };
    btn.addEventListener('click', submit);
    [rel, obj].forEach(i => i.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); submit(); }
    }));
  }

  function pending() {
    const wrap = el('div', 'turn eddi');
    const av   = el('div', 'avatar');
    av.textContent = 'E';
    const body = el('div');
    const bub  = el('div', 'bubble');
    const t    = el('span', 'typing');
    t.innerHTML = '<i></i><i></i><i></i>';
    bub.appendChild(t);
    body.appendChild(bub);
    wrap.appendChild(av);
    wrap.appendChild(body);
    thread.appendChild(wrap);
    empty.style.display = 'none';
    scrollDown();
    return wrap;
  }

  async function post(path, payload) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
    return data;
  }

  async function ask(text) {
    turn('user', text);
    const waiter = pending();
    send.disabled = true;
    try {
      const data = await post('/ask', { text });
      waiter.remove();
      turn('eddi', data.text || '(no answer)', data);
    } catch (err) {
      waiter.remove();
      const body = turn('eddi', 'Could not reach Eddi: ' + err.message);
      body.querySelector('.bubble').style.borderColor = '#d14343';
    } finally {
      send.disabled = false;
      box.focus();
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = box.value.trim();
    if (!text) return;
    box.value = '';
    box.style.height = 'auto';
    ask(text);
  });

  box.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  box.addEventListener('input', () => {
    box.style.height = 'auto';
    box.style.height = Math.min(box.scrollHeight, window.innerHeight * .38) + 'px';
  });

  document.querySelectorAll('.chip').forEach((c) =>
    c.addEventListener('click', () => { box.value = c.textContent; ask(c.textContent); }));

  document.getElementById('clear').addEventListener('click', (e) => {
    e.preventDefault();
    thread.innerHTML = '';
    empty.style.display = '';
  });

  async function health() {
    try {
      const [h, s] = await Promise.all([
        fetch('/health').then(r => r.json()),
        fetch('/stats').then(r => r.json()),
      ]);
      dot.classList.remove('off');
      state.textContent = 'online';
      document.getElementById('facts').textContent = s.facts ?? 0;
      document.getElementById('toks').textContent  = s.trained_tokens ?? 0;
      if (h.version) document.getElementById('ver').textContent = h.version;
    } catch (e) {
      dot.classList.add('off');
      state.textContent = 'offline';
    }
  }
  health();
  setInterval(health, 15000);
  box.focus();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class EddiHandler(BaseHTTPRequestHandler):
    """Serves the page and a small JSON API. Stateless by design."""

    agent: EddiAgent
    brain_path: Optional[str] = None
    save_on_change: bool = True

    server_version = f"Eddi/{__version__}"

    # -- helpers ----------------------------------------------------------
    def _payload(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("request too large")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError("invalid json") from exc
        return parsed if isinstance(parsed, dict) else {}

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _maybe_save(self) -> None:
        """Persist after a write, so a taught fact survives a restart."""
        if not (self.save_on_change and self.brain_path):
            return
        try:
            save_brain(self.agent, self.brain_path)
        except Exception:  # a failed save must not lose the HTTP response
            pass

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            page = PAGE.replace("__TOKENS__", _tokens()).replace("__VERSION__", __version__)
            self._html(page)
        elif path == "/health":
            self._json({"status": "ok", "version": __version__})
        elif path == "/stats":
            self._json(self.agent.stats())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        try:
            payload = self._payload()
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
            return

        if path == "/ask":
            text = str(payload.get("text", "")).strip()
            if not text:
                self._json({"error": "missing text"}, 400)
                return
            # One shared lock keeps the agent single-threaded while many
            # requests are in flight; the agent is not thread-safe.
            with self.server.lock:  # type: ignore[attr-defined]
                answer = self.agent.answer(text)
            self._json(
                {
                    "text": answer.text,
                    "kind": answer.kind,
                    "confidence": float(answer.confidence or 0.0),
                }
            )
            return

        if path == "/ingest":
            text = str(payload.get("text", "")).strip()
            if not text:
                self._json({"error": "missing text"}, 400)
                return
            with self.server.lock:  # type: ignore[attr-defined]
                learned = self.agent.ingest_text(text)
            self._maybe_save()
            self._json({"ok": True, "tokens_learned": learned})
            return

        if path == "/teach":
            subject = str(payload.get("subject", "")).strip()
            relation = str(payload.get("relation", "")).strip()
            obj = str(payload.get("object", "")).strip()
            if not (subject and relation and obj):
                self._json({"error": "subject, relation and object are required"}, 400)
                return
            with self.server.lock:  # type: ignore[attr-defined]
                written = self.agent.teach(subject, relation, obj)
            self._maybe_save()
            self._json({"ok": True, "facts": written})
            return

        if path == "/reset":
            # The transcript lives in the browser; this only acknowledges.
            self._json({"ok": True})
            return

        self._json({"error": "not found"}, 404)

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server API
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet by default; errors still surface through the response."""
        if os.environ.get("EDDI_HTTP_LOG"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class EddiServer(ThreadingHTTPServer):
    """Threaded so a slow answer cannot block the health check."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, agent: EddiAgent, brain_path: Optional[str],
                 save_on_change: bool = True):
        import threading

        super().__init__(address, handler)
        self.agent = agent
        self.lock = threading.Lock()
        handler.agent = agent
        handler.brain_path = brain_path
        handler.save_on_change = save_on_change


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eddi-webserver", description="Serve the Eddi chat UI."
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address (default 0.0.0.0, so previews work)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--brain", default=None, help="path to a saved brain .pkl")
    parser.add_argument("--no-save", action="store_true",
                        help="do not persist taught facts back to the brain")
    args = parser.parse_args(argv)

    brain = args.brain or default_brain_path()
    if brain_exists(brain):
        agent = load_brain(brain)
        print(f"loaded brain: {brain}")
    else:
        agent = EddiAgent(brain_path=brain)
        print(f"no brain at {brain} - starting fresh (train with: python -m eddi.cli --train)")

    server = EddiServer(
        (args.host, args.port), EddiHandler, agent, brain, save_on_change=not args.no_save
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Eddi UI on {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        agent.close(save=not args.no_save)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(main())
