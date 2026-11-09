const $ = (id) => document.getElementById(id);
const els = {
  surah: $("surah"), from: $("from"), to: $("to"), learnerLevel: $("learner-level"), mic: $("mic"), micLabel: $("mic-label"),
  file: $("file"), fileLabel: $("file-label"), level: $("level"), status: $("status"),
  mushaf: $("mushaf"), feedback: $("feedback"), retry: $("retry"), transcript: $("transcript"),
  stats: $("stats"), models: $("models"), tip: $("tip"),
};

let surahs = [];
let reference = [];
let wordEls = [];
let ws = null;
let source = null;          // {stop()} for the active mic or file stream
let asrReady = false;

const arabicDigits = (n) => String(n).replace(/\d/g, (d) => "٠١٢٣٤٥٦٧٨٩"[d]);

// ---------- setup -----------------------------------------------------------

async function init() {
  surahs = await (await fetch("/api/surahs")).json();
  els.surah.innerHTML = surahs
    .map((s) => `<option value="${s.number}">${s.number}. ${s.english_name} — ${s.name}</option>`)
    .join("");
  try { els.learnerLevel.value = localStorage.getItem("learnerLevel") || "beginner"; } catch {}
  els.learnerLevel.addEventListener("change", () => {
    try { localStorage.setItem("learnerLevel", els.learnerLevel.value); } catch {}
  });
  els.surah.addEventListener("change", onSurahChange);
  els.from.addEventListener("change", loadReference);
  els.to.addEventListener("change", loadReference);
  els.mic.addEventListener("click", () => (source ? stop() : startMic()));
  els.file.addEventListener("change", () => els.file.files[0] && startFile(els.file.files[0]));
  onSurahChange();
  pollHealth();
}

function onSurahChange() {
  const s = surahs.find((x) => x.number == els.surah.value);
  els.from.max = els.to.max = s.ayahs;
  els.from.value = 1;
  els.to.value = s.ayahs;
  loadReference();
}

async function loadReference() {
  const from = clamp(+els.from.value, 1, +els.to.max);
  const to = clamp(+els.to.value, from, +els.to.max);
  els.from.value = from;
  els.to.value = to;
  const r = await fetch(`/api/reference?surah=${els.surah.value}&start=${from}&end=${to}`);
  renderReference(await r.json());
  resetPanels();
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v || lo));

async function pollHealth() {
  const h = await (await fetch("/api/health")).json();
  const m = h.models;
  els.models.innerHTML = Object.entries(m)
    .map(([k, v]) => {
      const cls = v === "ready" ? "ready" : v.startsWith("error") ? "error" : "";
      return `<span class="pill ${cls}" title="${escapeHtml(v)}">${k === "asr" ? "ASR" : "Pronunciation"}: ${v.split(":")[0]}</span>`;
    })
    .join("");
  asrReady = m.asr === "ready";
  setIdle();
  if (m.asr === "loading" || m.pronunciation === "loading") setTimeout(pollHealth, 2000);
}

function setIdle() {
  if (source) return;
  els.mic.disabled = !asrReady;
  els.file.disabled = !asrReady;
  els.fileLabel.classList.toggle("disabled", !asrReady);
  els.status.textContent = asrReady
    ? "Ready. Press Start and recite from the first highlighted word."
    : "Loading models… (first run downloads them)";
}

// ---------- rendering -------------------------------------------------------

function renderReference(words) {
  reference = words;
  els.mushaf.innerHTML = "";
  wordEls = words.map((w, i) => {
    const span = document.createElement("span");
    span.className = "w";
    span.dataset.i = i;
    w.clusters.forEach((c) => {
      const cs = document.createElement("span");
      cs.textContent = c;
      span.appendChild(cs);
    });
    els.mushaf.appendChild(span);
    const next = words[i + 1];
    if (!next || next.ayah !== w.ayah) {
      const end = document.createElement("span");
      end.className = "ayah-end";
      end.textContent = `﴿${arabicDigits(w.ayah)}﴾`;
      els.mushaf.appendChild(end);
    }
    els.mushaf.appendChild(document.createTextNode(" "));
    span.addEventListener("mouseenter", (e) => showTip(e, i));
    span.addEventListener("mouseleave", () => (els.tip.hidden = true));
    return span;
  });
  if (wordEls[0]) wordEls[0].classList.add("cursor");
}

let lastState = null;

function renderState(st) {
  lastState = st;
  els.mushaf.querySelectorAll(".extra-chip").forEach((e) => e.remove());
  st.words.forEach((w, i) => {
    const el = wordEls[i];
    el.className = `w ${w.status}${w.tentative ? " tentative" : ""}`;
    const bad = new Set(w.letters.filter((l) => l.flagged).map((l) => l.cluster));
    [...el.children].forEach((c, k) => c.classList.toggle("bad", bad.has(k)));
  });
  if (st.type === "state" && wordEls[st.position]) wordEls[st.position].classList.add("cursor");
  const lastChip = new Map();   // keeps several extras at one spot in spoken order
  st.extras.forEach((e) => {
    const chip = document.createElement("span");
    chip.className = `extra-chip${e.tentative ? " tentative" : ""}`;
    chip.dir = "rtl";
    chip.textContent = `+ ${e.heard}`;
    chip.title = "Extra word";
    const prev = lastChip.get(e.after);
    const anchor = prev || (e.after >= 0 ? wordEls[e.after] : null);
    if (anchor) anchor.after(chip);
    else els.mushaf.prepend(chip);
    lastChip.set(e.after, chip);
  });

  const fb = st.feedback;
  els.feedback.innerHTML = fb.length
    ? fb.map((m) => `<li class="${m.severity}" data-i="${m.ref_index}">${withArabic(m.text)}</li>`).join("")
    : st.type === "final"
      ? `<li class="good">No mistakes detected. Well done.</li>`
      : `<li class="empty">No mistakes so far.</li>`;
  els.feedback.querySelectorAll("li[data-i]").forEach((li) =>
    li.addEventListener("click", () => focusWord(+li.dataset.i)));

  els.transcript.innerHTML = `${escapeHtml(st.transcript.committed)} <span class="partial">${escapeHtml(st.transcript.partial)}</span>`;

  const s = st.stats;
  const fmt = (v, unit) => (v == null ? "—" : `${v}${unit}`);
  els.stats.innerHTML = `
    <dt>Audio received</dt><dd>${fmt(s.audio_seconds, " s")}</dd>
    <dt>Feedback latency p50</dt><dd>${fmt(s.latency_ms_p50, " ms")}</dd>
    <dt>Feedback latency p90</dt><dd>${fmt(s.latency_ms_p90, " ms")}</dd>
    <dt>ASR decode p50</dt><dd>${fmt(s.decode_ms_p50, " ms")}</dd>`;

  if (st.type === "final") renderRetry(st);
}

function renderRetry(st) {
  const ayahs = [...new Set(st.feedback.filter((m) => m.kind !== "incomplete").map((m) => reference[m.ref_index].ayah))];
  els.retry.innerHTML = ayahs.map((a) => `<button data-a="${a}">Retry ayah ${a}</button>`).join("");
  els.retry.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", async () => {
      els.from.value = els.to.value = b.dataset.a;
      await loadReference();
      startMic();
    }));
}

function resetPanels() {
  els.feedback.innerHTML = `<li class="empty">Recite to see corrections here.</li>`;
  els.retry.innerHTML = "";
  els.transcript.textContent = "";
  els.stats.innerHTML = "";
  lastState = null;
}

function focusWord(i) {
  const el = wordEls[i];
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

function showTip(e, i) {
  if (!lastState) return;
  const w = lastState.words[i];
  // U+2068/U+2069 isolate Arabic runs so bidi does not swap them with the numbers beside them.
  const iso = (s) => String.fromCharCode(0x2068) + s + String.fromCharCode(0x2069);
  const lines = [`Status: ${w.status.replace("_", " ")}`];
  if (w.status === "checking") lines.push("Checking the audio against the expected word…");
  if (w.verified) lines.push(`Transcribed as ${iso(w.heard)}, but the audio matches the expected word`);
  else if (w.heard) lines.push(`Heard: ${iso(w.heard)}${w.similarity != null && w.similarity < 1 ? ` (${Math.round(w.similarity * 100)}% match)` : ""}`);
  const flagged = w.letters.filter((l) => l.flagged);
  if (flagged.length) lines.push(`Weak letters: ${flagged.map((l) => `${iso(l.letter)} (${l.score.toFixed(2)})`).join(", ")}`);
  const eased = w.letters.filter((l) => !l.flagged && l.level_score != null && l.score < 0.35);
  if (eased.length) lines.push(`Accepted at this level: ${eased.map((l) => `${iso(l.letter)} (exact ${l.score.toFixed(2)}, level ${l.level_score.toFixed(2)})`).join(", ")}`);
  else if (w.letters.length) {
    const min = Math.min(...w.letters.map((l) => l.score));
    lines.push(`Lowest letter score: ${min.toFixed(2)}`);
  }
  els.tip.textContent = lines.join("\n");
  els.tip.style.whiteSpace = "pre-line";
  const r = e.target.closest(".w").getBoundingClientRect();
  els.tip.style.left = `${Math.max(8, Math.min(window.innerWidth - 290, r.left))}px`;
  els.tip.style.top = `${r.bottom + 6}px`;
  els.tip.hidden = false;
}

// Quoted Arabic inside English messages is isolated so bidi reordering keeps it intact.
const withArabic = (s) =>
  escapeHtml(s).replace(/«([^»]*)»/g, '«<bdi dir="rtl" lang="ar">$1</bdi>»');

const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------- streaming -------------------------------------------------------

function openSocket() {
  return new Promise((resolve, reject) => {
    const sock = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    sock.binaryType = "arraybuffer";
    sock.onmessage = (ev) => onMessage(JSON.parse(ev.data));
    sock.onerror = () => reject(new Error("could not reach the server"));
    sock.onopen = () => {
      sock.send(JSON.stringify({ type: "start", surah: +els.surah.value, from: +els.from.value, to: +els.to.value,
                                 level: els.learnerLevel.value }));
    };
    sock.onclose = () => { if (ws === sock) ws = null; };
    sock.readyResolve = resolve;
    sock.readyReject = reject;
    ws = sock;
  });
}

function onMessage(msg) {
  if (msg.type === "ready") {
    renderReference(msg.reference);
    resetPanels();
    ws.readyResolve();
  } else if (msg.type === "state" || msg.type === "final") {
    renderState(msg);
    if (msg.type === "final") {
      els.status.textContent = "Finished. Hover a word for details, or retry an ayah.";
      ws?.close();
    }
  } else if (msg.type === "error") {
    ws?.readyReject(new Error(msg.message));
    stop();
    els.status.textContent = `Error: ${msg.message}`;
  }
}

async function startMic() {
  if (source) return;
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule("pcm-worklet.js");
    const node = new AudioWorkletNode(ctx, "pcm-worklet");
    const input = ctx.createMediaStreamSource(stream);
    input.connect(node);
    await openSocket();
    node.port.onmessage = (e) => {
      if (ws?.readyState === 1) ws.send(e.data);
      showLevel(new Int16Array(e.data));
    };
    source = {
      stop() {
        input.disconnect();
        node.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        ctx.close();
      },
    };
    setLive("Listening… recite at your normal pace.");
  } catch (err) {
    stream?.getTracks().forEach((t) => t.stop());
    ws?.close();
    els.status.textContent = `Could not start: ${err.message || err}`;
  }
}

async function startFile(file) {
  if (source) return;
  const buf = await file.arrayBuffer();
  const tmp = new AudioContext();
  const decoded = await tmp.decodeAudioData(buf.slice(0));
  tmp.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  try {
    await openSocket();
  } catch (err) {
    ws?.close();
    els.status.textContent = `Could not start: ${err.message || err}`;
    return;
  }

  const audio = new Audio(URL.createObjectURL(file));
  audio.play().catch(() => {});
  // Send 100 ms chunks at real-time pace, like a live microphone. Each tick sends
  // everything due by the wall clock: browsers throttle timers in background tabs,
  // and one chunk per tick would then stream far slower than real time.
  let k = 0;
  const started = performance.now();
  const timer = setInterval(() => {
    const due = Math.min(pcm.length, Math.floor(((performance.now() - started) / 1000) * 16000));
    while (k + 1600 <= due || (due === pcm.length && k < pcm.length)) {
      const chunk = pcm.subarray(k, k + 1600);
      k += chunk.length;
      const out = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) out[i] = Math.max(-1, Math.min(1, chunk[i])) * 0x7fff;
      if (ws?.readyState === 1) ws.send(out.buffer);
      showLevel(out);
    }
    if (k >= pcm.length) stop();
  }, 100);
  source = { stop() { clearInterval(timer); audio.pause(); } };
  els.file.value = "";
  setLive(`Streaming ${file.name}…`);
}

function setLive(text) {
  els.mic.classList.add("live");
  els.mic.disabled = false;
  els.micLabel.textContent = "Stop";
  els.file.disabled = true;
  els.fileLabel.classList.add("disabled");
  els.status.textContent = text;
}

function stop() {
  if (!source) return;
  source.stop();
  source = null;
  els.mic.classList.remove("live");
  els.micLabel.textContent = "Start reciting";
  els.level.style.width = "0";
  if (ws?.readyState === 1) {
    ws.send(JSON.stringify({ type: "stop" }));
    els.status.textContent = "Finishing analysis…";
  }
  els.file.disabled = !asrReady;
  els.fileLabel.classList.toggle("disabled", !asrReady);
}

function showLevel(pcm) {
  let s = 0;
  for (let i = 0; i < pcm.length; i++) s += (pcm[i] / 32768) ** 2;
  const db = 10 * Math.log10(s / pcm.length + 1e-10);
  els.level.style.width = `${Math.max(0, Math.min(100, (db + 60) * 2))}%`;
}

init();
