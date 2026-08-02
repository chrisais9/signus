"use strict";
const $ = (id) => document.getElementById(id);
const API = "/api/analyze";
const SURVEY_API = "/api/survey";
const FS_RE = /(?:^|[._])fs(\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:[._]|$)/i;
const RF_RE = /(?:^|[._])rf(\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:[._]|$)/i;
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const state = { file: null, sidecar: null, meta: null, resp: null, batch: [],
                burst: null, survey: null, drilled: false, overview: null };

/* --- filename parsing (mirror of sigio.parse_name; read-necessities only) --- */
const DTYPES = ["i8", "u8", "i16", "u16", "f32", "f64"];
// baudline-style aliases: <bits>t = two's complement (signed), <bits>o = offset (unsigned)
const DTYPE_ALIAS = { "8t": "i8", "8o": "u8", "16t": "i16", "16o": "u16", "32f": "f32", "64f": "f64" };
const FMTS = { cplx: "iq", real: "real", iq: "iq" };   // iq = 옛 밑줄 형식의 이름
/* <이름>.cplx|real.<샘플레이트>.<16t>.pcm — 샘플레이트는 접두사가 없으므로 포맷 바로
   다음 자리로 찾는다. 옛 밑줄 형식(<이름>_fs…_iq_i16.iq)도 그대로 읽는다. sigio.py 와 규칙이
   같아야 한다: 여기서 갈리면 웹은 되고 CLI 는 안 되는(또는 그 반대) 조용한 불일치가 된다. */
function parseName(name) {
  const base = name.replace(/^.*\//, "").toLowerCase();
  const mt = FS_RE.exec(base), rt = RF_RE.exec(base), toks = base.split(/[._]/);
  // 진짜 포맷 토막 = "숫자 + 아는 제원 토막"을 데리고 다니는 마지막 후보. 라벨의 real/cplx/
  // u8/be 오염과 점 낀 샘플레이트(2.4e6 -> 2 Hz)를 같은 관문으로 막는다 (sigio.py 와 동일 규칙).
  const spec = (t) => DTYPES.includes(t) || t in DTYPE_ALIAS || t === "be" || t === "bitrev"
                      || t === "pcm" || t.startsWith("rf");
  const hit = (k) => /^\d+$/.test(toks[k + 1] || "") && spec(toks[k + 2] || "");
  const cands = toks.map((t, k) => (t in FMTS ? k : -1)).filter((k) => k >= 0);
  const hits = cands.filter(hit);
  const i = hits.length ? hits[hits.length - 1] : (cands.length ? cands[0] : -1);
  const tail = i >= 0 ? toks.slice(i + 1) : toks;   // 제원은 포맷 토막 뒤에만 산다
  const dt = tail.find((t) => DTYPES.includes(t) || t in DTYPE_ALIAS);
  return {
    fs: hits.includes(i) ? parseFloat(toks[i + 1]) : (mt ? parseFloat(mt[1]) : null),
    fmt: i >= 0 ? FMTS[toks[i]] : null,
    dtype: dt ? (DTYPE_ALIAS[dt] || dt) : "i16",
    endian: tail.includes("be") ? "be" : "le",
    bitrev: tail.includes("bitrev"),
    rf: rt ? parseFloat(rt[1]) : null,
  };
}
/* SigMF: core:datatype -> our fmt/dtype/endian */
const SIGMF = { cf32: ["iq", "f32"], ci16: ["iq", "i16"], ci8: ["iq", "i8"],
                cu8: ["iq", "u8"], rf32: ["real", "f32"], ri16: ["real", "i16"] };
function metaFromSigmf(doc) {
  const g = doc && doc.global;
  if (!g || !g["core:datatype"]) return null;
  const dt = g["core:datatype"], base = dt.replace(/_[lb]e$/, "");
  if (!(base in SIGMF)) return null;
  const [fmt, dtype] = SIGMF[base];
  const cap = (doc.captures || [])[0] || {};
  return { fs: Number(g["core:sample_rate"]), fmt, dtype,
           endian: dt.endsWith("_be") ? "be" : "le", bitrev: false,
           rf: cap["core:frequency"] != null ? Number(cap["core:frequency"]) : null };
}

/* --- number formatting --- */
function sig(x) {
  const a = Math.abs(x);
  return parseFloat(a >= 100 ? x.toFixed(0) : a >= 10 ? x.toFixed(1) : x.toFixed(2)).toString();
}
function fmtHz(v) {
  if (v == null) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return sig(v / 1e6) + " MHz";
  if (a >= 1e3) return sig(v / 1e3) + " kHz";
  return sig(v) + " Hz";
}
function fmtRf(v) {   // absolute RF: keep kHz resolution even in the 100s-of-MHz range
  return (v / 1e6).toFixed(3) + " MHz";
}

/* --- ideal constellation points (mirror of constellations.ideal_points) --- */
function idealPoints(mod) {
  if (mod === "bpsk") return [{ i: 1, q: 0 }, { i: -1, q: 0 }];
  if (mod === "qpsk" || mod === "8psk") {
    const m = mod === "qpsk" ? 4 : 8, off = mod === "qpsk" ? Math.PI / 4 : 0, p = [];
    for (let k = 0; k < m; k++) p.push({ i: Math.cos(off + 2 * Math.PI * k / m), q: Math.sin(off + 2 * Math.PI * k / m) });
    return p;
  }
  const n = mod === "16qam" ? 4 : mod === "32qam" ? 6 : 8, lv = [], p = [];
  for (let k = 0; k < n; k++) lv.push(k * 2 - (n - 1));
  for (const qi of lv) for (const ii of lv) {
    if (mod === "32qam" && Math.abs(ii * qi) === 25) continue;  // cross: corners removed
    p.push({ i: ii, q: qi });
  }
  let e = 0;
  for (const pt of p) e += pt.i * pt.i + pt.q * pt.q;
  const norm = Math.sqrt(e / p.length);
  return p.map((pt) => ({ i: pt.i / norm, q: pt.q / norm }));
}

/* --- file intake: one data file (+sidecar) or many (batch) --- */
async function acceptFiles(list) {
  const files = [...list];
  const metas = files.filter((f) => /\.(json|sigmf-meta)$/i.test(f.name));
  const data = files.filter((f) => !metas.includes(f));
  if (!data.length) return showError("데이터 파일을 찾을 수 없어요.");
  if (data.length > 1) return runBatch(data, metas);

  state.file = data[0];
  state.resp = null;
  state.batch = [];
  state.burst = null;
  state.survey = null;
  state.drilled = false;
  state.overview = null;
  // 사이드카는 이름이 같은 캡처의 것만 쓴다 -- 확장자만 보고 집으면 다른 신호의 fs/fmt 로
  // 조용히 읽는다 (일괄 모드는 이미 stem 대조를 한다: 두 모드가 다르면 그게 또 함정이다)
  const mine = (m, ext) => {
    const stem = m.name.toLowerCase().slice(0, -ext.length);
    const dn = state.file.name.toLowerCase();
    return dn.startsWith(stem) && (dn.length === stem.length || dn[stem.length] === ".");
  };
  const sm = metas.find((m) => m.name.toLowerCase().endsWith(".sigmf-meta") && mine(m, ".sigmf-meta"));
  const truth = metas.find((m) => m.name.toLowerCase().endsWith(".json") && mine(m, ".json"));
  state.sidecar = truth ? await readJson(truth) : null;   // client-side only, never sent
  state.meta = (sm && metaFromSigmf(await readJson(sm))) || parseName(state.file.name);
  if (state.meta && state.meta.fs && state.meta.fmt) runSurvey();
  else showMetaForm();
}
const readJson = (f) => f.text().then(JSON.parse).catch(() => null);

function showMetaForm() {
  hideAll();
  state.meta = state.meta || {};
  if (state.meta.fs) $("mFs").value = state.meta.fs;
  $("mFmt").value = state.meta.fmt || "";
  $("mDtype").value = state.meta.dtype || "i16";
  show($("metaForm"));
}
$("metaGo").onclick = () => {
  const fs = parseFloat($("mFs").value), fmt = $("mFmt").value;
  if (!(fs > 0) || !fmt) return showError("샘플레이트와 포맷을 입력해주세요.");
  state.meta = { fs, fmt, dtype: $("mDtype").value, endian: "le", bitrev: false };
  runSurvey();
};

/* --- server call (contract-frozen) --- */
function query(file, m, burst) {
  const q = new URLSearchParams({ name: file.name });
  if (m.fs) q.set("fs", m.fs);
  if (m.fmt) q.set("fmt", m.fmt);
  if (m.dtype) q.set("dtype", m.dtype);
  if (m.endian) q.set("endian", m.endian);
  if (m.bitrev) q.set("bitrev", "1");
  if (m.rf != null) q.set("rf", m.rf);
  if (burst != null) q.set("burst", burst);
  return q;
}
async function postTo(api, file, m, burst) {
  const res = await fetch(api + "?" + query(file, m, burst), { method: "POST", body: file });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `서버 오류 (${res.status})`);
  return data;
}
async function runSurvey() {                  // entry: one capture -> single detail OR survey
  hideAll();
  state.drilled = false;
  $("loadMsg").textContent = "신호를 조사하고 있어요…";
  show($("loading"));
  try {
    const resp = await postTo(SURVEY_API, state.file, state.meta);
    if (resp.mode === "single") {
      state.survey = null; state.overview = resp.overview || null; state.resp = resp.result;
      render(resp.result);
      if (state.overview) renderBurstMap(state.overview, resp.result);
    } else renderSurvey(resp);
  } catch (e) {
    showError(e.message);
  }
}
async function analyze() {                     // burst re-selection within a single signal
  hideAll();
  $("loadMsg").textContent = "신호를 분석하고 있어요…";
  show($("loading"));
  try {
    state.resp = await postTo(API, state.file, state.meta, state.burst);
    render(state.resp);
    if (state.overview) renderBurstMap(state.overview, state.resp);
    $("backBatch").classList.toggle("hidden", !state.batch.length);
  } catch (e) {
    showError(e.message);
  }
}

/* --- batch mode --- */
async function runBatch(data, metas) {
  hideAll();
  show($("loading"));
  const truths = new Map(metas.filter((m) => /\.json$/i.test(m.name))
    .map((m) => [m.name.replace(/\.json$/i, ""), m]));
  const sigmfs = new Map(metas.filter((m) => /\.sigmf-meta$/i.test(m.name))
    .map((m) => [m.name.replace(/\.sigmf-meta$/i, ""), m]));
  const rows = [];
  for (let i = 0; i < data.length; i++) {
    const f = data[i];
    $("loadMsg").textContent = `일괄 분석 중… (${i + 1}/${data.length}) ${f.name}`;
    // same meta precedence as single-file mode: a .sigmf-meta sidecar (matched by stem)
    // beats filename tokens
    const sm = sigmfs.get(f.name.replace(/\.[^.]+$/, ""));
    const meta = (sm && metaFromSigmf(await readJson(sm))) || parseName(f.name);
    let resp = null, err = null;
    try {
      if (!meta.fs || !meta.fmt) throw new Error("파일명에 fs/포맷 정보가 없어요");
      resp = await postTo(API, f, meta);
    } catch (e) { err = e.message; }
    const tf = truths.get(f.name);
    // meta is stored per row: a later burst-chip re-analyze posts with THIS file's meta --
    // it used to read state.meta, which a fresh batch never set (crash) and a previous
    // single-file run left stale (silent wrong-fs decode)
    rows.push({ file: f, meta, resp, err, sidecar: tf ? await readJson(tf) : null });
  }
  state.batch = rows;
  hideAll();
  renderBatch(rows);
}

function renderBatch(rows) {
  $("batchBody").innerHTML = rows.map((r, i) => {
    if (r.err) return `<tr data-i="${i}"><td class="fn">${r.file.name}</td>` +
      `<td colspan="4" style="color:var(--red)">${r.err}</td></tr>`;
    const d = r.resp.detected, lock = Math.round(r.resp.quality.lock);
    const cls = lock >= 60 ? "ok" : lock >= 40 ? "warn" : "bad";
    return `<tr data-i="${i}"><td class="fn">${r.file.name}</td>` +
      `<td>${d.mod.toUpperCase()}</td><td>${fmtHz(d.fc)}</td><td>${fmtHz(d.baud)}</td>` +
      `<td><span class="pill ${cls}">${lock}</span></td></tr>`;
  }).join("");
  $("batchBody").querySelectorAll("tr").forEach((tr) => {
    tr.onclick = () => {
      const r = state.batch[+tr.dataset.i];
      if (!r.resp) return;
      state.file = r.file; state.meta = r.meta; state.resp = r.resp; state.sidecar = r.sidecar;
      state.burst = null; state.drilled = false; state.survey = null; state.overview = null;
      hideAll();
      render(r.resp);
      const b = $("backBatch");
      b.textContent = "← 목록으로";
      b.onclick = () => { hideAll(); renderBatch(state.batch); };
      b.classList.remove("hidden");
    };
  });
  show($("batchCard"));
}
$("backBatch").onclick = () => { hideAll(); renderBatch(state.batch); };
$("dlCsv").onclick = () => {
  const head = "file,mod,fc_hz,baud_hz,rolloff,lock,mer_db,snr_est_db\n";
  const body = state.batch.filter((r) => r.resp).map((r) => {
    const d = r.resp.detected, q = r.resp.quality;
    return [r.file.name, d.mod, d.fc, d.baud, d.rolloff ?? "", q.lock, q.mer_db,
            r.resp.snr_est_db ?? ""].join(",");
  }).join("\n");
  saveBlob("signus_batch.csv", head + body, "text/csv");
};

/* --- wideband survey overview (waterfall + detected-emitter boxes) --- */
function renderSurvey(resp) {
  hideAll();
  stopPlay();
  state.survey = resp;
  const dig = resp.emitters.filter((e) => e.result).length;
  $("survCount").textContent = `${resp.emitters.length}개 신호 · 디지털 ${dig}`;
  $("survInfo").classList.add("hidden");
  renderSurveyList(resp);
  show($("surveyCard"));                // show first so the canvas has a real width
  paintWaterfall($("survFall"), resp.overview.waterfall, 230);
  drawBoxes(resp);
}

function boxStyle(e) {   // -> [css class, short label]
  if (e.result) {
    const lock = Math.round(e.result.quality.lock);
    return [lock >= 60 ? "ok" : lock >= 40 ? "warn" : "bad",
            `${e.result.detected.mod.toUpperCase()} · ${lock}`];
  }
  if (e.kind === "chirp") {
    return ["gray", e.info && e.info.sf ? `LoRa SF${e.info.sf}` : "처프"];
  }
  return ["gray", { analog: "아날로그", tone: "톤/CW", tooshort: "짧음",
                    error: "오류" }[e.kind] || e.kind];
}

function drawBoxes(resp) {
  const host = $("survBoxes"), fs = resp.overview.fs, n = resp.overview.n || 1;
  host.innerHTML = "";
  resp.emitters.forEach((e) => {
    const d = e.det, [cls, lbl] = boxStyle(e);
    const wpct = Math.max(0.8, d.bw / fs * 100);          // width  = bandwidth / fs
    const left = (d.fc + fs / 2) / fs * 100 - wpct / 2;   // x      = carrier on -fs/2..fs/2
    const top = Math.max(0, d.t0 / n * 100);              // y      = burst start over capture
    const b = document.createElement("button");
    b.className = "box " + cls;
    b.style.left = Math.max(0, Math.min(100 - wpct, left)) + "%";
    b.style.width = wpct + "%";
    b.style.top = top + "%";
    b.style.height = Math.min(100 - top, Math.max(7, (d.t1 - d.t0) / n * 100)) + "%";
    b.title = lbl;
    b.innerHTML = `<span class="box-lbl">${lbl}</span>`;
    b.onclick = () => drillEmitter(e);
    host.appendChild(b);
  });
}

function renderSurveyList(resp) {
  const KIND = { linear: "디지털", fsk: "FSK", chirp: "처프/LoRa", analog: "아날로그",
                 tone: "톤/CW", tooshort: "짧음", error: "오류" };
  const rf0 = resp.rf_center;                      // real RF = capture centre + abs_fc
  $("survBody").innerHTML = resp.emitters.map((e, i) => {
    const [cls] = boxStyle(e), r = e.result;
    const lock = r ? `<span class="pill ${cls}">${Math.round(r.quality.lock)}</span>`
                   : `<span class="pill gray">–</span>`;
    return `<tr data-i="${i}"><td class="fc">${rf0 != null ? fmtRf(rf0 + e.abs_fc) : fmtHz(e.abs_fc)}</td>` +
      `<td>${KIND[e.kind] || e.kind}</td><td>${r ? r.detected.mod.toUpperCase() : "–"}</td>` +
      `<td>${r ? fmtHz(r.detected.baud) : "–"}</td><td>${lock}</td></tr>`;
  }).join("");
  $("survBody").querySelectorAll("tr").forEach((tr) => {
    tr.onclick = () => drillEmitter(resp.emitters[+tr.dataset.i]);
  });
}

function drillEmitter(e) {
  if (e.result) {                       // digital -> reuse the single-signal detail view
    state.drilled = true;
    state.overview = null;              // a cut channel has no whole-capture burst map
    state.resp = e.result;
    render(e.result);
    const rf0 = state.survey && state.survey.rf_center;
    $("fileName").textContent =
      `에미터 @ ${rf0 != null ? fmtRf(rf0 + e.abs_fc) : fmtHz(e.abs_fc)} · ${state.file.name}`;
    const b = $("backBatch");
    b.textContent = "← Survey로";
    b.onclick = backToSurvey;
    b.classList.remove("hidden");
  } else {                              // non-digital -> inline info, stay on the overview
    showSurveyInfo(e);
  }
}
function backToSurvey() { state.drilled = false; renderSurvey(state.survey); }

function showSurveyInfo(e) {
  const d = e.det;
  let title, rows;
  if (e.kind === "chirp" && e.info) {          // linear chirp / CSS (LoRa): characterized only
    const c = e.info;
    title = (c.sf ? `LoRa 계열 (SF${c.sf})` : "선형 처프 / CSS") + " — 복조 대상 아님";
    // rate + direction are robust (coherent from the beat tone); bw/SF only when the
    // channel is cleanly isolated enough to snap to a LoRa hypothesis.
    rows = [["중심주파수", fmtHz(e.abs_fc)],
            ["처프 방향", c.up ? "▲ 상승" : "▼ 하강"],
            ["처프율", (c.mu / 1e9).toFixed(3) + " MHz/ms"]];
    if (c.sf) rows.push(["대역폭", fmtHz(c.bw)], ["심볼레이트", fmtHz(c.rs)],
                        ["심볼시간", (c.tsym * 1e3).toFixed(2) + " ms"]);
  } else {
    const KIND = { analog: "아날로그 (FM/음성)", tone: "톤 / CW",
      tooshort: "너무 짧은 버스트", error: "분석 오류" };
    title = (KIND[e.kind] || e.kind) + " — 복조 대상 아님";
    rows = [["중심주파수", fmtHz(e.abs_fc)], ["대역폭", fmtHz(d.bw)],
      ["SNR", d.snr_db == null ? "–" : d.snr_db.toFixed(1) + " dB"],
      ["심볼레이트(추정)", fmtHz(d.baud_hint)]];
  }
  $("survInfo").innerHTML = `<h3>${title}</h3>` +
    '<div class="si-grid">' + rows.map(([k, v]) =>
      `<div><div class="si-k">${k}</div><div class="si-v">${v}</div></div>`).join("") + "</div>";
  $("survInfo").classList.remove("hidden");
  $("survInfo").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "nearest" });
}

/* --- burst map: whole-record waterfall with clickable time-burst boxes (single signal) --- */
function renderBurstMap(ov, result) {
  show($("burstMap"));
  paintWaterfall($("burstFall"), ov.waterfall, 150);
  const host = $("burstBoxes"), n = ov.n || 1;
  host.innerHTML = "";
  (result.bursts || []).forEach((b, i) => {
    const top = Math.max(0, b.start / n * 100);
    const dur = (b.end - b.start) / ov.fs;
    const lbl = `버스트 ${i + 1} · ${dur >= 1 ? dur.toFixed(2) + " s" : (dur * 1e3).toFixed(0) + " ms"}`;
    const box = document.createElement("button");   // full width (one signal), spans t0..t1 in time
    box.className = "box" + (i === result.burst_idx ? " sel" : "");
    box.style.left = "0%"; box.style.width = "100%"; box.style.top = top + "%";
    box.style.height = Math.min(100 - top, Math.max(5, (b.end - b.start) / n * 100)) + "%";
    box.title = lbl;
    box.innerHTML = `<span class="box-lbl">${lbl}</span>`;
    box.onclick = () => { state.burst = i; analyze(); };   // re-analyse that burst; map persists
    host.appendChild(box);
  });
}

/* --- rendering --- */
function statusOf(lock) {
  if (lock >= 60) return ["복조 성공", "ok"];
  if (lock >= 40) return ["부분 복조", "warn"];
  return ["복조 실패", "bad"];
}
function render(d) {
  hideAll();
  show($("results"));
  $("burstMap").classList.add("hidden");   // re-shown by renderBurstMap only in multi-burst single mode
  const lock = d.quality.lock, [txt, cls] = statusOf(lock);
  const pill = $("statusPill");
  pill.textContent = txt;
  pill.className = "pill " + cls;
  $("fileName").textContent =
    `${state.file.name} · ${fmtHz(d.fs)} · ${d.fmt === "real" ? "Real PCM" : "IQ"}`;
  $("merVal").textContent = d.quality.mer_db == null ? "–" : d.quality.mer_db.toFixed(1) + " dB";
  $("evmVal").textContent = d.quality.evm == null ? "–" : (d.quality.evm * 100).toFixed(1) + " %";
  $("snrVal").textContent = snrText(d);

  const flags = [];
  if (d.family === "fsk") flags.push('<span class="chip warn">FSK 계열 · 주파수 판별</span>');
  if (d.eq && d.eq.applied) flags.push(`<span class="chip hit">등화기 적용 · ${
    d.eq.mode === "fse" ? "T/2 분수간격" : "심볼간격"}</span>`);
  if (d.detected.alias_resolved) flags.push('<span class="chip warn">반송파 앨리어스 보정</span>');
  if (d.detected.baud_fallback) flags.push('<span class="chip warn">심볼레이트 대역폭 폴백</span>');
  if (d.detected.carrier_ambiguous) flags.push('<span class="chip warn">반송파 모호 · 앨리어싱 가능</span>');
  $("flagRow").innerHTML = flags.join("");

  renderBursts(d);
  animateGauge(lock, cls);
  renderParams(d);
  drawSpectrum(d);
  drawWaterfall(d);
  startPlay(d);
}
function snrText(d) {
  const s = d.snr_est_db;
  if (s == null || d.quality.lock < 30) return "–";
  return (s > 28 ? "≥28" : s.toFixed(1)) + " dB";
}

function renderBursts(d) {
  const row = $("burstRow"), bs = d.bursts || [];
  // a survey-drilled emitter is already a cut channel: burst re-selection would
  // re-post the whole capture, so it is disabled while drilled. When the burst MAP
  // is shown (multi-burst single signal) the map replaces the chips.
  if (state.drilled || state.overview || bs.length < 2) { row.innerHTML = ""; return; }
  const dur = (b) => {
    const sec = (b.end - b.start) / d.fs;
    return sec >= 1 ? sec.toFixed(2) + " s" : (sec * 1e3).toFixed(1) + " ms";
  };
  row.innerHTML = bs.map((b, i) =>
    `<button class="chip-btn${i === d.burst_idx ? " on" : ""}" data-i="${i}">` +
    `버스트 ${i + 1} · ${dur(b)}</button>`).join("");
  row.querySelectorAll("button").forEach((el) => {
    el.onclick = () => { state.burst = +el.dataset.i; analyze(); };
  });
}

function chip(hit, truthTxt) {
  return `<span class="chip ${hit ? "hit" : "miss"}">정답 ${truthTxt} · ${hit ? "일치" : "불일치"}</span>`;
}
function renderParams(d) {
  const det = d.detected, t = state.sidecar && state.sidecar.truth;
  const near = (a, b, tol) => Math.abs(a - b) <= tol;
  const rows = [
    ["중심주파수", det.rf_hz != null ? fmtRf(det.rf_hz) : fmtHz(det.fc),
      t && t.fc != null && chip(near(det.fc, t.fc, Math.max(300, 0.02 * t.baud)), fmtHz(t.fc)),
      det.rf_hz != null ? `기저대역 ${fmtHz(det.fc)}` : ""],
    ["심볼레이트", fmtHz(det.baud), t && t.baud != null && chip(near(det.baud, t.baud, 0.03 * t.baud), fmtHz(t.baud)),
      det.baud_conf ? `스펙트럴 라인 강도 ×${Math.round(det.baud_conf)}` : ""],
    ["변조방식", det.mod.toUpperCase(), t && t.mod != null && chip(det.mod === t.mod, t.mod.toUpperCase()),
      det.symmetry ? `대칭 ${det.symmetry}` : "주파수 레벨"],
  ];
  if (d.family === "fsk") {
    rows.push(["변조지수 h", det.h.toFixed(2),
      t && t.h != null && chip(near(det.h, t.h, 0.15), t.h.toFixed(2)), ""]);
  } else {
    rows.push(["롤오프", det.rolloff.toFixed(2),
      t && t.rolloff != null && chip(near(det.rolloff, t.rolloff, 0.11), t.rolloff.toFixed(2)), ""]);
  }
  $("paramGrid").innerHTML = rows.map(([label, val, hit, note]) => `
    <div class="card param">
      <div class="param-label">${label}</div>
      <div class="param-val">${val}</div>
      ${note ? `<div class="param-note">${note}</div>` : ""}
      ${hit ? `<div class="chips">${hit}</div>` : ""}
    </div>`).join("");
}

function animateGauge(lock, cls) {
  const arc = $("donutArc"), num = $("lockNum"), C = 2 * Math.PI * 52;
  arc.style.stroke = { ok: "#12b886", warn: "#f59f00", bad: "#fa5252" }[cls];
  arc.style.strokeDasharray = C;
  const target = C * (1 - Math.max(0, Math.min(100, lock)) / 100);
  if (REDUCED) { arc.style.strokeDashoffset = target; num.textContent = Math.round(lock); return; }
  arc.style.strokeDashoffset = C;
  requestAnimationFrame(() => { arc.style.strokeDashoffset = target; });
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 900);
    num.textContent = Math.round(lock * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* --- canvas helpers --- */
function fit(cv, hCss) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 320;
  const h = hCss || w;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}
function pct(a, p) {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))];
}

/* --- spectrum --- */
function drawSpectrum(d) {
  const [g, w, h] = fit($("specCanvas"), 120);
  g.fillStyle = "#0d1320"; g.fillRect(0, 0, w, h);
  if (!d.spectrum) return;
  const f = d.spectrum.f, db = d.spectrum.db;
  const lo = pct(db, 0.02), hi = pct(db, 1) + 2;
  const fmin = f[0], fmax = f[f.length - 1];
  const X = (v) => (v - fmin) / (fmax - fmin) * w;
  const Y = (v) => h - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (h - 8) - 4;
  const det = d.detected, half = det.baud / 2e3;
  g.strokeStyle = "rgba(120,180,255,.35)"; g.setLineDash([4, 4]);
  for (const gf of [det.fc / 1e3 - half, det.fc / 1e3 + half]) {
    g.beginPath(); g.moveTo(X(gf), 0); g.lineTo(X(gf), h); g.stroke();
  }
  g.setLineDash([]);
  g.strokeStyle = "rgba(255,255,255,.45)";
  g.beginPath(); g.moveTo(X(det.fc / 1e3), 0); g.lineTo(X(det.fc / 1e3), h); g.stroke();
  g.strokeStyle = "#5aa9ff"; g.lineWidth = 1.4; g.beginPath();
  f.forEach((v, k) => (k ? g.lineTo(X(v), Y(db[k])) : g.moveTo(X(v), Y(db[k]))));
  g.stroke();
  g.fillStyle = "rgba(255,255,255,.5)"; g.font = "10px sans-serif";
  g.fillText(sig(fmin) + " kHz", 4, h - 4);
  const tmax = sig(fmax) + " kHz";
  g.fillText(tmax, w - g.measureText(tmax).width - 4, h - 4);
}

/* --- waterfall --- */
function drawWaterfall(d) { paintWaterfall($("fallCanvas"), d.waterfall, 150); }
function paintWaterfall(canvas, wf, hCss) {
  const [g, w, h] = fit(canvas, hCss);
  g.fillStyle = "#0d1320"; g.fillRect(0, 0, w, h);
  if (!wf || !wf.rows) return;
  const lo = pct(wf.db, 0.05), hi = pct(wf.db, 0.95);
  const img = g.createImageData(wf.bins, wf.rows);
  for (let k = 0; k < wf.db.length; k++) {
    const t = Math.max(0, Math.min(1, (wf.db[k] - lo) / (hi - lo + 1e-9)));
    const o = k * 4;                             // dark -> blue -> white ramp
    img.data[o] = t < 0.5 ? 13 + t * 120 : 73 + (t - 0.5) * 364;
    img.data[o + 1] = t < 0.5 ? 19 + t * 260 : 149 + (t - 0.5) * 212;
    img.data[o + 2] = t < 0.5 ? 32 + t * 446 : 255;
    img.data[o + 3] = 255;
  }
  createImageBitmap(img).then((bmp) => { g.imageSmoothingEnabled = true; g.drawImage(bmp, 0, 0, w, h); });
}

/* --- constellation playback (the headline) --- */
const play = { raf: 0, i: 0, on: false, data: null };
let CG = null;   // cached constellation geometry

function stopPlay() { cancelAnimationFrame(play.raf); play.raf = 0; play.on = false; }
function setPlayBtn(on) { $("playBtn").textContent = on ? "⏸ 일시정지" : "▶ 재생"; }

function startPlay(d) {
  stopPlay();
  play.data = d;
  play.i = 0;
  const n = d.constellation.i.length;
  $("scrub").max = String(Math.max(1, n));
  $("playInfo").textContent = `${n.toLocaleString()} 심볼 · 시간 순서로 재생 (잔광 효과)`;
  drawConstBase(d);
  if (REDUCED) {                       // static full view, no motion
    drawPoints(d, 0, n);
    $("scrub").value = String(n);
    setPlayBtn(false);
    return;
  }
  play.on = true;
  setPlayBtn(true);
  play.raf = requestAnimationFrame(loop);
}
$("playBtn").onclick = () => {
  if (!play.data) return;
  if (play.on) { stopPlay(); setPlayBtn(false); }
  else { play.on = true; setPlayBtn(true); play.raf = requestAnimationFrame(loop); }
};
$("scrub").oninput = () => {
  if (!play.data) return;
  stopPlay(); setPlayBtn(false);
  play.i = +$("scrub").value;
  drawConstBase(play.data);
  drawPoints(play.data, 0, play.i);    // redraw history up to the scrub point
};

function loop() {
  const d = play.data, n = d.constellation.i.length;
  const step = Math.max(1, Math.round(n / 600) * (+$("speedSel").value));
  const to = Math.min(n, play.i + step);
  fade();                              // phosphor persistence
  drawPoints(d, play.i, to);
  play.i = to >= n ? 0 : to;
  if (play.i === 0) drawConstBase(d);  // loop: clear the trail
  $("scrub").value = String(play.i);
  if (play.on) play.raf = requestAnimationFrame(loop);
}

function drawConstBase(d) {
  const [g, w] = fit($("constCanvas"));
  const fsk = d.family === "fsk";
  const I = d.constellation.i, Q = d.constellation.q;
  const ideal = fsk ? [] : idealPoints(d.detected.mod);
  let lim = 0;
  for (let k = 0; k < I.length; k++) lim = Math.max(lim, Math.abs(I[k]), Math.abs(Q[k]));
  for (const p of ideal) lim = Math.max(lim, Math.abs(p.i), Math.abs(p.q));
  lim = (lim || 1) * 1.12;
  const R = w / 2 * 0.9;
  CG = { g, w, lim, R, map: (re, im) => [w / 2 + re / lim * R, w / 2 - im / lim * R] };

  g.fillStyle = "#0d1320"; g.fillRect(0, 0, w, w);
  g.strokeStyle = "rgba(255,255,255,.06)"; g.lineWidth = 1;
  for (let f = -1; f <= 1; f += 0.5) {
    const [gx] = CG.map(f, 0), [, gy] = CG.map(0, f);
    g.beginPath(); g.moveTo(gx, 0); g.lineTo(gx, w); g.stroke();
    g.beginPath(); g.moveTo(0, gy); g.lineTo(w, gy); g.stroke();
  }
  if (fsk) {                            // frequency levels as dashed bands
    g.strokeStyle = "rgba(255,255,255,.5)"; g.setLineDash([6, 5]);
    for (const lv of [...new Set(I.map((v) => Math.round(v)))].filter((v) => Math.abs(v) <= 8)) {
      const [, yy] = CG.map(0, lv);
      g.beginPath(); g.moveTo(0, yy); g.lineTo(w, yy); g.stroke();
    }
    g.setLineDash([]);
  } else {
    g.strokeStyle = "rgba(255,255,255,.75)"; g.lineWidth = 1.4;
    for (const p of ideal) {
      const [x, y] = CG.map(p.i, p.q);
      g.beginPath(); g.arc(x, y, 6, 0, 7); g.stroke();
    }
  }
}
function fade() {
  if (!CG) return;
  CG.g.fillStyle = "rgba(13,19,32,.14)";   // translucent wash = phosphor decay
  CG.g.fillRect(0, 0, CG.w, CG.w);
}
function drawPoints(d, from, to) {
  if (!CG || to <= from) return;
  const g = CG.g, I = d.constellation.i, Q = d.constellation.q;
  const fsk = d.family === "fsk";
  const a = Math.max(0.08, Math.min(0.55, 150 / (to - from)));
  g.globalCompositeOperation = "lighter";
  g.fillStyle = `rgba(90,170,255,${a})`;
  for (let k = from; k < to; k++) {
    // FSK has no I/Q plane: spread levels vertically, jitter horizontally for density
    const [x, y] = fsk ? CG.map((k % 89) / 89 - 0.5, I[k]) : CG.map(I[k], Q[k]);
    g.beginPath(); g.arc(x, y, 1.8, 0, 7); g.fill();
  }
  g.globalCompositeOperation = "source-over";
}

/* --- downloads --- */
function saveBlob(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}
const stem = () => state.file.name.replace(/\.[^.]*$/, "");
$("copyBits").onclick = async (e) => {   // decoded bitstream -> clipboard, with brief feedback
  const btn = e.currentTarget, was = btn.textContent;
  try { await navigator.clipboard.writeText(state.resp.bits); btn.textContent = "복사됨 ✓"; }
  catch { btn.textContent = "복사 실패"; }
  setTimeout(() => { btn.textContent = was; }, 1400);
};
$("dlBits").onclick = () => saveBlob(stem() + ".bits.txt", state.resp.bits, "text/plain");
$("dlSyms").onclick = () => {
  const c = state.resp.constellation;
  saveBlob(stem() + ".symbols.csv", "i,q\n" + c.i.map((v, k) => `${v},${c.q[k]}`).join("\n"), "text/csv");
};
$("dlReport").onclick = () =>
  saveBlob(stem() + ".report.json", JSON.stringify(state.resp, null, 2), "application/json");

/* --- view helpers --- */
function show(el) { el.classList.remove("hidden"); }
function hideAll() {
  stopPlay();
  ["metaForm", "loading", "errorCard", "results", "batchCard", "surveyCard"]
    .forEach((id) => $(id).classList.add("hidden"));
  $("backBatch").classList.add("hidden");
}
function showError(msg) { hideAll(); $("errMsg").textContent = msg; show($("errorCard")); }

/* --- dropzone wiring --- */
const drop = $("dropCard"), input = $("fileInput");
// reset value first: re-picking the SAME file fires no change event otherwise
const pick = () => { input.value = ""; input.click(); };
drop.onclick = pick;
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } };
input.onchange = () => { if (input.files.length) acceptFiles(input.files); };
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("drag");
  if (e.dataTransfer.files.length) acceptFiles(e.dataTransfer.files);
});
