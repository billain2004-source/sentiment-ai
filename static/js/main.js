'use strict';

// ── State ──────────────────────────────────────────────────────
let analysisCount = 0;
let history = [];

const EXAMPLES = [
  "She played cricket brilliantly in the final match.",
  "Apple Inc. launched an incredible new product today that everyone loved.",
  "He is deeply frustrated with the constant delays and poor service.",
  "The temperature today is about 22 degrees Celsius.",
  "They won the championship after years of hard work and dedication.",
  "The factory polluted the river, destroying the local ecosystem.",
  "The library opens at nine and closes at six on weekdays.",
  "Python is a versatile programming language used across many domains.",
  "She overcame every obstacle with remarkable courage and resilience.",
  "The match ended in a dull draw with no goals scored.",
];

// ── DOM refs ───────────────────────────────────────────────────
const txt         = document.getElementById('txt');
const charCount   = document.getElementById('char-count');
const btnAnalyze  = document.getElementById('btn-analyze');
const btnClear    = document.getElementById('btn-clear');
const btnExample  = document.getElementById('btn-example');
const btnClearH   = document.getElementById('btn-clear-hist');
const histBody    = document.getElementById('history-body');
const loading     = document.getElementById('loading');
const statCount   = document.getElementById('stat-count');

// gauge refs
const gaugeIdle   = document.getElementById('gauge-idle');
const gaugeResult = document.getElementById('gauge-result');
const needle      = document.getElementById('needle');
const gaugeEmoji  = document.getElementById('gauge-emoji');
const gaugeLbl    = document.getElementById('gauge-label');
const gaugeConf   = document.getElementById('gauge-conf');
const intensChip  = document.getElementById('intensity-chip');
const compVal     = document.getElementById('compound-val');

const barPos  = document.getElementById('bar-pos');
const barNeu  = document.getElementById('bar-neu');
const barNeg  = document.getElementById('bar-neg');
const pctPos  = document.getElementById('pct-pos');
const pctNeu  = document.getElementById('pct-neu');
const pctNeg  = document.getElementById('pct-neg');

// context refs
const ctxIdle  = document.getElementById('ctx-idle');
const ctxGrid  = document.getElementById('ctx-grid');
const ctxCount = document.getElementById('ctx-count');

// token refs
const tokensCard = document.getElementById('tokens-card');
const tokensWrap = document.getElementById('tokens-wrap');

// ── Event listeners ────────────────────────────────────────────
txt.addEventListener('input', () => {
  if (txt.value.length > 512) txt.value = txt.value.slice(0, 512);
  charCount.textContent = `${txt.value.length} / 512`;
});

btnAnalyze.addEventListener('click', runAnalysis);
btnClear.addEventListener('click', clearAll);
btnExample.addEventListener('click', loadExample);
btnClearH.addEventListener('click', clearHistory);
txt.addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') runAnalysis(); });

// ── Core: run analysis ────────────────────────────────────────
async function runAnalysis() {
  const text = txt.value.trim();
  if (!text) { txt.focus(); return; }

  setLoading(true);
  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    renderGauge(data.sentiment);
    renderTokens(data.sentiment.word_scores);
    renderContext(data.context);
    pushHistory(text, data.sentiment);

    analysisCount++;
    statCount.textContent = analysisCount;
  } catch (err) {
    gaugeIdle.innerHTML = `<i class="fas fa-circle-exclamation" style="font-size:2rem;color:#ef4444"></i>
      <p style="color:#ef4444">${esc(err.message || 'Analysis failed. Is the server running?')}</p>`;
    gaugeIdle.style.display = 'flex';
    gaugeResult.style.display = 'none';
  } finally {
    setLoading(false);
  }
}

// ── Gauge renderer ────────────────────────────────────────────
function renderGauge(s) {
  gaugeIdle.style.display   = 'none';
  gaugeResult.style.display = 'block';

  const emojis    = { positive: '😊', negative: '😞', neutral: '😐' };
  const colors    = { positive: '#10b981', negative: '#ef4444', neutral: '#f59e0b' };
  const intensBg  = { positive: 'rgba(16,185,129,.15)', negative: 'rgba(239,68,68,.15)', neutral: 'rgba(245,158,11,.15)' };
  const label     = s.label;

  // Needle: compound -1→180°, 0→90°, +1→0°
  const compound   = s.compound;
  const angleDeg   = 90 - compound * 90;           // 0→90, -1→180, +1→0
  const angleRad   = angleDeg * Math.PI / 180;
  const R          = 70;                            // needle length
  const cx = 110, cy = 115;
  const nx = cx + R * Math.cos(Math.PI - angleRad);
  const ny = cy - R * Math.sin(Math.PI - angleRad);
  needle.setAttribute('x2', nx.toFixed(1));
  needle.setAttribute('y2', ny.toFixed(1));

  gaugeEmoji.textContent = emojis[label] || '🤔';
  gaugeLbl.textContent   = label;
  gaugeLbl.className     = `gauge-label ${label}`;
  gaugeConf.textContent  = `${Math.round(s.confidence * 100)}% BERT confidence`;

  // intensity chip
  intensChip.textContent       = s.intensity;
  intensChip.style.display     = 'inline';
  intensChip.style.background  = intensBg[label] || 'rgba(124,58,237,.15)';
  intensChip.style.color       = colors[label] || '#a78bfa';
  intensChip.style.border      = `1px solid ${colors[label]}44`;

  // bars — animate after a short delay so they're visible
  setTimeout(() => {
    const bd = s.breakdown;
    barPos.style.width = bd.positive + '%';
    barNeu.style.width = bd.neutral  + '%';
    barNeg.style.width = bd.negative + '%';
    pctPos.textContent = bd.positive + '%';
    pctNeu.textContent = bd.neutral  + '%';
    pctNeg.textContent = bd.negative + '%';
  }, 80);

  compVal.textContent = compound >= 0 ? `+${compound.toFixed(3)}` : compound.toFixed(3);
  compVal.style.color = colors[label] || 'var(--text)';
}

// ── Word token renderer ───────────────────────────────────────
function renderTokens(wordScores) {
  if (!wordScores || wordScores.length === 0) {
    tokensCard.style.display = 'none';
    return;
  }
  tokensCard.style.display = 'block';
  tokensWrap.innerHTML = wordScores.map(w =>
    `<span class="token ${w.sentiment}" title="Score: ${w.score}">${esc(w.word)}</span>`
  ).join(' ');
}

// ── Context renderer ──────────────────────────────────────────
function renderContext(contexts) {
  if (!contexts || contexts.length === 0) {
    ctxIdle.style.display = 'flex';
    ctxGrid.style.display = 'none';
    ctxCount.style.display = 'none';
    return;
  }

  ctxIdle.style.display = 'none';
  ctxGrid.style.display = 'flex';
  ctxCount.style.display = 'inline';
  ctxCount.textContent = `${contexts.length} found`;

  ctxGrid.innerHTML = contexts.map((c, i) => `
    <div class="ctx-card ${esc(c.css_class || 'default')}" style="animation-delay:${i * 55}ms">
      <div class="ctx-word">${esc(c.word)}</div>
      <div class="ctx-type">${esc(c.type || '')}</div>
      <div class="ctx-desc">${esc(c.description || '')}</div>
    </div>
  `).join('');
}

// ── History ───────────────────────────────────────────────────
function pushHistory(text, sentiment) {
  history.unshift({ text, sentiment, time: new Date().toLocaleTimeString() });
  if (history.length > 30) history.pop();
  renderHistory();
}

function renderHistory() {
  if (history.length === 0) {
    histBody.innerHTML = '<p class="empty-hint">No analyses yet.</p>';
    return;
  }
  histBody.innerHTML = history.map((item, i) => `
    <div class="hist-item" onclick="rerun(${i})">
      <div class="hist-text">${esc(item.text)}</div>
      <div class="hist-meta">
        <span class="sbadge ${item.sentiment.label}">${item.sentiment.label}</span>
        <span class="hist-time">${item.time}</span>
      </div>
    </div>
  `).join('');
}

window.rerun = function(index) {
  txt.value = history[index].text;
  charCount.textContent = `${txt.value.length} / 512`;
  runAnalysis();
};

// ── Helpers ───────────────────────────────────────────────────
function clearAll() {
  txt.value = '';
  charCount.textContent = '0 / 512';
  gaugeIdle.innerHTML = `
    <i class="fas fa-brain gauge-idle-icon"></i>
    <p>Enter text and press <strong>Analyze</strong></p>
    <span><kbd>Ctrl</kbd> + <kbd>Enter</kbd></span>`;
  gaugeIdle.style.display = 'flex';
  gaugeResult.style.display = 'none';
  ctxIdle.style.display = 'flex';
  ctxGrid.style.display = 'none';
  ctxCount.style.display = 'none';
  tokensCard.style.display = 'none';
  intensChip.style.display = 'none';
  txt.focus();
}

function clearHistory() {
  history = [];
  renderHistory();
}

function loadExample() {
  const ex = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
  txt.value = ex;
  charCount.textContent = `${ex.length} / 512`;
  txt.focus();
}

function setLoading(show) {
  loading.classList.toggle('active', show);
  loading.setAttribute('aria-hidden', String(!show));
  btnAnalyze.disabled = show;
}

function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}
