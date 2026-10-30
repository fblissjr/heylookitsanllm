// Perf: read-only admin view onto /v1/system/metrics and
// /v1/performance/profile/<range>. Deliberately inert -- no polling, no
// websockets, no auto-refresh. Data only moves on mount or an explicit
// click (Refresh button, time-range button).
//
// Invariants:
// - System section containers are built once in setup(); refresh updates
//   text/bar values in place via element refs kept in ctx.state. Per-model
//   rows are diffed by model id (add/remove only when the loaded-model set
//   actually changes) rather than torn down wholesale.
// - The two profile tables (timing breakdown, trends) are the one place a
//   full replaceChildren() is expected -- row counts change with the range.

import { createPage } from '../page.js';
import { causeWords, createEl, formatBytes, formatTokens } from '../utils.js';
import { api } from '../api.js';
import * as drawer from '../settings-drawer.js';

const RANGES = ['1h', '6h', '24h', '7d'];
const DEFAULT_RANGE = '1h';
const OBS_LEVELS = ['off', 'minimal', 'standard', 'debug'];
const OBS_STREAMS = ['events', 'metrics'];

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.modelRows = new Map();
    s.activeRange = DEFAULT_RANGE;
    s.activeStream = 'events';

    buildSkeleton(ctx);
    // No sampler/page settings here -- register so the drawer still offers the
    // global Display prefs and hides the sampler panel.
    ctx.onTeardown(drawer.registerSettings({ samplers: 'hidden' }));

    s.refreshBtn.addEventListener('click', () => {
      loadMetrics(ctx, true);
      loadObservability(ctx);
    });
    for (const btn of s.rangeButtons) {
      btn.addEventListener('click', () => {
        if (btn.dataset.range === s.activeRange) return;
        loadProfile(ctx, btn.dataset.range);
      });
    }
    for (const btn of s.streamButtons) {
      btn.addEventListener('click', () => {
        if (btn.dataset.stream === s.activeStream) return;
        loadLogs(ctx, btn.dataset.stream);
      });
    }
    s.obsLevelSelect.addEventListener('change', () => updateObsSetting(ctx, {
      observability_level: s.obsLevelSelect.value,
    }));
    s.obsRetentionInput.addEventListener('change', () => {
      const days = Number.parseInt(s.obsRetentionInput.value, 10);
      if (Number.isNaN(days) || days < 0) return;
      updateObsSetting(ctx, { observability_retention_days: days });
    });

    await Promise.all([
      loadMetrics(ctx, false),
      loadProfile(ctx, DEFAULT_RANGE),
      loadObservability(ctx),
    ]);
  },
});

// ---------------------------------------------------------------------------
// skeleton
// ---------------------------------------------------------------------------

function buildSkeleton(ctx) {
  const s = ctx.state;

  s.errorEl = createEl('div', { class: 'error-note', role: 'alert', hidden: true });

  s.updatedEl = createEl('span', { class: 'perf__updated small muted' }, ['']);
  s.refreshBtn = createEl('button', { class: 'btn btn--sm' }, ['Refresh']);

  const header = createEl('header', { class: 'perf__header' }, [
    createEl('h1', {}, ['Performance']),
    createEl('div', { class: 'perf__header-actions' }, [s.updatedEl, s.refreshBtn]),
  ]);

  // system --------------------------------------------------------------
  s.ramUsedEl = createEl('span', { class: 'perf-row__value' }, ['--']);
  s.ramAvailEl = createEl('span', { class: 'perf-row__value' }, ['--']);
  s.ramTotalEl = createEl('span', { class: 'perf-row__value' }, ['--']);
  s.cpuEl = createEl('span', { class: 'perf-row__value' }, ['--']);

  const sysRow = (label, valueEl) =>
    createEl('div', { class: 'perf-row' }, [createEl('span', { class: 'muted' }, [label]), valueEl]);

  s.modelsEl = createEl('div', { class: 'perf__models' });

  const systemSection = createEl('section', { class: 'perf__section' }, [
    createEl('h2', { class: 'perf__section-title' }, ['System']),
    createEl('div', { class: 'perf__rows' }, [
      sysRow('RAM used', s.ramUsedEl),
      sysRow('RAM available', s.ramAvailEl),
      sysRow('RAM total', s.ramTotalEl),
      sysRow('CPU', s.cpuEl),
    ]),
    s.modelsEl,
  ]);

  // profile ---------------------------------------------------------------
  s.rangeButtons = RANGES.map((range) => createEl('button', {
    class: `btn btn--sm perf__range-btn${range === DEFAULT_RANGE ? ' perf__range-btn--active' : ''}`,
    dataset: { range },
    'aria-pressed': range === DEFAULT_RANGE ? 'true' : 'false',
  }, [range]));

  s.profileBody = createEl('div', { class: 'perf__profile-body' });

  const profileSection = createEl('section', { class: 'perf__section' }, [
    createEl('div', { class: 'perf__section-head' }, [
      createEl('h2', { class: 'perf__section-title' }, ['Profile']),
      createEl('div', { class: 'perf__range-buttons' }, s.rangeButtons),
    ]),
    s.profileBody,
  ]);

  // observability ---------------------------------------------------------
  s.obsLevelSelect = createEl('select', {
    class: 'input perf__obs-select',
    'aria-label': 'Observability level',
  }, OBS_LEVELS.map((lvl) => createEl('option', { value: lvl }, [lvl])));

  s.obsRetentionInput = createEl('input', {
    class: 'input perf__obs-retention',
    type: 'number',
    min: '0',
    step: '1',
    value: '30',
    'aria-label': 'Observability retention days',
  });

  s.obsStatusEl = createEl('span', { class: 'small muted perf__obs-status', role: 'status' }, ['']);

  s.streamButtons = OBS_STREAMS.map((stream) => createEl('button', {
    class: `btn btn--sm perf__range-btn${stream === 'events' ? ' perf__range-btn--active' : ''}`,
    dataset: { stream },
    'aria-pressed': stream === 'events' ? 'true' : 'false',
  }, [stream]));

  s.obsLogsBody = createEl('div', { class: 'perf__obs-logs' });

  const obsSection = createEl('section', { class: 'perf__section perf__obs' }, [
    createEl('div', { class: 'perf__section-head' }, [
      createEl('h2', { class: 'perf__section-title' }, ['Observability']),
      createEl('div', { class: 'perf__range-buttons' }, s.streamButtons),
    ]),
    createEl('div', { class: 'perf__obs-controls' }, [
      createEl('label', { class: 'perf__obs-field' }, [
        createEl('span', { class: 'muted small' }, ['Level']),
        s.obsLevelSelect,
      ]),
      createEl('label', { class: 'perf__obs-field' }, [
        createEl('span', { class: 'muted small' }, ['Retention (days)']),
        s.obsRetentionInput,
      ]),
      s.obsStatusEl,
    ]),
    s.obsLogsBody,
  ]);

  ctx.el.append(createEl('div', { class: 'perf' }, [
    s.errorEl,
    header,
    systemSection,
    profileSection,
    obsSection,
  ]));
}

function setError(ctx, text) {
  ctx.state.errorEl.textContent = text;
  ctx.state.errorEl.hidden = !text;
}

// ---------------------------------------------------------------------------
// system metrics
// ---------------------------------------------------------------------------

async function loadMetrics(ctx, force) {
  try {
    const data = await api.systemMetrics(force, { signal: ctx.signal });
    if (!ctx.alive) return;
    setError(ctx, '');
    applyMetrics(ctx, data);
  } catch (err) {
    if (ctx.alive) setError(ctx, `Could not load metrics: ${err.message}`);
  }
}

function applyMetrics(ctx, data) {
  const s = ctx.state;
  const sys = data.system ?? {};
  s.ramUsedEl.textContent = fmtGb(sys.ram_used_gb);
  s.ramAvailEl.textContent = fmtGb(sys.ram_available_gb);
  s.ramTotalEl.textContent = fmtGb(sys.ram_total_gb);
  s.cpuEl.textContent = sys.cpu_percent != null ? `${sys.cpu_percent.toFixed(1)}%` : '--';
  s.updatedEl.textContent = data.timestamp ? `Updated ${formatClock(data.timestamp)}` : '';
  updateModels(ctx, data.models ?? {});
}

function updateModels(ctx, models) {
  const s = ctx.state;
  const ids = Object.keys(models);

  // drop rows for models that are no longer loaded
  for (const [id, row] of s.modelRows) {
    if (!(id in models)) {
      row.root.remove();
      s.modelRows.delete(id);
    }
  }

  if (!ids.length) {
    s.modelsEl.replaceChildren(createEl('div', { class: 'empty-state' }, ['No models loaded right now.']));
    return;
  }

  if (s.modelsEl.querySelector('.empty-state')) s.modelsEl.replaceChildren();

  for (const id of ids) {
    let row = s.modelRows.get(id);
    if (!row) {
      row = buildModelRow(id);
      s.modelRows.set(id, row);
      s.modelsEl.append(row.root);
    }
    applyModelData(row, models[id]);
  }
}

// Not models.js's `buildModelRow`: that one builds a management row (load,
// config, residency) from a model object; this builds a metrics row from an id.
function buildModelRow(id) {
  const memEl = createEl('span', { class: 'perf-model__mem small muted' }, ['--']);
  const fillEl = createEl('div', { class: 'perf-bar__fill' });
  const ctxEl = createEl('span', {}, ['--']);
  const reqsEl = createEl('span', {}, ['--']);

  const root = createEl('div', { class: 'perf-model' }, [
    createEl('div', { class: 'perf-model__head' }, [
      createEl('span', { class: 'perf-model__id' }, [id]),
      memEl,
    ]),
    createEl('div', { class: 'perf-model__bar-row' }, [
      createEl('div', { class: 'perf-bar' }, [fillEl]),
    ]),
    createEl('div', { class: 'perf-model__meta small muted' }, [ctxEl, reqsEl]),
  ]);

  return { root, memEl, fillEl, ctxEl, reqsEl };
}

// Null means unknown (the server says so rather than sending 0): memory it
// could not read, or a context no request has used yet. Shown as "--", never
// as a measured zero. memory_source differs by engine (gguf: its own
// llama-server process; MLX: the whole server's Metal memory), so it rides
// the title.
function applyModelData(row, m) {
  row.memEl.textContent = m.memory_mb != null ? formatBytes(m.memory_mb * 1024 * 1024) : 'memory --';
  row.memEl.title = m.memory_source ?? '';
  const pct = m.context_percent != null ? Math.max(0, Math.min(100, m.context_percent)) : null;
  row.fillEl.style.width = `${pct ?? 0}%`;
  row.ctxEl.textContent = `ctx ${fmtInt(m.context_used)} / ${fmtInt(m.context_capacity)}`
    + (pct != null ? ` (${pct.toFixed(1)}%)` : '');
  row.reqsEl.textContent = `${fmtInt(m.requests_active)} active · ${fmtInt(m.requests_queued)} queued`;
}

// ---------------------------------------------------------------------------
// profile
// ---------------------------------------------------------------------------

async function loadProfile(ctx, range) {
  const s = ctx.state;
  s.activeRange = range;
  for (const btn of s.rangeButtons) {
    const on = btn.dataset.range === range;
    btn.classList.toggle('perf__range-btn--active', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
  s.profileBody.replaceChildren(createEl('div', { class: 'empty-state' }, ['Loading…']));

  try {
    const data = await api.perfProfile(range, { signal: ctx.signal });
    if (!ctx.alive || s.activeRange !== range) return;
    renderProfile(ctx, data);
  } catch (err) {
    if (!ctx.alive || s.activeRange !== range) return;
    renderProfileEmpty(ctx);
  }
}

function renderProfile(ctx, data) {
  const breakdown = data?.timing_breakdown ?? [];
  const trends = data?.trends ?? [];
  const cache = data?.cache ?? [];
  if (!breakdown.length && !trends.length && !cache.length) {
    renderProfileEmpty(ctx);
    return;
  }

  const children = [];
  if (breakdown.length) {
    children.push(createEl('h3', {}, ['Timing breakdown']));
    children.push(buildTimingTable(breakdown));
  }
  if (cache.length) {
    children.push(createEl('h3', {}, ['Cache']));
    children.push(buildCacheTable(cache));
  }
  if (trends.length) {
    children.push(createEl('h3', {}, ['Recent trends']));
    children.push(buildTrendsTable(trends));
  }
  ctx.state.profileBody.replaceChildren(...children);
}

function renderProfileEmpty(ctx) {
  ctx.state.profileBody.replaceChildren(createEl('div', { class: 'empty-state' },
    ['No profiling data yet — it accumulates in memory while the server handles requests, and resets on restart.']));
}

function buildTimingTable(rows) {
  const body = rows.map((r) => createEl('tr', {}, [
    createEl('td', {}, [r.operation ?? '--']),
    createEl('td', { class: 'perf-table__num' }, [formatMs(r.avg_time_ms)]),
    createEl('td', { class: 'perf-table__num' }, [fmtInt(r.count)]),
    createEl('td', {}, [buildPctCell(r.percentage)]),
  ]));
  return createEl('div', { class: 'perf-table-wrap' }, [
    createEl('table', { class: 'perf-table' }, [
      createEl('thead', {}, [createEl('tr', {}, [
        createEl('th', { scope: 'col' }, ['Operation']),
        createEl('th', { scope: 'col' }, ['Avg ms']),
        createEl('th', { scope: 'col' }, ['Count']),
        createEl('th', { scope: 'col' }, ['%']),
      ])]),
      createEl('tbody', {}, body),
    ]),
  ]);
}

function buildPctCell(fraction) {
  const pct = Math.max(0, Math.min(100, (fraction ?? 0) * 100));
  const fill = createEl('div', { class: 'perf-bar__fill' });
  fill.style.width = `${pct}%`;
  return createEl('div', { class: 'perf-table__pct-cell' }, [
    createEl('div', { class: 'perf-bar perf-bar--sm' }, [fill]),
    createEl('span', { class: 'perf-table__pct-text small' }, [`${pct.toFixed(1)}%`]),
  ]);
}

// Per model over the range (plan W5): how much of its prompts was reused,
// token-weighted, and how many requests had each outcome and cause.
function buildCacheTable(rows) {
  const counts = (bag, label = (k) => k) => Object.entries(bag ?? {})
    .map(([k, n]) => `${label(k)} ${n}`).join(' · ') || '--';
  const body = rows.map((r) => createEl('tr', {}, [
    createEl('td', {}, [r.model ?? '--']),
    createEl('td', { class: 'perf-table__num' }, [fmtInt(r.requests)]),
    createEl('td', { class: 'perf-table__num' },
      [`${fmtPct(r.cache_share)} of ${formatTokens(r.prompt_tokens)}`]),
    createEl('td', {}, [counts(r.outcomes)]),
    createEl('td', {}, [counts(r.causes, causeWords)]),
  ]));
  return createEl('div', { class: 'perf-table-wrap' }, [
    createEl('table', { class: 'perf-table' }, [
      createEl('thead', {}, [createEl('tr', {}, [
        createEl('th', { scope: 'col' }, ['Model']),
        createEl('th', { scope: 'col' }, ['Requests']),
        createEl('th', { scope: 'col' }, ['Prompt reused']),
        createEl('th', { scope: 'col' }, ['Outcomes']),
        createEl('th', { scope: 'col' }, ['Why']),
      ])]),
      createEl('tbody', {}, body),
    ]),
  ]);
}

// Token-weighted ratios, each shown only when the range reported it: an
// all "--" column is noise. The two draft rates are different quantities
// (accepted out of drafted, gguf only; out of emitted, both engines).
const TREND_RATIOS = [
  ['cache_share', 'Cache'],
  ['draft_acceptance', 'Draft acc'],
  ['draft_share', 'Draft share'],
];

function buildTrendsTable(rows) {
  const last = rows.slice(-8);
  const ratios = TREND_RATIOS.filter(([key]) => last.some((r) => r[key] != null));
  const body = last.map((r) => createEl('tr', {}, [
    createEl('td', { class: 'perf-table__mono' }, [formatHour(r.hour)]),
    createEl('td', { class: 'perf-table__num' }, [formatMs(r.response_time_ms)]),
    createEl('td', { class: 'perf-table__num' }, [r.tokens_per_second != null ? r.tokens_per_second.toFixed(1) : '--']),
    createEl('td', { class: 'perf-table__num' }, [fmtInt(r.requests)]),
    ...ratios.map(([key]) => createEl('td', { class: 'perf-table__num' }, [fmtPct(r[key])])),
  ]));
  return createEl('div', { class: 'perf-table-wrap' }, [
    createEl('table', { class: 'perf-table' }, [
      createEl('thead', {}, [createEl('tr', {}, [
        createEl('th', { scope: 'col' }, ['Hour']),
        createEl('th', { scope: 'col' }, ['Resp ms']),
        createEl('th', { scope: 'col' }, ['Tok/s']),
        createEl('th', { scope: 'col' }, ['Requests']),
        ...ratios.map(([, label]) => createEl('th', { scope: 'col' }, [label])),
      ])]),
      createEl('tbody', {}, body),
    ]),
  ]);
}

function fmtPct(v) {
  return v != null ? `${(v * 100).toFixed(0)}%` : '--';
}

// ---------------------------------------------------------------------------
// observability config + JSONL log viewer
// ---------------------------------------------------------------------------

async function loadObservability(ctx) {
  const s = ctx.state;
  try {
    const [cfg, logs] = await Promise.all([
      api.adminGetConfig({ signal: ctx.signal }).catch(() => null),
      api.adminGetLogs(s.activeStream, 50, { signal: ctx.signal }).catch(() => null),
    ]);
    if (!ctx.alive) return;
    if (cfg?.effective) applyObsConfig(ctx, cfg);
    renderObsLogs(ctx, logs);
  } catch {
    if (ctx.alive) renderObsLogs(ctx, null);
  }
}

function applyObsConfig(ctx, cfg) {
  const s = ctx.state;
  const eff = cfg.effective ?? {};
  if (eff.observability_level) s.obsLevelSelect.value = eff.observability_level;
  if (eff.observability_retention_days != null) {
    s.obsRetentionInput.value = String(eff.observability_retention_days);
  }
}

async function updateObsSetting(ctx, patch) {
  const s = ctx.state;
  s.obsStatusEl.textContent = 'Saving…';
  try {
    const cfg = await api.adminSetConfig(patch, { signal: ctx.signal });
    if (!ctx.alive) return;
    applyObsConfig(ctx, cfg);
    s.obsStatusEl.textContent = 'Saved';
    loadLogs(ctx, s.activeStream);
  } catch (err) {
    if (ctx.alive) s.obsStatusEl.textContent = `Save failed: ${err.message}`;
  }
}

async function loadLogs(ctx, stream) {
  const s = ctx.state;
  s.activeStream = stream;
  for (const btn of s.streamButtons) {
    const on = btn.dataset.stream === stream;
    btn.classList.toggle('perf__range-btn--active', on);
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
  try {
    const data = await api.adminGetLogs(stream, 50, { signal: ctx.signal });
    if (!ctx.alive || s.activeStream !== stream) return;
    renderObsLogs(ctx, data);
  } catch {
    if (ctx.alive && s.activeStream === stream) renderObsLogs(ctx, null);
  }
}

function summarizeRecord(rec) {
  const parts = [];
  if (rec.model_id || rec.model) parts.push(rec.model_id ?? rec.model);
  if (rec.endpoint) parts.push(rec.endpoint);
  if (rec.status_code != null) parts.push(`HTTP ${rec.status_code}`);
  if (rec.tokens_per_second != null) parts.push(`${Number(rec.tokens_per_second).toFixed(1)} tok/s`);
  if (rec.total_ms != null) parts.push(`${Number(rec.total_ms).toFixed(0)} ms`);
  if (rec.system_ram_used_gb != null) parts.push(`RAM ${Number(rec.system_ram_used_gb).toFixed(1)} GB`);
  if (rec.error_type || rec.error) parts.push(rec.error_type ?? rec.error);
  return parts.join(' · ') || JSON.stringify(rec);
}

function renderObsLogs(ctx, data) {
  const s = ctx.state;
  const records = data?.records ?? [];
  const level = data?.level ?? s.obsLevelSelect.value ?? 'off';
  if (!records.length) {
    const note = level === 'off'
      ? 'Observability is off — choose minimal, standard, or debug above to record structured JSONL telemetry.'
      : `No ${s.activeStream} records written yet.`;
    s.obsLogsBody.replaceChildren(createEl('div', { class: 'empty-state' }, [note]));
    return;
  }
  const rows = records.slice(0, 25).map((r) => createEl('tr', {}, [
    createEl('td', { class: 'perf-table__mono' }, [formatClock(r.ts ?? r.timestamp)]),
    createEl('td', { class: 'perf-table__mono' }, [r.event_type ?? r.type ?? s.activeStream]),
    createEl('td', { class: 'perf-table__mono perf__obs-summary' }, [summarizeRecord(r)]),
  ]));
  s.obsLogsBody.replaceChildren(createEl('div', { class: 'perf-table-wrap' }, [
    createEl('table', { class: 'perf-table' }, [
      createEl('thead', {}, [createEl('tr', {}, [
        createEl('th', { scope: 'col' }, ['Time']),
        createEl('th', { scope: 'col' }, ['Event']),
        createEl('th', { scope: 'col' }, ['Details']),
      ])]),
      createEl('tbody', {}, rows),
    ]),
  ]));
}

// ---------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------

function fmtGb(v) {
  return v != null ? `${v.toFixed(1)} GB` : '--';
}

function fmtInt(v) {
  return v != null ? v.toLocaleString() : '--';
}

function formatMs(ms) {
  return ms != null ? ms.toFixed(1) : '--';
}

function formatClock(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '--:--:--';
  return d.toLocaleTimeString([], { hour12: false });
}

function formatHour(iso) {
  return typeof iso === 'string' ? iso.replace('T', ' ').slice(0, 16) : '--';
}
