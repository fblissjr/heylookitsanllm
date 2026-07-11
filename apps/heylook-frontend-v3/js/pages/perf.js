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
import { createEl, formatBytes } from '../utils.js';
import { api } from '../api.js';
import * as drawer from '../settings-drawer.js';

const RANGES = ['1h', '6h', '24h', '7d'];
const DEFAULT_RANGE = '1h';

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.modelRows = new Map();
    s.activeRange = DEFAULT_RANGE;

    buildSkeleton(ctx);
    // No sampler/page settings here -- register so the drawer still offers the
    // global Display prefs and hides the sampler panel.
    ctx.onTeardown(drawer.registerSettings({ samplers: 'hidden' }));

    s.refreshBtn.addEventListener('click', () => loadMetrics(ctx, true));
    for (const btn of s.rangeButtons) {
      btn.addEventListener('click', () => {
        if (btn.dataset.range === s.activeRange) return;
        loadProfile(ctx, btn.dataset.range);
      });
    }

    await Promise.all([
      loadMetrics(ctx, false),
      loadProfile(ctx, DEFAULT_RANGE),
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

  ctx.el.append(createEl('div', { class: 'perf' }, [
    s.errorEl,
    header,
    systemSection,
    profileSection,
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

function applyModelData(row, m) {
  row.memEl.textContent = formatBytes((m.memory_mb ?? 0) * 1024 * 1024);
  const pct = Math.max(0, Math.min(100, m.context_percent ?? 0));
  row.fillEl.style.width = `${pct}%`;
  row.ctxEl.textContent = `ctx ${fmtInt(m.context_used)} / ${fmtInt(m.context_capacity)} (${pct.toFixed(1)}%)`;
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
  if (!breakdown.length && !trends.length) {
    renderProfileEmpty(ctx);
    return;
  }

  const children = [];
  if (breakdown.length) {
    children.push(createEl('h3', {}, ['Timing breakdown']));
    children.push(buildTimingTable(breakdown));
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

function buildTrendsTable(rows) {
  const last = rows.slice(-8);
  const body = last.map((r) => createEl('tr', {}, [
    createEl('td', { class: 'perf-table__mono' }, [formatHour(r.hour)]),
    createEl('td', { class: 'perf-table__num' }, [formatMs(r.response_time_ms)]),
    createEl('td', { class: 'perf-table__num' }, [r.tokens_per_second != null ? r.tokens_per_second.toFixed(1) : '--']),
    createEl('td', { class: 'perf-table__num' }, [fmtInt(r.requests)]),
  ]));
  return createEl('div', { class: 'perf-table-wrap' }, [
    createEl('table', { class: 'perf-table' }, [
      createEl('thead', {}, [createEl('tr', {}, [
        createEl('th', { scope: 'col' }, ['Hour']),
        createEl('th', { scope: 'col' }, ['Resp ms']),
        createEl('th', { scope: 'col' }, ['Tok/s']),
        createEl('th', { scope: 'col' }, ['Requests']),
      ])]),
      createEl('tbody', {}, body),
    ]),
  ]);
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
