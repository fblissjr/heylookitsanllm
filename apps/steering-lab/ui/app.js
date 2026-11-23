const state = {
  status: null,
  vectors: [],
  scales: {},
  geometry: null,
  lastSweep: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return data;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function heatBg(metric, tok) {
  let v = 0;
  if (metric === "g_t") v = tok.g_t;
  else if (metric === "rel_delta") v = Math.min(1, tok.rel_delta * 5);
  else if (metric === "kl_vs_base") v = Math.min(1, (tok.kl_vs_base || 0) / 1.5);
  else if (metric === "ray_cos") v = Math.max(0, Math.min(1, (tok.ray_cos + 0.1) * 2.2));
  else if (metric === "entropy") v = Math.min(1, tok.entropy / 3.0);
  if (v <= 0.02) return "transparent";
  const hue = metric === "entropy" || metric === "kl_vs_base" ? 8 : metric === "ray_cos" ? 36 : 210;
  const alpha = (0.12 + 0.48 * Math.min(1, v)).toFixed(2);
  return `hsla(${hue}, 85%, 52%, ${alpha})`;
}

function activeArmPayload(label, override = {}) {
  const rankRaw = $("opt-rank-limit").value;
  return {
    label,
    scales: { ...state.scales },
    use_peer_ros: $("opt-peer-ros").checked,
    use_clamped_union: $("opt-clamped-union").checked,
    use_holds: $("opt-holds").checked,
    push_mode: $("opt-norm-matched").checked ? "norm_matched" : "mu_d_norm",
    use_s3_random: $("opt-s3-random").checked,
    layer_filter: $("opt-layer-filter").value,
    phase_override: $("opt-phase").value || null,
    token_mask_override: $("opt-token-mask").value || null,
    center_mode: $("opt-center-mode").value,
    rank_limit: rankRaw ? Number(rankRaw) : null,
    ...override,
  };
}

function renderStatus() {
  const s = state.status;
  const pill = $("status-pill");
  if (!s) return;
  if (s.model_loaded) {
    pill.textContent = `Live MLX: ${s.fingerprint?.model_id || "loaded"}`;
    pill.className = "pill pill--good";
    $("btn-unload").hidden = false;
  } else if (s.bank_loaded) {
    pill.textContent = `CPU Bank: ${s.fingerprint?.model_id || "ready"} (${s.vector_count} vectors, 0% GPU)`;
    pill.className = "pill pill--warn";
    $("btn-unload").hidden = true;
  } else {
    pill.textContent = "Idle (load a bank or model)";
    pill.className = "pill pill--muted";
    $("btn-unload").hidden = true;
  }
  if (s.default_model_path && !$("input-model-path").value) {
    $("input-model-path").value = s.default_model_path;
  }
}

function renderSliders() {
  const list = $("slider-list");
  list.innerHTML = "";
  const clSel = $("cl-source");
  clSel.innerHTML = `<option value="__blend__">Active Multi-Slider Blend</option>`;

  if (!state.vectors.length) {
    list.innerHTML = `<p class="empty-hint">No vectors loaded. Run <code>demo-bank</code> or <code>build</code> first.</p>`;
    return;
  }

  const allLayers = new Set();
  for (const vec of state.vectors) {
    vec.layers.forEach((l) => allLayers.add(l));
    if (!(vec.id in state.scales)) state.scales[vec.id] = 0;

    const opt = document.createElement("option");
    opt.value = vec.id;
    opt.textContent = `${vec.label} [${vec.operator}]`;
    clSel.appendChild(opt);

    const card = document.createElement("div");
    card.className = "slider-card";
    const val = Number(state.scales[vec.id] || 0);
    card.innerHTML = `
      <div class="slider-card__top">
        <span class="slider-card__name">${escapeHtml(vec.label)}</span>
        <code data-val="${escapeHtml(vec.id)}">${val.toFixed(2)}</code>
      </div>
      <div class="slider-card__meta">
        <code>${escapeHtml(vec.operator)}</code> · phase: <code>${escapeHtml(vec.phase)}</code> ·
        mask: <code>${escapeHtml(vec.token_mask)}</code> · L[${vec.layers.join(",")}]
      </div>
      <input type="range" min="${vec.scale.min}" max="${vec.scale.max}" step="0.05" value="${val}" aria-label="${escapeHtml(vec.label)}">
    `;
    const range = card.querySelector("input");
    const badge = card.querySelector("code[data-val]");
    range.addEventListener("input", () => {
      const next = Number(range.value);
      state.scales[vec.id] = next;
      badge.textContent = next.toFixed(2);
    });
    range.addEventListener("change", () => analyzeBlend());
    list.appendChild(card);
  }

  const sortedLayers = [...allLayers].sort((a, b) => a - b);
  for (const selId of ["opt-compose-layer", "cl-layer"]) {
    const sel = $(selId);
    const prev = sel.value;
    sel.innerHTML = sortedLayers.map((l) => `<option value="${l}">L${l}</option>`).join("");
    if (prev && sortedLayers.includes(Number(prev))) sel.value = prev;
    else if (sortedLayers.includes(47)) sel.value = "47";
    else if (sortedLayers.length) sel.value = String(sortedLayers[sortedLayers.length - 1]);
  }
}

function renderGeometry() {
  const g = state.geometry;
  if (!g || !g.layers) return;
  const layerKeys = Object.keys(g.layers).map(Number).sort((a, b) => a - b);
  const sel = $("geom-layer");
  const prev = sel.value;
  sel.innerHTML = layerKeys.map((l) => `<option value="${l}">L${l}</option>`).join("");
  if (prev && layerKeys.includes(Number(prev))) sel.value = prev;
  else if (layerKeys.includes(47)) sel.value = "47";
  else if (layerKeys.length) sel.value = String(layerKeys[layerKeys.length - 1]);

  // Rank-collapse strip across all captured layers (L15..L63) with FA/GDN tags
  $("rank-strip").innerHTML = layerKeys
    .map((l) => {
      const row = g.layers[String(l)];
      const kind = row.layer_kind || (l % 4 === 3 ? "FA" : "GDN");
      return `<span class="rank-pill"><strong>L${l} ${escapeHtml(kind)}</strong> effR=${row.effective_rank.toFixed(2)} · p90=${row.p90_abs_cos.toFixed(2)}</span>`;
    })
    .join("");

  renderCosineTable();
}

function renderCosineTable() {
  const g = state.geometry;
  const layer = $("geom-layer").value;
  const row = g?.layers?.[layer];
  if (!row || !row.concepts?.length) {
    $("cosine-wrap").innerHTML = `<p class="empty-hint">No cosine matrix for this layer.</p>`;
    return;
  }
  const head = row.concepts.map((c) => `<th>${escapeHtml(c.slice(0, 10))}</th>`).join("");
  const body = row.concepts
    .map((c, i) => {
      const cells = row.cosine_matrix[i]
        .map((v, j) => {
          const a = i === j ? 0 : Math.min(0.45, Math.abs(v) * 0.65);
          const bg = a > 0.04 ? `hsla(${v >= 0 ? 15 : 210}, 75%, 48%, ${a.toFixed(2)})` : "transparent";
          return `<td style="background:${bg}">${v.toFixed(2)}</td>`;
        })
        .join("");
      return `<tr><th>${escapeHtml(c.slice(0, 10))}</th>${cells}</tr>`;
    })
    .join("");
  $("cosine-wrap").innerHTML = `<table class="cosine-table"><thead><tr><th>L${layer}</th>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

async function analyzeBlend() {
  const layer = Number($("opt-compose-layer").value || 47);
  try {
    const res = await api("/api/compose", {
      method: "POST",
      body: JSON.stringify({
        layer,
        scales: state.scales,
        use_peer_ros: $("opt-peer-ros").checked,
        use_clamped_union: $("opt-clamped-union").checked,
        use_holds: $("opt-holds").checked,
        push_mode: $("opt-norm-matched").checked ? "norm_matched" : "mu_d_norm",
        use_s3_random: $("opt-s3-random").checked,
      }),
    });
    if (!res.active) {
      $("compose-readout").textContent = "All Regime-A sliders at L" + layer + " are 0 (exact identity).";
      return;
    }
    const adds = (res.xray?.adds || []).slice(0, 5).map((x) => `${x.token.trim()}(+${x.score.toFixed(2)})`).join(", ");
    const erases = (res.xray?.erases || []).slice(0, 5).map((x) => `${x.token.trim()}(${x.score.toFixed(2)})`).join(", ");
    $("compose-readout").textContent =
      `L${res.layer} | union_rank=${res.union_rank} | max_raw_eig=${res.max_raw_eig} (clamped=${res.clamped_eigs})\n` +
      `peer_ros=${res.peer_ros_applied} | holds=${res.holds_applied} | mode=${res.push_mode || "mu_d_norm"} | ||push||=${res.push_norm}\n` +
      (res.xray
        ? `[${res.xray.advisory ? "Advisory" : "Terminal"} X-Ray]\n  Adds:   ${adds}\n  Erases: ${erases}`
        : "No lm_head available for X-Ray.");
  } catch (err) {
    $("compose-readout").textContent = `Error: ${err.message}`;
  }
}

async function runXray() {
  const src = $("cl-source").value;
  const conceptId = src === "__blend__" ? state.vectors[0]?.id : src;
  if (!conceptId) return;
  const layer = Number($("geom-layer").value || 47);
  try {
    const res = await api("/api/xray", {
      method: "POST",
      body: JSON.stringify({ concept_id: conceptId, layer, component: "mu_d", top_k: 8 }),
    });
    const adds = res.adds.map((a) => `<li><span>${escapeHtml(a.token)}</span><span>+${a.score.toFixed(2)}</span></li>`).join("");
    const erases = res.erases.map((a) => `<li><span>${escapeHtml(a.token)}</span><span>${a.score.toFixed(2)}</span></li>`).join("");
    $("xray-out").innerHTML = `
      <p class="note-line"><strong>${escapeHtml(res.concept_id)} @ L${res.layer}</strong> — ${escapeHtml(res.advisory_note)}</p>
      <div class="xray-cols">
        <div><strong>Adds (+)</strong><ul class="xray-list">${adds}</ul></div>
        <div><strong>Erases (−)</strong><ul class="xray-list">${erases}</ul></div>
      </div>`;
  } catch (err) {
    $("xray-out").innerHTML = `<p class="note-line">X-Ray error: ${escapeHtml(err.message)}</p>`;
  }
}

async function runChainLens() {
  const src = $("cl-source").value;
  const payload = {
    concept_id: src === "__blend__" ? null : src,
    component: src === "__blend__" ? "blend" : "mu_d",
    layer: Number($("cl-layer").value || 47),
    scales: state.scales,
    use_peer_ros: $("opt-peer-ros").checked,
    use_clamped_union: $("opt-clamped-union").checked,
    use_holds: $("opt-holds").checked,
    use_jlens: $("cl-jlens").checked,
    direction_sign: Number($("cl-sign").value),
    alpha: Number($("cl-alpha").value),
    steps: Number($("cl-steps").value),
    beam_width: Number($("cl-beams").value),
  };
  try {
    const res = await api("/api/chain-lens", { method: "POST", body: JSON.stringify(payload) });
    const rows = res.chains
      .map(
        (c) => `
      <div class="chain-item">
        <span class="chain-item__text">${escapeHtml(c.text)}</span>
        <span class="chain-item__score">score ${c.score.toFixed(2)}${c.round_trip_cos !== null ? ` · rt ${c.round_trip_cos.toFixed(2)}` : ""}${c.round_trip_rank !== null ? ` (#${c.round_trip_rank})` : ""}${c.round_trip_percentile !== null ? ` p${Math.round(c.round_trip_percentile)}` : ""}</span>
      </div>`
      )
      .join("");
    const jlensTag = res.exact_upper_applied
      ? " · exact L_mid→L63"
      : res.jlens_applied
      ? ` · J-Lens val R²=${res.jlens_val_r2 ?? res.jlens_train_r2}`
      : " · direct";
    $("chain-lens-out").innerHTML =
      `<p class="note-line"><code>${escapeHtml(res.mode)}</code> · ${escapeHtml(res.source)} @ L${res.layer} (${escapeHtml(res.layer_kind || "")})${jlensTag} · α=${res.alpha}</p>` +
      rows;
  } catch (err) {
    $("chain-lens-out").innerHTML = `<p class="note-line">Chain Lens error: ${escapeHtml(err.message)}</p>`;
  }
}

function renderSweep() {
  const data = state.lastSweep;
  if (!data) return;
  const metric = $("heatmap-metric").value;
  const banner = $("sweep-mode-banner");
  banner.hidden = false;
  banner.textContent =
    data.mode === "live_mlx"
      ? "Live MLX run (fresh GDN/FA caches per arm, seed-locked)."
      : "Demo synthetic sweep (0% GPU — load an MLX checkpoint via Model… for live generation).";

  const grid = $("sweep-grid");
  grid.innerHTML = "";
  data.arms.forEach((arm, armIdx) => {
    const s = arm.summary;
    const mStatus = s.manifold?.status || "nominal";
    const mClass =
      mStatus === "manifold_collapse" || mStatus === "loop_collapse"
        ? "pill--bad"
        : mStatus === "in_manifold_shift"
        ? "pill--good"
        : "pill--muted";

    const card = document.createElement("article");
    card.className = "arm-card";
    card.innerHTML = `
      <div class="arm-card__head">
        <strong>${escapeHtml(arm.arm_label)}</strong>
        <span class="pill ${mClass}" title="${escapeHtml(s.manifold?.note || "")}">${escapeHtml(mStatus)}</span>
      </div>
      <div class="arm-card__stats">
        <span>think=${s.think_tokens}</span>
        <span>ans=${s.content_tokens}</span>
        <span>fire=${(s.gate_fire_frac * 100).toFixed(0)}%</span>
        <span>relΔ=${s.mean_rel_delta.toFixed(3)}</span>
        <span>H=${s.mean_entropy.toFixed(2)}</span>
        ${s.mean_kl_vs_base !== null && s.mean_kl_vs_base !== undefined ? `<span>KL=${s.mean_kl_vs_base.toFixed(2)}</span>` : ""}
        <span>rep4=${((s.ngram_repeat_4 || 0) * 100).toFixed(0)}%</span>
        ${s.mtp_share !== null ? `<span>MTP=${(s.mtp_share * 100).toFixed(0)}%</span>` : ""}
      </div>
      <div class="token-stream" data-arm="${armIdx}"></div>
    `;

    const stream = card.querySelector(".token-stream");
    let lastPhase = null;
    arm.tokens.forEach((tok) => {
      if (tok.phase !== lastPhase) {
        const tag = document.createElement("span");
        tag.className = "token-section-tag";
        tag.textContent = tok.phase;
        stream.appendChild(tag);
        lastPhase = tok.phase;
      }
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tok-chip";
      btn.textContent = tok.text;
      btn.style.background = heatBg(metric, tok);
      btn.title = `g_t=${tok.g_t} cos=${tok.ray_cos} relΔ=${tok.rel_delta} H=${tok.entropy}`;
      btn.addEventListener("click", () => inspectToken(arm, tok));
      stream.appendChild(btn);
    });
    grid.appendChild(card);
  });
}

async function runSweep() {
  const prompt = $("sweep-prompt").value.trim();
  if (!prompt) return;
  const arms = [
    activeArmPayload("Baseline (s=0)", { scales: {}, use_s3_random: false }),
    activeArmPayload("Active Mixer"),
  ];
  if ($("opt-compare-s3").checked) {
    arms.push(activeArmPayload("S3 Random Control", { use_s3_random: true }));
  }
  $("btn-run-sweep").disabled = true;
  try {
    state.lastSweep = await api("/api/sweep", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        arms,
        seed: Number($("sweep-seed").value || 42),
        max_tokens: Number($("sweep-max-tokens").value || 96),
      }),
    });
    renderSweep();
  } catch (err) {
    $("sweep-grid").innerHTML = `<p class="note-line">Sweep error: ${escapeHtml(err.message)}</p>`;
  } finally {
    $("btn-run-sweep").disabled = false;
  }
}

async function inspectToken(arm, tok) {
  const panel = $("token-inspector");
  $("inspector-title").textContent = `Token #${tok.index}: "${tok.text}" (${tok.phase})`;
  $("inspector-body").innerHTML = `
    <p class="note-line">
      <code>g_t=${tok.g_t}</code> · <code>ray_cos=${tok.ray_cos}</code> ·
      <code>relΔ=${tok.rel_delta}</code> · <code>H=${tok.entropy}</code>
      ${tok.mtp_accepted !== null ? ` · <code>mtp_ok=${tok.mtp_accepted}</code>` : ""}
    </p>
    <p class="empty-hint">Computing causal Δ logprob vs. unsteered forward…</p>
  `;
  if (typeof panel.showPopover === "function") panel.showPopover();
  else panel.setAttribute("open", "");

  try {
    const res = await api("/api/token-delta", {
      method: "POST",
      body: JSON.stringify({
        prompt: arm.prompt,
        token_ids: arm.tokens.map((t) => t.token_id),
        target_index: tok.index,
        arm: arm.plan,
      }),
    });
    const adds = res.adds.map((a) => `<li><span>${escapeHtml(a.token)}</span><span>+${a.delta_logprob.toFixed(3)}</span></li>`).join("");
    const erases = res.erases.map((a) => `<li><span>${escapeHtml(a.token)}</span><span>${a.delta_logprob.toFixed(3)}</span></li>`).join("");
    $("inspector-body").innerHTML = `
      <p class="note-line">
        <code>${escapeHtml(res.mode)}</code> · <code>g_t=${tok.g_t}</code> ·
        <code>ray_cos=${tok.ray_cos}</code> · <code>relΔ=${tok.rel_delta}</code> · <code>H=${tok.entropy}</code>
      </p>
      <div class="xray-cols">
        <div><strong>Causal Δ logprob (+)</strong><ul class="xray-list">${adds}</ul></div>
        <div><strong>Causal Δ logprob (−)</strong><ul class="xray-list">${erases}</ul></div>
      </div>`;
  } catch (err) {
    $("inspector-body").innerHTML += `<p class="note-line">Error: ${escapeHtml(err.message)}</p>`;
  }
}

async function refreshAll() {
  state.status = await api("/api/status");
  renderStatus();
  const vecRes = await api("/api/vectors");
  state.vectors = vecRes.vectors || [];
  renderSliders();
  if (state.status.bank_loaded) {
    state.geometry = await api("/api/bank/geometry");
    renderGeometry();
  }
}

// Safari fallback for closedby="any" light-dismiss on modal <dialog>
if (!("closedBy" in HTMLDialogElement.prototype)) {
  const outside = (e, dlg) => {
    const r = dlg.getBoundingClientRect();
    return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
  };
  for (const dlg of document.querySelectorAll('dialog[closedby="any"]')) {
    let downOut = false;
    dlg.addEventListener("pointerdown", (e) => {
      downOut = outside(e, dlg);
    });
    dlg.addEventListener("pointerup", (e) => {
      if (downOut && outside(e, dlg)) dlg.close();
    });
  }
}

$("btn-reset-scales").addEventListener("click", () => {
  for (const k of Object.keys(state.scales)) state.scales[k] = 0;
  renderSliders();
  analyzeBlend();
});
$("btn-compose").addEventListener("click", analyzeBlend);
$("btn-run-sweep").addEventListener("click", runSweep);
$("btn-chain-lens").addEventListener("click", runChainLens);
$("btn-xray").addEventListener("click", runXray);
$("geom-layer").addEventListener("change", renderCosineTable);
$("heatmap-metric").addEventListener("change", renderSweep);

const loadDlg = $("load-dialog");
$("btn-open-load").addEventListener("click", () => loadDlg.showModal());
loadDlg.addEventListener("close", async () => {
  const choice = loadDlg.returnValue;
  loadDlg.returnValue = "";
  if (choice !== "load") return;
  const modelPath = $("input-model-path").value.trim();
  if (!modelPath) return;
  try {
    $("status-pill").textContent = "Loading MLX checkpoint…";
    await api("/api/model/load", { method: "POST", body: JSON.stringify({ model_path: modelPath }) });
    await refreshAll();
  } catch (err) {
    $("status-pill").textContent = `Load failed: ${err.message}`;
    $("status-pill").className = "pill pill--bad";
  }
});

$("btn-unload").addEventListener("click", async () => {
  await api("/api/model/unload", { method: "POST" });
  await refreshAll();
});

refreshAll().catch((err) => {
  $("status-pill").textContent = `Init error: ${err.message}`;
  $("status-pill").className = "pill pill--bad";
});
