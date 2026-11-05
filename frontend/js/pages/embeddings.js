// Embeddings studio: Compare workbench and Embedding Lab (retrieval benchmark).
import { createPage } from "../page.js";
import { createEl as el } from "../utils.js";
import { api, requestId, httpError } from "../api.js";
import * as drawer from "../settings-drawer.js";
import {
  buildPlan, buildSampleDataset, embedWithRetry,
  evaluateCase, fileToBase64, kindOf, maxBytesFor, mediaBlock, mimeOf,
  planBatches, summarize, textBlock, thumbnail,
} from "../embed-kit.js";

function button(text, action, cls = "btn btn--sm") {
  return el("button", { type: "button", class: cls, onclick: action }, [text]);
}
function field(label, control) {
  return el("label", { class: "embeddings__field" }, [el("span", {}, [label]), control]);
}
function download(value, filename = "embeddings.json") {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const a = el("a", { href: url, download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const signed = (n) => (n == null ? "–" : `${n < 0 ? "−" : "+"}${Math.abs(n).toFixed(3)}`);
const fmtSec = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtSpan = (r) => (r.start != null && r.end != null ? ` [${fmtSec(r.start)}–${fmtSec(r.end)}]` : "");
const isMlx = (m) => m.engine?.runtime?.value === "mlx-vlm" || m.provider === "mlx" || Boolean(m.engine?.embedding?.video_windows);

export default createPage({
  async setup(ctx) {
    ctx.onTeardown(drawer.registerSettings({ samplers: "hidden" }));
    const s = ctx.state;
    s.mode = "compare";
    s.items = [];
    s.models = [];
    s.runId = null;
    s.abort = null;
    // Embedding Lab state
    s.labSetName = "";
    s.labFiles = new Map();
    s.labPlan = null;
    s.labConfigs = [];
    s.labResults = [];

    const model = el("select", { "aria-label": "Embedding model" });
    const dimensions = el("select", { "aria-label": "Embedding dimensions" });
    const status = el("p", { role: "status", "aria-live": "polite" }, ["Loading embedding models…"]);
    const note = el("p", { class: "muted small" });
    const cards = el("div", { class: "embeddings__items" });
    const results = el("section", { class: "embeddings__results", "aria-label": "Embedding results" });
    const controls = el("fieldset", { class: "embeddings__controls" });

    const run = button("Embed & compare", execute);
    const stop = button("Stop", async () => {
      s.abort?.abort();
      if (!s.runId) return;
      const id = s.runId;
      try {
        const response = await fetch(`/v1/requests/${encodeURIComponent(id)}`, { method: "DELETE" });
        if (!response.ok) throw await httpError(response);
        if (ctx.alive && s.runId === id) status.textContent = "Cancellation requested. Waiting for the current operation to finish…";
      } catch (error) { if (ctx.alive) status.textContent = error.message; }
    });
    stop.hidden = true;
    const add = button("Add item", () => { addItem(); invalidate(); });

    // Mode tabs: Compare (default) | Embedding Lab
    const tabCompare = el("button", { type: "button", class: "btn btn--sm embeddings__tab", "aria-pressed": "true" }, ["Compare"]);
    const tabLab = el("button", { type: "button", class: "btn btn--sm embeddings__tab", "aria-pressed": "false" }, ["Embedding Lab"]);
    const modeBar = el("div", { class: "embeddings__modes", role: "group", "aria-label": "Embedding tools" }, [
      tabCompare, tabLab,
    ]);

    const comparePane = el("div", { class: "embeddings__pane" }, [
      cards,
      el("div", { class: "embeddings__toolbar" }, [add, run]),
    ]);
    const labPane = el("div", { class: "embeddings__pane", hidden: true });

    function switchMode(mode) {
      s.mode = mode;
      tabCompare.setAttribute("aria-pressed", String(mode === "compare"));
      tabLab.setAttribute("aria-pressed", String(mode === "lab"));
      comparePane.hidden = mode !== "compare";
      results.hidden = mode !== "compare";
      labPane.hidden = mode !== "lab";
    }
    tabCompare.addEventListener("click", () => switchMode("compare"));
    tabLab.addEventListener("click", () => switchMode("lab"));

    controls.append(
      el("div", { class: "embeddings__toolbar" }, [field("Model", model), field("Dimensions", dimensions)]),
      note,
      modeBar,
      comparePane,
      labPane,
    );

    ctx.el.append(el("div", { class: "embeddings" }, [
      el("h1", {}, ["Embeddings"]),
      el("p", { class: "muted" }, ["Compare meaning across text, images, audio and video, or benchmark retrieval configurations."]),
      el("details", {}, [el("summary", {}, ["Ways to use this"]), el("p", {}, [
        "Compare: put a query first and candidates after it, or combine text + media inside one item. Embedding Lab: evaluate Recall@1, MRR, and cosine margins across models, dimensions, and task prompts.",
      ])]),
      controls, stop, status, results,
    ]));

    const selected = () => s.models.find((m) => m.id === model.value);
    function invalidate() { results.replaceChildren(); }
    function prompts(select, previous = "", emptyLabel = "No task prefix") {
      select.replaceChildren(
        el("option", { value: "" }, [emptyLabel]),
        ...Object.keys(selected()?.engine.embedding.prompts ?? {}).map((name) => el("option", { value: name }, [name])),
      );
      if ([...select.options].some((o) => o.value === previous)) select.value = previous;
    }

    // ---- Mode 2: Embedding Lab (Retrieval Benchmark) -----------------------

    const labQueryPrompt = el("select", { "aria-label": "Lab query prompt" });
    const labDocPrompt = el("select", { "aria-label": "Lab document prompt" });
    const labFolderInput = el("input", { type: "file", webkitdirectory: true, multiple: true, hidden: true, "aria-label": "Choose benchmark folder" });
    const labSetInfo = el("p", { class: "muted small" }, ["Load the built-in 8-case multimodal sample set or choose a folder with set.json."]);
    const labConfigList = el("ol", { class: "embeddings__lab-configs" });
    const labProgress = el("progress", { class: "embeddings__progress", value: 0, max: 1, hidden: true });
    const labResultsBox = el("div", { class: "embeddings__lab-results" });
    const labBlocks = new Map();

    const labLabel = (c) => `${c.model} · ${c.dimensions}d · query: ${c.queryPrompt || "none"} · doc: ${c.docPrompt || "none"}`;
    const labRunBtn = button("Run benchmark", () => runLabBenchmark());
    const labDownloadBtn = button("Download benchmark JSON", () => {
      if (!s.labPlan || !s.labResults.length) return;
      download({ set: s.labSetName, cases: s.labPlan.cases.map((c) => c.id), results: s.labResults }, "embedding-lab-results.json");
    });
    labRunBtn.disabled = true;
    labDownloadBtn.disabled = true;

    function resetLabResults() {
      s.labResults = [];
      labResultsBox.replaceChildren();
      labDownloadBtn.disabled = true;
    }

    function renderLabConfigs() {
      labConfigList.replaceChildren(...s.labConfigs.map((c, i) =>
        el("li", { class: "embeddings__lab-config-item" }, [
          el("span", {}, [labLabel(c)]),
          button("Remove", () => { s.labConfigs.splice(i, 1); renderLabConfigs(); }),
        ]),
      ));
      labRunBtn.disabled = !s.labPlan || !s.labConfigs.length;
    }

    function pushLabConfig(c) {
      if (!c.model || s.labConfigs.some((o) => labLabel(o) === labLabel(c))) return;
      s.labConfigs.push(c);
      renderLabConfigs();
    }

    const labSampleBtn = button("Load sample set (8 cases)", async () => {
      try {
        labBlocks.clear();
        resetLabResults();
        const { setName, files, plan } = await buildSampleDataset();
        s.labSetName = setName; s.labFiles = files; s.labPlan = plan;
        const media = plan.pool.filter((p) => p.file).length + plan.queries.filter((q) => q.file).length;
        labSetInfo.textContent = `${setName}: ${plan.cases.length} cases, ${plan.pool.length} documents, ${media} generated shape PNGs.`;
        if (!s.labConfigs.length && model.value) addLabAblation();
        else renderLabConfigs();
      } catch (err) {
        if (ctx.alive) status.textContent = err.message;
      }
    });

    function addLabAblation() {
      const names = Object.keys(selected()?.engine.embedding.prompts ?? {});
      const q = names.includes("query") ? "query" : (names[0] || "");
      const d = names.includes("document") ? "document" : (names[0] || "");
      pushLabConfig({ model: model.value, dimensions: Number(dimensions.value), queryPrompt: q, docPrompt: d });
      pushLabConfig({ model: model.value, dimensions: Number(dimensions.value), queryPrompt: "", docPrompt: "" });
    }

    labFolderInput.addEventListener("change", async (e) => {
      const files = new Map();
      for (const f of e.target.files || []) {
        const rel = (f.webkitRelativePath || f.name).split("/").slice(1).join("/") || f.name;
        files.set(rel, f);
      }
      e.target.value = "";
      s.labPlan = null; s.labFiles = files; labBlocks.clear();
      resetLabResults();
      try {
        const setFile = files.get("set.json");
        if (!setFile) throw new Error("No set.json at the top of that folder.");
        const set = JSON.parse(await setFile.text());
        const info = selected()?.engine.embedding;
        const plan = buildPlan(set, (path) => {
          const f = files.get(path);
          const k = f && kindOf(f);
          if (!k) return false;
          const cap = maxBytesFor(info, k);
          return f.size <= cap;
        });
        s.labPlan = plan; s.labSetName = set.name || "test set";
        const media = plan.pool.filter((p) => p.file).length + plan.queries.filter((q) => q.file).length;
        labSetInfo.textContent = `${s.labSetName}: ${plan.cases.length} cases, ${plan.pool.length} documents, ${media} media references.`;
      } catch (err) {
        labSetInfo.textContent = `Cannot use that folder: ${err.message}`;
      }
      renderLabConfigs();
    });

    labPane.append(
      el("div", { class: "embeddings__toolbar" }, [
        labSampleBtn, labFolderInput, button("Choose folder (set.json)", () => labFolderInput.click()),
      ]),
      labSetInfo,
      el("div", { class: "embeddings__toolbar" }, [
        field("Query prompt", labQueryPrompt),
        field("Document prompt", labDocPrompt),
        button("Add configuration", () => pushLabConfig({
          model: model.value, dimensions: Number(dimensions.value),
          queryPrompt: labQueryPrompt.value, docPrompt: labDocPrompt.value,
        })),
        button("Add prompts on vs. off", addLabAblation),
      ]),
      labConfigList,
      el("div", { class: "embeddings__toolbar" }, [labRunBtn, labDownloadBtn]),
      labProgress,
      labResultsBox,
    );

    async function labEmbedAll(items, prompt, cfg) {
      const info = s.models.find((m) => m.id === cfg.model).engine.embedding;
      for (const item of items) {
        const kind = item.file && kindOf(s.labFiles.get(item.file));
        if (kind && !info.modalities.includes(kind)) throw new Error(`${cfg.model} does not take ${kind} (${item.file}).`);
      }
      const entries = items.map((item) => ({ item, bytes: item.file ? s.labFiles.get(item.file).size : 1 }));
      const vectors = [];
      for (const batch of planBatches(entries, info.limits)) {
        const input = [];
        for (const e of batch) {
          const content = [];
          if (e.item.text) content.push(textBlock(e.item.text));
          if (e.item.file) {
            if (!labBlocks.has(e.item.file)) labBlocks.set(e.item.file, mediaBlock(s.labFiles.get(e.item.file)));
            content.push(await labBlocks.get(e.item.file));
          }
          input.push({ content, ...(prompt ? { prompt_name: prompt } : {}) });
        }
        s.runId = requestId();
        const res = await embedWithRetry({
          model: cfg.model, dimensions: cfg.dimensions, items: input,
          reqId: s.runId, signal: s.abort.signal,
        });
        if (res.data.length !== batch.length) throw new Error("The server returned the wrong number of vectors.");
        for (const row of res.data) vectors.push(row.embedding);
      }
      return vectors;
    }

    function renderLabResults() {
      labResultsBox.replaceChildren();
      if (!s.labResults.length) return;
      const cols = [
        ["Recall@1", (sum) => sum.r1, (v) => `${(v * 100).toFixed(0)}%`],
        ["Recall@3", (sum) => sum.r3, (v) => `${(v * 100).toFixed(0)}%`],
        ["MRR", (sum) => sum.mrr, (v) => v.toFixed(3)],
        ["Mean margin", (sum) => sum.meanMargin, signed],
        ["Worst margin", (sum) => sum.minMargin, signed],
      ];
      const bests = cols.map(([, get]) => {
        const vals = s.labResults.map((r) => get(r.summary)).filter((v) => v != null);
        return vals.length > 1 ? Math.max(...vals) : null;
      });
      const sumTable = el("table", { class: "embeddings__matrix" }, [
        el("caption", {}, [`Summary: ${s.labPlan.cases.length} cases`]),
        el("thead", {}, [el("tr", {}, [
          el("th", {}, ["Configuration"]),
          ...cols.map(([title]) => el("th", {}, [title])),
          el("th", {}, ["Seconds"]),
        ])]),
        el("tbody", {}, s.labResults.map((r) => el("tr", {}, [
          el("th", {}, [r.label]),
          ...cols.map(([, get, fmt], j) => {
            const v = get(r.summary);
            const isBest = bests[j] != null && v === bests[j];
            return el("td", { class: isBest ? "embeddings__best" : "" }, [v == null ? "–" : fmt(v)]);
          }),
          el("td", {}, [r.seconds.toFixed(2)]),
        ]))),
      ]);
      const describe = (item) => [item?.text && item.text.slice(0, 48), item?.file].filter(Boolean).join(" + ");
      const caseTable = el("table", { class: "embeddings__matrix" }, [
        el("caption", {}, ["Per case: rank of best positive and margin over best negative"]),
        el("thead", {}, [el("tr", {}, [
          el("th", {}, ["Case"]),
          ...s.labResults.map((r) => el("th", {}, [r.label])),
        ])]),
        el("tbody", {}, s.labPlan.cases.map((c, i) => el("tr", {}, [
          el("th", {}, [c.id]),
          ...s.labResults.map((r) => {
            const x = r.cases[i];
            const td = el("td", { class: x.rank === 1 ? "embeddings__hit" : "embeddings__miss" }, [
              `${x.rank === 1 ? "✓" : "✗"} #${x.rank} ${signed(x.margin)}`,
            ]);
            if (x.rank !== 1 && x.bestNegIndex >= 0) {
              td.append(el("div", { class: "embeddings__beaten" }, [`beaten by: ${describe(s.labPlan.pool[x.bestNegIndex])}`]));
            }
            return td;
          }),
        ]))),
      ]);
      labResultsBox.append(
        el("div", { class: "embeddings__table-scroll" }, [sumTable]),
        el("div", { class: "embeddings__table-scroll" }, [caseTable]),
      );
    }

    async function runLabBenchmark() {
      if (!s.labPlan || !s.labConfigs.length) return;
      const order = [...s.labConfigs].sort((a, b) => a.model.localeCompare(b.model));
      s.labResults = [];
      renderLabResults();
      controls.disabled = true;
      stop.hidden = false;
      labProgress.hidden = false;
      labProgress.max = order.length;
      labProgress.value = 0;
      s.abort = new AbortController();
      try {
        for (const [i, cfg] of order.entries()) {
          status.textContent = `Running ${i + 1} of ${order.length}: ${labLabel(cfg)}…`;
          const t0 = performance.now();
          const pool = await labEmbedAll(s.labPlan.pool, cfg.docPrompt, cfg);
          const queries = await labEmbedAll(s.labPlan.queries, cfg.queryPrompt, cfg);
          const cases = s.labPlan.cases.map((c, idx) => ({
            id: c.id,
            ...evaluateCase(queries[idx], pool, new Set(c.positives), new Set(c.excluded)),
          }));
          s.labResults.push({
            cfg, label: labLabel(cfg), cases, summary: summarize(cases),
            seconds: (performance.now() - t0) / 1000,
          });
          labProgress.value = i + 1;
          renderLabResults();
        }
        status.textContent = `Done. Evaluated ${order.length} configuration(s).`;
      } catch (err) {
        if (ctx.alive) status.textContent = err.name === "AbortError" ? "Benchmark stopped." : `Failed: ${err.message}`;
      } finally {
        s.runId = null; s.abort = null; labProgress.hidden = true;
        if (ctx.alive) {
          controls.disabled = false;
          stop.hidden = true;
          labDownloadBtn.disabled = !s.labResults.length;
        }
      }
    }

    // ---- Mode 1: Compare workbench -----------------------------------------

    function refreshModel() {
      const info = selected()?.engine.embedding;
      dimensions.replaceChildren(...(info?.dimensions ?? []).map((n) => el("option", { value: n }, [String(n)])));
      if (info) dimensions.value = info.native_dimensions;
      note.textContent = info ? `Inputs: ${info.modalities.join(", ")}. Each item must fit ${info.max_input_tokens.toLocaleString()} tokens including media. ${Object.keys(info.prompts).length ? "Task prompts come from the checkpoint." : "This checkpoint has no named prompt catalog; task instructions can be entered as text."}` : "";
      for (const item of s.items) { prompts(item.prompt, item.prompt.value); item.paint?.(); }
      prompts(labQueryPrompt, "query");
      prompts(labDocPrompt, "document");
      run.disabled = !info;
      add.disabled = !info || s.items.length >= requireLimits().items;
      invalidate();
    }

    function requireLimits() {
      const lim = selected()?.engine?.embedding?.limits;
      if (!lim || !Number.isInteger(lim.items) || !Number.isInteger(lim.blocks_per_item) || !Number.isInteger(lim.request_media_bytes)) {
        throw new Error("Missing engine.embedding.limits on selected model.");
      }
      return lim;
    }

    function addItem(text = "", defaultPrompt = "") {
      const maxItems = () => requireLimits().items;
      const maxBlocks = () => requireLimits().blocks_per_item;
      if (s.items.length >= maxItems()) return;
      const item = { blocks: [] };
      item.name = el("input", { type: "text", class: "input", value: `Item ${s.items.length + 1}`, maxLength: 80, "aria-label": "Item label", oninput: invalidate });
      item.prompt = el("select", { "aria-label": "Task prompt", onchange: invalidate });
      prompts(item.prompt, defaultPrompt);
      const blocks = el("div", { class: "embeddings__blocks" });
      const paintBlocks = () => {
        const vw = selected()?.engine.embedding?.video_windows;
        blocks.replaceChildren(...item.blocks.map((block, index) => {
          let winSelect = null;
          if (block.kind === "video" && vw) {
            winSelect = el("select", {
              "aria-label": "Video windows",
              onchange: (e) => { block.windows = e.target.value; invalidate(); },
            }, [
              el("option", { value: "" }, ["Whole video only"]),
              el("option", { value: "fps" }, [`Windows: fixed rate (${vw.max_seconds}s)`]),
              el("option", { value: "content" }, [`Windows: content (${vw.max_seconds}s)`]),
            ]);
            winSelect.value = block.windows || "";
          }
          const content = block.file
            ? el("div", { class: "embeddings__media-row" }, [
              block.thumb ? el("img", { class: "embeddings__inline-thumb", src: block.thumb, alt: block.file.name }) : null,
              el("span", {}, [`${block.kind}: ${block.file.name}`]),
              winSelect,
            ])
            : el("textarea", { value: block.text, rows: 3, "aria-label": "Text content", placeholder: "Text, code, query or label description",
              oninput: (event) => { block.text = event.target.value; invalidate(); } });
          return el("div", { class: "embeddings__block" }, [content, el("div", { class: "embeddings__toolbar" }, [
            button("Move up", () => { if (index) { [item.blocks[index-1], item.blocks[index]] = [block, item.blocks[index-1]]; paintBlocks(); invalidate(); } }),
            button("Remove block", () => { item.blocks.splice(index, 1); paintBlocks(); invalidate(); }),
          ])]);
        }));
      };
      item.paint = paintBlocks;
      item.blocks.push({ kind: "text", text });
      const files = el("input", { type: "file", class: "input", multiple: true, accept: "image/*,audio/*,video/*", "aria-label": "Add images, audio or video" });
      files.addEventListener("change", async () => {
        const chosen = [...(files.files || [])];
        files.value = "";
        const info = selected()?.engine.embedding;
        try {
          for (const file of chosen) {
            const kind = kindOf(file);
            if (!kind) { status.textContent = `Unsupported file type: ${file.name}`; continue; }
            const cap = maxBytesFor(info, kind);
            if (file.size > cap) { status.textContent = `${file.name} exceeds the media size limit.`; continue; }
            if (item.blocks.length >= maxBlocks()) { status.textContent = `An item can contain at most ${maxBlocks()} blocks.`; break; }
            const thumb = await thumbnail(file, kind);
            item.blocks.push({ kind, file, thumb });
          }
          paintBlocks(); invalidate();
        } catch (err) {
          if (ctx.alive) status.textContent = err.message;
        }
      });
      item.card = el("section", { class: "embeddings__item" }, [
        el("div", { class: "embeddings__toolbar" }, [field("Label", item.name), field("Task", item.prompt),
          button("Remove item", () => { s.items.splice(s.items.indexOf(item), 1); item.card.remove(); add.disabled = false; invalidate(); })]),
        blocks, el("div", { class: "embeddings__toolbar" }, [button("Add text", () => {
          if (item.blocks.length < maxBlocks()) { item.blocks.push({ kind: "text", text: "" }); paintBlocks(); invalidate(); }
        }), field("Attach media", files)]),
      ]);
      s.items.push(item); cards.append(item.card); paintBlocks();
      add.disabled = s.items.length >= maxItems();
    }

    async function execute() {
      controls.disabled = true; invalidate();
      status.textContent = "Preparing inputs…";
      try {
        if (!s.items.length) throw new Error("Add at least one item.");
        const info = selected().engine.embedding;
        const reqCap = requireLimits().request_media_bytes;
        let bytes = 0;
        const input = [];
        const labels = s.items.map((item, i) => item.name.value.trim() || `Item ${i + 1}`);
        for (const [index, item] of s.items.entries()) {
          const content = [];
          for (const block of item.blocks) {
            if (!info.modalities.includes(block.kind)) throw new Error(`This model does not support ${block.kind}.`);
            if (!block.file) { if (block.text.trim()) content.push({ type: "text", text: block.text }); continue; }
            bytes += block.file.size;
            if (bytes > reqCap) throw new Error("Combined media exceeds the request size limit.");
            const data = await fileToBase64(block.file);
            if (!ctx.alive) return;
            const win = block.kind === "video" && block.windows && info.video_windows ? { sampling: block.windows } : null;
            content.push({
              type: block.kind,
              source: { type: "base64", media_type: mimeOf(block.file), data },
              ...(win ? { windows: win } : {}),
            });
          }
          if (!content.length) throw new Error(`${labels[index]} is empty.`);
          if (content.some((b) => b.windows) && content.length > 1) {
            throw new Error(`${labels[index]}: an item with video windows must contain only that video block.`);
          }
          input.push({ content, ...(item.prompt.value ? { prompt_name: item.prompt.value } : {}) });
        }
        if (!ctx.alive) return;
        const body = { model: model.value, dimensions: Number(dimensions.value), input };
        s.runId = requestId(); stop.hidden = false;
        status.textContent = "Embedding… The model loads automatically if needed.";
        const response = await fetch("/v1/embeddings", { method: "POST", signal: ctx.signal,
          headers: { "Content-Type": "application/json", "X-Request-ID": s.runId }, body: JSON.stringify(body) });
        if (!response.ok) throw await httpError(response);
        const result = await response.json();
        if (!ctx.alive) return;
        const vectors = result.data.map((row) => row.embedding);
        const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
        const table = el("table", { class: "embeddings__matrix" }, [
          el("caption", {}, ["Cosine similarity — higher means more similar"]),
          el("thead", {}, [el("tr", {}, [el("th", { scope: "col" }, ["Item"]), ...labels.map((label) => el("th", { scope: "col" }, [label]))])]),
          el("tbody", {}, vectors.map((vector, i) => el("tr", {}, [el("th", { scope: "row" }, [labels[i]]),
            ...vectors.map((other, j) => {
              const sim = dot(vector, other);
              const pct = Math.round(Math.max(0, Math.min(1, sim)) * 28);
              const td = el("td", {}, [sim.toFixed(3)]);
              if (i !== j && pct > 0) td.style.background = `color-mix(in oklch, var(--brand) ${pct}%, var(--surface))`;
              return td;
            })]))),
        ]);
        const ranking = vectors.slice(1).map((v, i) => ({ label: labels[i+1], score: dot(vectors[0], v) })).sort((a, b) => b.score - a.score);
        const winHits = [];
        result.data.forEach((row, i) => {
          if (i === 0 || !Array.isArray(row.windows)) return;
          for (const w of row.windows) {
            winHits.push({ label: `${labels[i]}${fmtSpan(w)}`, score: dot(vectors[0], w.embedding) });
          }
        });
        winHits.sort((a, b) => b.score - a.score);
        results.append(el("h2", {}, ["Results"]), el("p", {}, [`${result.model} · ${result.dimensions} dimensions · ${result.data.length} vectors`]),
          ...(ranking.length ? [el("h3", {}, [`Closest to ${labels[0]}`]), el("ol", {}, ranking.map((r) => el("li", {}, [`${r.label}: ${r.score.toFixed(3)}`])))] : []),
          ...(winHits.length ? [el("h3", {}, [`Closest video windows to ${labels[0]}`]), el("ol", { class: "embeddings__windows-list" }, winHits.map((r) => el("li", {}, [`${r.label}: ${r.score.toFixed(3)}`])))] : []),
          el("div", { class: "embeddings__table-scroll", tabIndex: 0, "aria-label": "Similarity table" }, [table]),
          button("Download vectors", () => download({ labels, ...result })),
          el("details", {}, [el("summary", {}, ["Inspect vectors"]), el("pre", {}, [JSON.stringify(result.data, null, 2)])]));
        status.textContent = `Complete${result.elapsed_ms != null ? ` in ${(result.elapsed_ms / 1000).toFixed(2)}s` : ""}. ${result.input_tokens == null ? "Token usage unavailable." : `${result.input_tokens} input tokens.`}`;
      } catch (error) { if (ctx.alive) status.textContent = error.message; }
      finally { s.runId = null; if (ctx.alive) { controls.disabled = false; stop.hidden = true; } }
    }

    model.addEventListener("change", () => refreshModel());
    dimensions.addEventListener("change", () => invalidate());
    ctx.onTeardown(() => {
      s.abort?.abort();
      if (s.runId) fetch(`/v1/requests/${encodeURIComponent(s.runId)}`, { method: "DELETE", keepalive: true }).catch(() => {});
    });
    try {
      const data = await api.listModels();
      if (!ctx.alive) return;
      s.models = (data.data ?? []).filter((m) => m.engine?.embedding).sort((a, b) => Number(isMlx(b)) - Number(isMlx(a)));
      model.replaceChildren(...s.models.map((m) => el("option", { value: m.id }, [m.id])));
      refreshModel();
      if (s.models.length) {
        const names = Object.keys(selected()?.engine.embedding.prompts ?? {});
        const q = names.includes("query") ? "query" : "";
        const d = names.includes("document") ? "document" : "";
        addItem("A dog playing on a beach.", q);
        addItem("A puppy running beside the ocean.", d);
        addItem("A database query joining two tables.", d);
      }
      status.textContent = s.models.length ? "Ready. Edit the examples or attach your own media." : "No embedding models found. Add a supported checkpoint to a scanned model folder and rescan on Models.";
    } catch (error) { if (ctx.alive) { status.textContent = error.message; run.disabled = true; } }
  },
});
