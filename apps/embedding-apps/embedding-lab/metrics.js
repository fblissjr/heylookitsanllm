// Pure retrieval metrics for the Embedding Lab. No DOM, no network.
// Server vectors are L2-normalised, but cosine is computed in full so a test
// set run against any vectors gives the right answer.

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

/**
 * Rank one query against the whole pool.
 * `positives` is a Set of pool indices that are right answers; every other pool
 * entry (this case's negatives and every other case's documents) is wrong.
 * `excluded` holds pool entries identical to the query itself: a query always
 * matches its own copy, so those take no part in the ranking.
 * rank: 1 + the number of wrong documents scoring above the best right one.
 * margin: best right score minus best wrong score (negative means it failed).
 */
export function evaluateCase(query, pool, positives, excluded = new Set()) {
  let bestPos = -Infinity, bestNeg = -Infinity, bestPosIndex = -1, bestNegIndex = -1;
  const scores = pool.map((vec, i) => cosine(query, vec));
  scores.forEach((score, i) => {
    if (excluded.has(i)) return;
    if (positives.has(i)) { if (score > bestPos) { bestPos = score; bestPosIndex = i; } }
    else if (score > bestNeg) { bestNeg = score; bestNegIndex = i; }
  });
  if (bestPosIndex < 0) throw new Error('A case needs at least one positive.');
  const rank = 1 + scores.filter((score, i) => !excluded.has(i) && !positives.has(i) && score > bestPos).length;
  return {
    rank, bestPos, bestNeg: bestNegIndex < 0 ? null : bestNeg, bestNegIndex, bestPosIndex,
    margin: bestNegIndex < 0 ? null : bestPos - bestNeg,
  };
}

/** Roll per-case results up to the numbers a configuration is compared on. */
export function summarize(results) {
  const n = results.length;
  const margins = results.map((r) => r.margin).filter((m) => m != null);
  return {
    n,
    r1: results.filter((r) => r.rank <= 1).length / n,
    r3: results.filter((r) => r.rank <= 3).length / n,
    mrr: results.reduce((sum, r) => sum + 1 / r.rank, 0) / n,
    meanMargin: margins.length ? margins.reduce((a, b) => a + b, 0) / margins.length : null,
    minMargin: margins.length ? Math.min(...margins) : null,
  };
}

/**
 * Validate and normalise a test set into a deduplicated document pool.
 * Item forms: "text", {text}, {file}, {text, file}. Returns
 * { queries: [item], pool: [item], cases: [{id, query, positives:[poolIdx], negatives:[poolIdx]}] }.
 * `known(file)` says whether a referenced file exists in the chosen folder.
 */
export function buildPlan(set, known) {
  if (!set || !Array.isArray(set.cases) || !set.cases.length) throw new Error('set.json needs a non-empty "cases" array.');
  const pool = [], index = new Map(), queries = [], cases = [], seenIds = new Set();
  const norm = (raw, where) => {
    const item = typeof raw === 'string' ? { text: raw } : raw;
    if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error(`${where}: expected a string or {text, file}.`);
    const extra = Object.keys(item).filter((k) => k !== 'text' && k !== 'file');
    if (extra.length) throw new Error(`${where}: unknown field "${extra[0]}".`);
    const text = item.text == null ? undefined : String(item.text);
    const file = item.file == null ? undefined : String(item.file);
    if (!text?.trim() && !file) throw new Error(`${where}: needs text or a file.`);
    if (file && !known(file)) throw new Error(`${where}: file "${file}" is not in the chosen folder.`);
    return { ...(text?.trim() ? { text } : {}), ...(file ? { file } : {}) };
  };
  const keyOf = (item) => JSON.stringify([item.text ?? null, item.file ?? null]);
  const add = (raw, where) => {
    const item = norm(raw, where);
    const key = keyOf(item);
    if (!index.has(key)) { index.set(key, pool.length); pool.push(item); }
    return index.get(key);
  };
  set.cases.forEach((c, i) => {
    const id = String(c.id ?? `case ${i + 1}`);
    if (seenIds.has(id)) throw new Error(`Duplicate case id "${id}".`);
    seenIds.add(id);
    const where = (part) => `${id} ${part}`;
    const query = norm(c.query, where('query'));
    const positives = (Array.isArray(c.positives) ? c.positives : []).map((p, j) => add(p, where(`positives[${j}]`)));
    const negatives = (Array.isArray(c.negatives) ? c.negatives : []).map((p, j) => add(p, where(`negatives[${j}]`)));
    if (!positives.length) throw new Error(`${id}: needs at least one positive.`);
    if (positives.some((p) => negatives.includes(p))) throw new Error(`${id}: the same item is both a positive and a negative.`);
    queries.push(query);
    cases.push({ id, query, positives, negatives, excluded: [] });
  });
  // A pool item identical to the query is the query itself; it would always win.
  // Resolved after every case has added its documents, so another case's copy counts too.
  cases.forEach((c) => {
    const own = index.get(keyOf(c.query));
    if (own == null) return;
    if (c.positives.includes(own)) throw new Error(`${c.id}: the query is identical to one of its own positives.`);
    c.excluded = [own];
  });
  return { queries, pool, cases };
}
