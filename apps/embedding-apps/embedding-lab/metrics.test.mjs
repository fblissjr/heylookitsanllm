// Run: node --test apps/embedding-apps/embedding-lab/metrics.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPlan, cosine, evaluateCase, summarize } from './metrics.js';

test('cosine of parallel, orthogonal and zero vectors', () => {
  assert.equal(cosine([1, 0], [2, 0]), 1);
  assert.equal(cosine([1, 0], [0, 1]), 0);
  assert.equal(cosine([0, 0], [1, 0]), 0);
});

test('rank counts only wrong documents above the best right one', () => {
  const pool = [[0, 1], [1, 0], [0.9, 0.1], [0.6, 0.8]];
  // query points along x: pool[1] is best, pool[2] next, pool[3] then pool[0].
  const right = evaluateCase([1, 0], pool, new Set([1]));
  assert.equal(right.rank, 1);
  assert.ok(right.margin > 0);
  const wrong = evaluateCase([1, 0], pool, new Set([3]));
  assert.equal(wrong.rank, 3); // pool[1] and pool[2] both outscore pool[3]
  assert.ok(wrong.margin < 0);
  const second = evaluateCase([1, 0], pool, new Set([1, 2]));
  assert.equal(second.rank, 1); // two positives: the best of them leads
});

test('a case with no positive is an error, and no negatives gives a null margin', () => {
  assert.throws(() => evaluateCase([1, 0], [[1, 0]], new Set()), /positive/);
  assert.equal(evaluateCase([1, 0], [[1, 0]], new Set([0])).margin, null);
});

test('summarize', () => {
  const s = summarize([{ rank: 1, margin: 0.2 }, { rank: 2, margin: -0.1 }, { rank: 5, margin: -0.3 }, { rank: 1, margin: null }]);
  assert.equal(s.r1, 0.5);
  assert.equal(s.r3, 0.75);
  assert.ok(Math.abs(s.mrr - (1 + 0.5 + 0.2 + 1) / 4) < 1e-12);
  assert.ok(Math.abs(s.meanMargin - (-0.2 / 3)) < 1e-12);
  assert.equal(s.minMargin, -0.3);
});

const has = () => true;

test('buildPlan dedupes the pool across cases and tracks indexes', () => {
  const plan = buildPlan({ cases: [
    { id: 'a', query: 'q1', positives: ['doc one'], negatives: ['shared'] },
    { id: 'b', query: 'q2', positives: ['shared'], negatives: [{ text: 'doc one' }] },
  ] }, has);
  assert.equal(plan.pool.length, 2);
  assert.deepEqual(plan.cases[0].positives, [0]);
  assert.deepEqual(plan.cases[0].negatives, [1]);
  assert.deepEqual(plan.cases[1].positives, [1]);
  assert.deepEqual(plan.cases[1].negatives, [0]);
});

test('a pool item identical to the query is excluded from that case only', () => {
  const plan = buildPlan({ cases: [
    { id: 'a', query: 'red circle', positives: ['pic'], negatives: ['other'] },
    { id: 'b', query: 'unrelated', positives: ['x'], negatives: ['red circle'] },
  ] }, has);
  const redCircle = plan.pool.findIndex((p) => p.text === 'red circle');
  assert.deepEqual(plan.cases[0].excluded, [redCircle]);
  assert.deepEqual(plan.cases[1].excluded, []);
  // Without the exclusion the identical copy (cosine 1) would outrank the positive.
  const pool = plan.pool.map((p) => (p.text === 'red circle' ? [1, 0] : p.text === 'pic' ? [0.8, 0.6] : [0, 1]));
  const q = [1, 0];
  assert.equal(evaluateCase(q, pool, new Set(plan.cases[0].positives)).rank, 2);
  const fixed = evaluateCase(q, pool, new Set(plan.cases[0].positives), new Set(plan.cases[0].excluded));
  assert.equal(fixed.rank, 1);
  assert.ok(fixed.margin > 0);
});

test('a query identical to its own positive is rejected', () => {
  assert.throws(() => buildPlan({ cases: [{ id: 'x', query: 'same', positives: ['same'] }] }, has), /identical to one of its own positives/);
});

test('buildPlan rejects malformed sets with a message naming the place', () => {
  assert.throws(() => buildPlan({ cases: [] }, has), /cases/);
  assert.throws(() => buildPlan({ cases: [{ id: 'x', query: 'q', positives: [] }] }, has), /x: needs at least one positive/);
  assert.throws(() => buildPlan({ cases: [{ id: 'x', query: 'q', positives: ['p', 'p'], negatives: ['p'] }] }, has), /both a positive and a negative/);
  assert.throws(() => buildPlan({ cases: [{ id: 'x', query: { image: 'a' }, positives: ['p'] }] }, has), /unknown field "image"/);
  assert.throws(() => buildPlan({ cases: [{ id: 'x', query: { file: 'a.png' }, positives: ['p'] }] }, () => false), /not in the chosen folder/);
  assert.throws(() => buildPlan({ cases: [{ id: 'x', query: '  ', positives: ['p'] }] }, has), /needs text or a file/);
  assert.throws(() => buildPlan({ cases: [{ id: 'x', query: 'q', positives: ['p'] }, { id: 'x', query: 'q', positives: ['p'] }] }, has), /Duplicate case id/);
});
