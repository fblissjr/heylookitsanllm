// Reading the SERVER's view of a document, from inside the page.
//
// The suites assert on persisted outcomes rather than transient UI (see the
// hardening principles in ../README.md), so almost every check ends by asking
// the API what actually landed. Doing that with an in-page `fetch` -- rather
// than a node-side HTTP call -- keeps the read on the same origin and session
// the app itself uses, so it sees exactly what the app would see.

// GET `path` inside the page; parsed JSON, or null when the response is not ok.
// Suite-specific shapes (message counts, notebook rows) compose on top of this
// rather than each re-implementing the evaluate + fetch + json dance.
export async function serverGet(page, path) {
  return page.evaluate(async (p) => {
    const res = await fetch(p);
    return res.ok ? res.json() : null;
  }, path);
}
