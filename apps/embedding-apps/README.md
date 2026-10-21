# embedding-apps

Small, self-contained apps on heylook's multimodal `POST /v1/embeddings`. Plain
static HTML and ES modules; no build step, no dependencies.

The server sends no CORS headers on purpose, so these pages cannot call it from
another origin. `serve.py` serves them and forwards three routes (`GET /v1/models`,
`POST /v1/embeddings`, `DELETE /v1/requests/{id}`) to the server, so the browser
sees one origin. The server is not modified.

```bash
python3 apps/embedding-apps/serve.py --server http://127.0.0.1:8000 --port 8765
# open http://127.0.0.1:8765/media-finder/
```

## Apps

- `media-finder/`: index a folder of images, audio and video, then search by words,
  by an example file, or both. Vectors and thumbnails persist in IndexedDB, keyed
  by model, dimensions and index prompt, so a different configuration never mixes
  with another. Re-choosing the folder resumes and skips files already indexed.
- `embedding-lab/`: a retrieval-quality probe. Choose a folder holding `set.json`
  (cases of query, positives, negatives; items are text, a file, or both) plus
  any media it names, add one or more configurations (model, dimensions, query
  prompt, document prompt), and run. It reports Recall@1, Recall@3, MRR and the
  margin of the best right answer over the best wrong one, per configuration and
  per case, so MLX against GGUF, 256 against 768 dimensions, and prompts on or
  off sit side by side. Every case ranks against the pool of all cases'
  documents; a pool item identical to the case's own query is left out of that
  case. `embedding-lab/sample/` is a ready test set. Pure metrics test:
  `node --test apps/embedding-apps/embedding-lab/metrics.test.mjs`.

## Shared kit

`kit/embed-kit.js` holds what every app needs because the server has no index:
batch planning (16 items, 48 MB per request), retry on a busy server, cancel via
`DELETE /v1/requests/{id}`, the IndexedDB store, a flat cosine `topK`, thumbnails.

## Notes

- Video is embedded as its visual timeline only (the server's contract).
- Files over 32 MB are skipped and listed. A file the server rejects is skipped
  alone; the rest of its batch is retried one by one.
- Indexes from different engines or quants are not interchangeable; change the
  model or dimensions and you get a fresh, separate index.
