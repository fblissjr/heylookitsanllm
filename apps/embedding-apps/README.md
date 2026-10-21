# embedding-apps

Small, self-contained apps on heylook's multimodal `POST /v1/embeddings`. Plain
static HTML and ES modules; no build step, no dependencies.

The server sends no CORS headers on purpose, so these pages cannot call it from
another origin. `serve.py` serves them and forwards three routes (`GET /v1/models`,
`POST /v1/embeddings`, `DELETE /v1/requests/{id}`) to the server, so the browser
sees one origin. The server is not modified.

```bash
python3 apps/embedding-apps/serve.py --server http://127.0.0.1:8000 --port 8765
# then open http://127.0.0.1:8765/<app>/
```

## Apps

Each has its own README with usage, limits and file layout.

- [`media-finder/`](./media-finder/README.md): index a folder of images, audio and
  video, then search it by words, an example file, or both.
- [`embedding-lab/`](./embedding-lab/README.md): measure retrieval quality across
  models, engines, dimension sizes and task prompts with a test set you write.

## Shared kit

`kit/embed-kit.js` holds what every app needs because the server has no index:
batch planning (16 items, 48 MB per request), retry on a busy server, cancel via
`DELETE /v1/requests/{id}`, the IndexedDB store, a flat cosine `topK`, thumbnails.

## Notes

- Video is embedded as its visual timeline only (the server's contract).
- Indexes from different engines or quants are not interchangeable.
- The proxy test is `tests/unit/test_embedding_apps_proxy.py`.
