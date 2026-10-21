# Media Finder

Index a folder of images, audio and video, then search it by words, by an
example file, or both. Everything stays in the browser; only the files you index
are sent to your heylook server.

## Run

```bash
python3 apps/embedding-apps/serve.py --server http://127.0.0.1:8000
# open http://127.0.0.1:8765/media-finder/
```

## Use

1. Pick a model and dimensions. The index prompt and query prompt default to the
   catalog's `document` and `query` entries when the model has them; both are
   changeable.
2. Choose a folder (or individual files). Files are embedded in bounded batches
   and saved as they finish.
3. Search with words, an example image, audio or video file, or both together.
   Results are cosine similarity, not probabilities. Click a result to open the
   original while this tab still holds the file.

## What to know

- **Index identity.** Vectors are stored in IndexedDB keyed by model, dimensions
  and index prompt. Changing any of them starts a separate index; they never mix.
  "Clear this index" deletes only the current configuration.
- **Resume.** Choosing the folder again skips files already indexed (matched by
  path, size and modified time). Stop, or a failure, loses nothing already saved.
- **Limits.** Files over 32 MB are skipped and listed. A file the server rejects
  is skipped alone: its batch is retried one file at a time.
- **Video** is embedded as its visual timeline only; its soundtrack is not
  included.
- **Reload.** The saved index keeps vectors, paths and thumbnails, not the
  originals, so after a reload a result opens only once its folder is chosen
  again.
- **Browsers.** Folder choice uses `webkitdirectory`, which Chrome, Safari and
  Firefox support.

## Files

- `app.js`: the page. `index.html`, `style.css`: its markup and style.
- `../kit/embed-kit.js`: batching, retry, cancel, the store and the search.
