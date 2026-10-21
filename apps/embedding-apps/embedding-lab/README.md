# Embedding Lab

A retrieval-quality probe for heylook's embeddings endpoint. You describe what
should match what; the Lab measures how well each configuration (model, engine,
dimension size, task prompts) gets it right, side by side.

## Run

```bash
python3 apps/embedding-apps/serve.py --server http://127.0.0.1:8000
# open http://127.0.0.1:8765/embedding-lab/
```

Try it on `sample/`: choose that folder as the test set.

## Test set

A folder containing `set.json` plus any media files it names (paths relative to
the folder):

```json
{
  "name": "optional label",
  "cases": [
    {
      "id": "optional-unique-name",
      "query": "text" ,
      "positives": ["text", { "file": "a.png" }, { "text": "caption", "file": "a.png" }],
      "negatives": ["text that looks relevant but is wrong"]
    }
  ]
}
```

- An item is a string (text), or an object with `text`, `file`, or both. A query
  may also be a file, for example an image query with a text answer.
- Every case needs at least one positive. Negatives are optional but are what
  make a case hard; write ones that look plausible.
- Unknown fields, missing files, duplicate ids and a query identical to its own
  positive are rejected with a message naming the place.
- All documents from all cases form one pool. A case ranks its query against the
  whole pool, so it also competes with other cases' documents. A pool item
  identical to the case's own query is left out of that case, since a query
  always matches its own copy.

## Reading the results

- **Recall@1 / Recall@3:** the share of cases whose best right answer ranks first
  or within the top three.
- **MRR:** the mean of 1 / rank of the best right answer.
- **Margin:** the best right answer's score minus the best wrong one's. Positive
  means the case passed with room to spare; near zero is a near-tie; negative is a
  miss. **Worst margin** is the weakest case.
- **Per-case table:** `✓ #1 +0.083` or `✗ #3 −0.050`, and for a miss the document
  that beat it.
- Scores are cosine similarity. Underlined values are the best in their column.
- Few cases give coarse numbers. Eight cases move in steps of 12.5 points, so
  treat small gaps as noise and look at which cases flip.

## Configurations

Each configuration is one model, dimension size, query prompt and document prompt
(prompt names come from the model's own catalog). Run several and compare, for
example MLX against GGUF, 256 against 768 dimensions, or prompts on against off.
Configurations run grouped by model, so a model load or an engine switch happens
once. Results can be downloaded as JSON.

A run is atomic per configuration: if the server rejects any request, that
configuration stops with the error. Stop cancels the in-flight request.

## Files

- `metrics.js`: scoring, summaries and set validation (no DOM, no network).
- `metrics.test.mjs`: `node --test apps/embedding-apps/embedding-lab/metrics.test.mjs`
- `app.js`, `index.html`, `lab.css`: the page.
- `sample/`: a ready test set with four generated shapes.
