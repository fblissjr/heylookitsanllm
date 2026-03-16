# RLM Advanced Patterns

Composable patterns for building real workflows on top of the RLM endpoint. Each pattern can be combined with others.

See [rlm_guide.md](./rlm_guide.md) for basics.

## Pattern: Pipeline (chain RLM calls)

Feed the output of one RLM call as context to the next. Each stage does one thing well.

```python
import httpx
import orjson

BASE = "http://localhost:8080/v1/rlm/completions"
MODEL = "YOUR_MODEL_ID"

def rlm(context: str, query: str, **kwargs) -> dict:
    resp = httpx.post(BASE, json={"model": MODEL, "context": context, "query": query, **kwargs})
    return resp.json()

# Stage 1: Extract structure
raw = open("messy_report.txt").read()
stage1 = rlm(raw, "Extract all sections with their headers. Return as a numbered list.")

# Stage 2: Summarize each section (using stage 1's answer as new context)
stage2 = rlm(
    stage1["answer"],
    "For each section, write a one-sentence summary.",
    system="Return as a markdown list"
)

# Stage 3: Synthesize
stage3 = rlm(
    stage2["answer"],
    "Based on these section summaries, what are the 3 most important takeaways?",
    max_iterations=5
)

print(stage3["answer"])
```

Why pipeline instead of one big call: each stage has a focused context and a clear task. The model doesn't need to juggle extraction + summarization + synthesis simultaneously.

## Pattern: Fan-out / Fan-in

Split a large context into chunks, process each independently, then merge results.

```python
import httpx
import orjson

BASE = "http://localhost:8080/v1/rlm/completions"
MODEL = "YOUR_MODEL_ID"

def rlm(context: str, query: str, **kwargs) -> dict:
    resp = httpx.post(BASE, json={"model": MODEL, "context": context, "query": query, **kwargs})
    return resp.json()

# Split a book into chapters (or any logical chunks)
book = open("long_book.txt").read()

# Fan-out: process each chunk
# Let RLM itself do the splitting and per-chunk work
split_result = rlm(
    book,
    "Split this text into chapters or major sections. "
    "For each section, call llm_query() with a prompt asking for a 2-sentence summary. "
    "Collect all summaries into a list.",
    max_iterations=25,
    sub_max_tokens=256,
    sub_temperature=0.0,
)

# Fan-in: synthesize the summaries
final = rlm(
    split_result["answer"],
    "These are chapter summaries of a book. Write a cohesive overall summary.",
    max_iterations=5,
)

print(final["answer"])
print(f"Used {split_result['rlm']['sub_queries']} sub-calls for chunked summarization")
```

## Pattern: Structured extraction with validation

Extract structured data, then validate it in a second pass.

```python
import httpx
import orjson

BASE = "http://localhost:8080/v1/rlm/completions"
MODEL = "YOUR_MODEL_ID"

def rlm(context: str, query: str, **kwargs) -> dict:
    resp = httpx.post(BASE, json={"model": MODEL, "context": context, "query": query, **kwargs})
    return resp.json()

contract_text = open("contract.pdf.txt").read()

# Extract
extraction = rlm(
    contract_text,
    "Extract all named parties, monetary amounts with currency, dates, and obligations. "
    "Return as JSON.",
    system='Return valid JSON: {"parties": [...], "amounts": [...], "dates": [...], "obligations": [...]}',
    max_iterations=15,
)

# Validate by feeding the extraction back alongside the original
validation = rlm(
    f"=== ORIGINAL DOCUMENT ===\n{contract_text}\n\n=== EXTRACTED DATA ===\n{extraction['answer']}",
    "Compare the extracted data against the original document. "
    "For each extracted item, verify it appears in the original. "
    "Flag any items that are hallucinated or incorrectly extracted. "
    "Flag any items in the original that were missed.",
    max_iterations=15,
)

print("Extraction:", extraction["answer"])
print("Validation:", validation["answer"])
```

This catches hallucinations by giving the model a chance to cross-reference its own output against the source.

## Pattern: Comparative analysis

Load two (or more) documents into one context and compare them.

```python
import httpx
import orjson

BASE = "http://localhost:8080/v1/rlm/completions"
MODEL = "YOUR_MODEL_ID"

def rlm(context: str, query: str, **kwargs) -> dict:
    resp = httpx.post(BASE, json={"model": MODEL, "context": context, "query": query, **kwargs})
    return resp.json()

doc_a = open("proposal_v1.md").read()
doc_b = open("proposal_v2.md").read()

result = rlm(
    f"=== VERSION 1 ===\n{doc_a}\n\n=== VERSION 2 ===\n{doc_b}",
    "Compare these two versions. What was added? What was removed? What was changed? "
    "Focus on substantive changes, not formatting.",
    max_iterations=15,
)

print(result["answer"])
```

Works well for: comparing drafts, diffing specs, reviewing policy changes, analyzing competing proposals.

## Pattern: Streaming progress monitor

Watch the model work in real time and collect iteration data.

```python
import httpx
import orjson

BASE = "http://localhost:8080/v1/rlm/completions"
MODEL = "YOUR_MODEL_ID"

context = open("dataset.csv").read()

iterations = []
with httpx.stream("POST", BASE, json={
    "model": MODEL,
    "context": context,
    "query": "Compute descriptive statistics for each numeric column and identify outliers",
    "stream": True,
    "max_iterations": 15,
}) as resp:
    event_type = None
    for line in resp.iter_lines():
        if line.startswith("event:"):
            event_type = line.split(": ", 1)[1]
        elif line.startswith("data:"):
            data = orjson.loads(line.split(": ", 1)[1])

            if event_type == "iteration_start":
                print(f"\n--- Iteration {data['iteration']} ---")

            elif event_type == "assistant_response":
                # Show first 200 chars of what the model is thinking
                preview = data["text"][:200]
                print(f"Model: {preview}...")

            elif event_type == "code_output":
                iterations.append(data)
                if data["stderr"]:
                    print(f"ERROR: {data['stderr'][:200]}")
                else:
                    print(f"Output ({data['stdout_len']} chars): {data['stdout'][:200]}")

            elif event_type == "rlm_complete":
                print(f"\nDone in {data['rlm']['iterations']} iterations")
                print(f"Answer: {data['answer'][:500]}")
```

## Pattern: Retry with escalation

Start with a fast/small model. If it hits max iterations, retry with a bigger one.

```python
import httpx
import orjson

BASE = "http://localhost:8080/v1/rlm/completions"

def rlm(context: str, query: str, model: str, **kwargs) -> dict:
    resp = httpx.post(BASE, json={"model": model, "context": context, "query": query, **kwargs})
    return resp.json()

context = open("complex_data.txt").read()
query = "Identify all causal relationships described in this text and draw a dependency graph"

# Try fast model first
result = rlm(context, query, model="Qwen3-4B", max_iterations=10)

if result["finish_reason"] in ("max_iterations", "error_threshold"):
    print(f"Small model failed ({result['finish_reason']}), escalating...")
    result = rlm(context, query, model="Qwen3-30B", max_iterations=20)

print(result["answer"])
print(f"Finished with: {result['finish_reason']}")
```

## Pattern: Batch RLM (process many documents)

Run RLM on a collection of files, accumulating results.

```python
import httpx
import orjson
from pathlib import Path

BASE = "http://localhost:8080/v1/rlm/completions"
MODEL = "YOUR_MODEL_ID"

def rlm(context: str, query: str, **kwargs) -> dict:
    resp = httpx.post(BASE, json={"model": MODEL, "context": context, "query": query, **kwargs}, timeout=120)
    return resp.json()

# Process all txt files in a directory
results = {}
for path in sorted(Path("papers/").glob("*.txt")):
    print(f"Processing {path.name}...")
    result = rlm(
        path.read_text(),
        "Extract: title, authors, main contribution, methodology, key results. Return as JSON.",
        system='Return valid JSON with keys: title, authors, contribution, methodology, results',
        max_iterations=10,
    )
    results[path.name] = {
        "answer": result["answer"],
        "iterations": result["rlm"]["iterations"],
        "finish_reason": result["finish_reason"],
    }
    print(f"  -> {result['finish_reason']} in {result['rlm']['iterations']} iterations")

# Save collected results
Path("extracted.json").write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
```

## Pattern: Self-correcting with error feedback

The RLM loop naturally handles errors -- if code throws an exception, the error is fed back and the model fixes its approach. You can lean into this by writing queries that encourage exploration:

```json
{
  "model": "YOUR_MODEL_ID",
  "context": "...some data in an unknown format...",
  "query": "Figure out the format of this data, then extract all records. Try different parsing approaches if one fails.",
  "max_iterations": 15
}
```

The model might try CSV parsing, fail, try JSON, fail, try custom delimiters, succeed. Each error gives it information about the actual format. This is fundamentally different from single-shot prompting where you have to know the format upfront.

## Pattern: Long-running with compaction

For tasks that need many iterations (20+), enable compaction to prevent hitting the context window.

```python
result = rlm(
    open("huge_dataset.csv").read(),
    "Clean this data: find and fix inconsistencies, remove duplicates, "
    "standardize formats. Report every change you make.",
    max_iterations=40,
    compaction=True,
    compaction_threshold=0.8,
    max_context_tokens=32768,  # Match your model's actual context window
    max_errors=5,              # Don't waste iterations on repeated failures
)

print(f"Compacted {result['rlm']['compactions']} times")
```

When history exceeds 80% of the context window, the engine asks the model to summarize its progress, then replaces the full history with the summary. The REPL namespace (all variables) is preserved -- only the chat messages get compacted. This lets a 4B model work through 40+ iterations without losing early context.

## Pattern: Recursive decomposition with `rlm_query()`

Use `max_depth=2+` to let the model spawn child RLMs for complex sub-problems. Unlike `llm_query()` (single-shot), `rlm_query()` gives the child its own REPL loop.

```python
result = rlm(
    open("annual_report.txt").read(),
    "For each section of this report, write a detailed analysis. "
    "Use rlm_query() for sections that need multi-step investigation "
    "(e.g., financial data that requires computation). "
    "Use llm_query() for sections that just need summarization.",
    max_depth=2,
    max_iterations=25,
    sub_max_tokens=1024,
)

print(result["answer"])
print(f"Child RLMs spawned: {len(result['rlm']['child_traces'] or [])}")
for i, child in enumerate(result["rlm"]["child_traces"] or []):
    print(f"  Child {i+1}: {child['iterations']} iterations, {child['sub_queries']} sub-queries")
```

`max_depth=2` means the parent can spawn children, but children can only use `llm_query()`. Use `max_depth=3` sparingly -- it allows grandchildren, which can get expensive.

## Pattern: Error-bounded exploration

Use `max_errors` to stop the model from spinning on unsolvable sub-tasks, combined with retry-with-escalation.

```python
# Try with sandbox + error bound
result = rlm(
    data,
    "Parse this data and compute statistics",
    sandbox=True,
    max_errors=3,
    max_iterations=10,
)

if result["finish_reason"] == "error_threshold":
    # Model kept hitting sandbox restrictions -- retry without sandbox
    result = rlm(
        data,
        "Parse this data and compute statistics. You may use any Python library.",
        sandbox=False,
        max_errors=5,
        max_iterations=15,
    )
```

## Pattern: Batched sub-queries

Use `llm_query_batched()` inside the REPL to process multiple sub-questions in one call. When the backend provider supports GPU batching, all queries run in a single GPU pass.

```json
{
  "model": "YOUR_MODEL_ID",
  "context": "...a document with multiple sections...",
  "query": "Summarize each section in one sentence",
  "max_iterations": 10,
  "sub_max_tokens": 256,
  "sub_temperature": 0.0
}
```

The model might write:

```python
sections = context.split("\n\n")
summaries = llm_query_batched([f"Summarize: {s}" for s in sections])
for i, s in enumerate(summaries):
    print(f"Section {i+1}: {s}")
FINAL("\n".join(summaries))
```

This is faster than `for s in sections: llm_query(...)` because the backend can batch the GPU work. Falls back to sequential automatically if batching isn't available.

`rlm_query_batched()` works the same way but spawns child RLMs (requires `max_depth >= 2`). Each child runs sequentially since they need their own REPL loops.

## Pattern: Custom tools (programmatic use)

When using `RLMEngine` directly (not through the HTTP endpoint), you can inject custom Python functions as tools available in the REPL.

```python
from heylook_llm.rlm import RLMEngine, RLMRequest

def fetch_url(url: str) -> str:
    """Fetch a URL and return its text content."""
    import httpx
    return httpx.get(url).text

def query_db(sql: str) -> str:
    """Run a SQL query and return results as text."""
    # ... your db logic
    return results

engine = RLMEngine(
    router,
    custom_tools=[
        fetch_url,
        {"tool": query_db, "description": "Run a read-only SQL query"},
    ],
)

result = engine.run(RLMRequest(
    model="YOUR_MODEL_ID",
    context="Analyze data from these sources",
    query="Compare web prices with database prices",
    sandbox=False,  # Custom tools need full Python access
))
```

Custom tools appear in the system prompt and are available in child RLMs. Names cannot conflict with builtins (`FINAL`, `llm_query`, etc.). Tools bypass the sandbox since they're server-registered and trusted.

## Pattern: Event callbacks (programmatic use)

Monitor RLM execution with callbacks for logging, metrics, or progress tracking.

```python
engine = RLMEngine(
    router,
    on_iteration_start=lambda i: print(f"Starting iteration {i}"),
    on_iteration_complete=lambda i, dur: print(f"Iteration {i} took {dur:.2f}s"),
    on_subcall_start=lambda depth, preview: print(f"Sub-call at depth {depth}: {preview}"),
    on_subcall_complete=lambda depth, model, dur: print(f"Sub-call done: {model} in {dur:.2f}s"),
)
```

Callback exceptions are caught and swallowed -- they never crash the RLM loop.

## Combining patterns

These patterns compose. A real workflow might:

1. **Fan-out** a large codebase into files
2. **Pipeline** each file through extraction -> analysis
3. **Fan-in** the per-file results into a summary
4. **Validate** the summary against the original
5. **Retry with escalation** if validation fails

```python
# Pseudocode for a full pipeline
files = split_codebase_into_files(repo_path)

# Fan-out: analyze each file
per_file = []
for f in files:
    result = rlm(f.read_text(), "List all functions, their purposes, and any bugs you spot")
    per_file.append(result["answer"])

# Fan-in: synthesize
combined = "\n\n".join(f"=== {f.name} ===\n{analysis}" for f, analysis in zip(files, per_file))
summary = rlm(combined, "Write a codebase health report: architecture patterns, common issues, recommendations")

# Validate
validation = rlm(
    f"=== REPORT ===\n{summary['answer']}\n\n=== FILE ANALYSES ===\n{combined}",
    "Check this report against the per-file analyses. Flag any claims not supported by the data.",
)
```
