# RLM (Recursive Language Model) Guide

The RLM endpoint lets a language model write and execute Python code to work through a problem iteratively. Instead of trying to answer in one shot, the model gets a persistent REPL where it can explore your data, call itself on sub-problems, and build up an answer step by step.

Based on the paper by Zhang, Kraska, and Khattab (arxiv 2512.24601v2), which showed 2x+ improvement over vanilla LLMs on long-context tasks.

## When to use RLM

- **Long documents**: The model can slice, search, and summarize sections instead of cramming everything into one prompt
- **Multi-step reasoning**: Count things, extract structured data, cross-reference sections
- **Data processing**: Parse, filter, aggregate -- anything you'd write a script for
- **Tasks where "show your work" matters**: The trace gives you full visibility into what the model did

When NOT to use it: simple Q&A, creative writing, short conversations. If the answer fits in one generation, use `/v1/chat/completions` instead.

## Quick start

```bash
curl -X POST http://localhost:8080/v1/rlm/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL_ID",
    "context": "Alice has 3 cats. Bob has 5 dogs. Carol has 2 fish.",
    "query": "How many animals does each person have, and what is the total?"
  }'
```

The model explores the data across multiple iterations (this is a real response):

```
Iteration 1: print(context)         -> sees the raw text
Iteration 2: regex extraction       -> parses names/counts, prints results
Iteration 3: FINAL("Alice: 3...")   -> formats and returns answer
```

## Use cases that actually matter

The quick start above works, but it's a toy example -- a regular chat call handles that fine. RLM's value shows up on tasks like these:

### Analyzing a large codebase or log file

```bash
# Feed in a big log file and ask a question that requires searching
curl -X POST http://localhost:8080/v1/rlm/completions \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import orjson
print(orjson.dumps({
    'model': 'YOUR_MODEL_ID',
    'context': open('server.log').read(),
    'query': 'Find all error patterns, group by type, and identify the root cause of the most frequent one',
    'max_iterations': 15
}).decode())
")"
```

The model will typically: check length and structure, search for ERROR/Exception lines, group them, count frequencies, then drill into the most common one. A single-shot LLM would struggle with a 50K-line log -- it can't search, filter, or count reliably in one pass.

### Structured data extraction from unstructured text

```json
{
  "model": "YOUR_MODEL_ID",
  "context": "...a long contract or legal document...",
  "query": "Extract all monetary amounts, the parties involved, key dates, and any penalty clauses. Return as JSON.",
  "system": "Return your final answer as valid JSON with keys: parties, amounts, dates, penalties",
  "max_iterations": 15
}
```

The model can scan sections, build up partial results, cross-reference, and assemble the final JSON. The `system` field shapes the output format.

### Cross-referencing multiple sections of a document

```json
{
  "model": "YOUR_MODEL_ID",
  "context": "...a research paper or technical report...",
  "query": "Does the methodology section support the claims made in the conclusion? List any unsupported claims.",
  "max_iterations": 20
}
```

The model will typically find the methodology and conclusion sections by searching, extract claims from each, then compare. This is the kind of task where vanilla LLMs hallucinate -- they "remember" the conclusion but can't actually re-read the methodology to verify.

### Comparing two datasets embedded in one context

```bash
# Combine two CSVs into one context
CONTEXT="=== Q1 Sales ===
$(cat q1_sales.csv)

=== Q2 Sales ===
$(cat q2_sales.csv)"

curl -X POST http://localhost:8080/v1/rlm/completions \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import orjson
context = '''$CONTEXT'''
print(orjson.dumps({
    'model': 'YOUR_MODEL_ID',
    'context': context,
    'query': 'Compare Q1 and Q2. Which products grew? Which declined? What is the overall revenue change?',
    'max_iterations': 10
}).decode())
")"
```

The model can parse both CSVs, compute diffs per product, and aggregate -- tasks that require actual arithmetic, not just pattern matching.

### Using sub-calls for divide-and-conquer

```json
{
  "model": "YOUR_MODEL_ID",
  "context": "...a 200-page book or very long document...",
  "query": "Write a chapter-by-chapter summary",
  "sub_model": "YOUR_MODEL_ID",
  "sub_max_tokens": 512,
  "sub_temperature": 0.3,
  "max_iterations": 30
}
```

The main model slices the text into chapters, then calls `llm_query(f"Summarize this chapter: {chapter_text}")` for each one. Each sub-call is a separate generation with its own token budget. The main model aggregates the summaries at the end.

## How it works

1. Your `context` is loaded into a Python variable called `context`
2. The model receives your `query` as its task
3. Each iteration, the model writes code in a ` ```repl ` block
4. The code runs in a sandboxed namespace; stdout/stderr are fed back as the next message
5. The model calls `FINAL("answer")` or `FINAL_VAR("variable_name")` when done
6. If it responds without a code block, that text becomes the answer directly

The model also has `llm_query("sub-question")` -- it can call itself on smaller pieces of the context.

## Request fields

```json
{
  "model": "Qwen3-4B",
  "context": "...your long text...",
  "query": "What should the model figure out?",

  "system": null,
  "max_iterations": 10,
  "max_tokens": 2048,
  "temperature": null,
  "top_p": null,
  "stream": false,
  "sandbox": true,
  "timeout": 30,
  "max_output_chars": 10000,
  "sub_model": null,
  "sub_max_tokens": null,
  "sub_temperature": null,
  "sub_top_p": null,
  "enable_thinking": null
}
```

| Field | Default | What it does |
|-------|---------|-------------|
| `model` | required | Which model to use |
| `context` | required | The data to process (loaded as a string variable) |
| `query` | required | The task/question |
| `system` | null | Extra system instructions appended to the RLM prompt |
| `max_iterations` | 10 | How many code/execute cycles before giving up (1-50) |
| `max_tokens` | 2048 | Generation limit per iteration |
| `stream` | false | SSE streaming of iteration events |
| `sandbox` | true | Restrict builtins, block dangerous attribute access |
| `timeout` | 30 | Seconds before killing a code execution (1-120) |
| `max_output_chars` | 10000 | Truncate stdout/stderr per iteration |
| `sub_model` | null | Use a different model for `llm_query()` calls |
| `sub_max_tokens` | null | Override max_tokens for sub-calls |
| `sub_temperature` | null | Override temperature for sub-calls |
| `sub_top_p` | null | Override top_p for sub-calls |
| `enable_thinking` | null | Enable thinking mode (Qwen3 models) |

## Response

```json
{
  "id": "rlm-a1b2c3d4e5f6",
  "object": "rlm.completion",
  "created": 1700000000,
  "model": "Qwen3-4B",
  "answer": "The final answer text",
  "finish_reason": "final",
  "usage": {
    "prompt_tokens": 500,
    "completion_tokens": 200,
    "total_tokens": 700
  },
  "rlm": {
    "iterations": 3,
    "sub_queries": 1,
    "trace": [
      {"iteration": 1, "code_len": 45, "stdout_len": 120, "stderr_len": 0, "action": null},
      {"iteration": 2, "code_len": 80, "stdout_len": 50, "stderr_len": 0, "action": null},
      {"iteration": 3, "code_len": 30, "stdout_len": 0, "stderr_len": 0, "action": "FINAL"}
    ]
  }
}
```

**finish_reason values:**
- `"final"` -- Model called `FINAL()` or `FINAL_VAR()`
- `"direct_response"` -- Model responded without a code block
- `"max_iterations"` -- Hit the iteration limit

The `trace` shows what happened each iteration: how much code was written, how much output it produced, and whether it terminated.

## Streaming

Set `"stream": true` to get Server-Sent Events as the model works:

```bash
curl -N -X POST http://localhost:8080/v1/rlm/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL_ID",
    "context": "...long document...",
    "query": "Summarize the key findings",
    "stream": true
  }'
```

Events:

```
event: rlm_start
data: {"id": "rlm-xxx", "model": "...", "context_length": 12345}

event: iteration_start
data: {"iteration": 1}

event: assistant_response
data: {"iteration": 1, "text": "Let me start by checking the length...\n```repl\nprint(len(context))\n```"}

event: code_output
data: {"iteration": 1, "stdout": "12345\n", "stderr": "", "code_len": 20}

event: iteration_start
data: {"iteration": 2}

...

event: rlm_complete
data: {"answer": "...", "finish_reason": "final", "usage": {...}, "rlm": {...}}
```

## Using sub-calls

The model can call `llm_query("question")` inside its code to ask itself (or a different model) a sub-question. This is useful for reasoning over chunks:

```json
{
  "model": "Qwen3-30B",
  "context": "...a very long document...",
  "query": "What are the three most important claims?",
  "sub_model": "Qwen3-4B",
  "sub_max_tokens": 512,
  "sub_temperature": 0.0
}
```

Here the main model (30B) orchestrates, but delegates sub-questions to a smaller model (4B) for speed. The sub-call count appears in `rlm.sub_queries`.

## Sandbox

By default, code runs in a restricted environment:

**Allowed**: `len`, `print`, `range`, `sorted`, `map`, `filter`, `enumerate`, `zip`, `str`, `int`, `float`, `list`, `dict`, `set`, `tuple`, `type`, `isinstance`, `hasattr`, `getattr`, `max`, `min`, `sum`, `abs`, `round`, `reversed`, `any`, `all`, `bool`, `bytes`, `chr`, `ord`, `hex`, `oct`, `bin`, `format`, `repr`, `hash`, `id`, `iter`, `next`, `pow`, `slice`, `frozenset`, `object`

**Blocked**: `open`, `__import__`, `exec`, `eval`, `compile` (builtins level), plus AST-level blocking of `__class__`, `__bases__`, `__subclasses__`, `__globals__`, `__code__`, and other dunder attributes that enable sandbox escapes.

Each code execution has a timeout (default 30s). Output is truncated at `max_output_chars`.

Set `"sandbox": false` to disable restrictions (full Python access). Only do this if you trust the model's output or are running in an isolated environment.

## Python client example

```python
import httpx

response = httpx.post("http://localhost:8080/v1/rlm/completions", json={
    "model": "Qwen3-4B",
    "context": open("big_document.txt").read(),
    "query": "Extract all dates mentioned and list them chronologically",
    "max_iterations": 15,
})

result = response.json()
print(result["answer"])
print(f"Took {result['rlm']['iterations']} iterations, "
      f"{result['rlm']['sub_queries']} sub-queries")
```

Streaming with SSE:

```python
import httpx

with httpx.stream("POST", "http://localhost:8080/v1/rlm/completions", json={
    "model": "Qwen3-4B",
    "context": "...data...",
    "query": "Analyze this data",
    "stream": True,
}) as response:
    for line in response.iter_lines():
        if line.startswith("event:"):
            event_type = line.split(": ", 1)[1]
        elif line.startswith("data:"):
            import orjson
            data = orjson.loads(line.split(": ", 1)[1])
            if event_type == "code_output":
                print(f"[iter {data['iteration']}] {data['stdout']}")
            elif event_type == "rlm_complete":
                print(f"\nAnswer: {data['answer']}")
```

## Tips

- **Start with low `max_iterations`** (3-5) while testing. Bump to 10-20 for real tasks. The model often converges in 3-5 iterations but complex tasks need room to explore.
- **Use `system` for output format**: "Return your final answer as JSON", "Use markdown tables", "Be concise". This shapes what `FINAL()` produces.
- **Bigger context = bigger win**: RLM's advantage over vanilla chat scales with context size. For short texts (<1K chars), just use `/v1/chat/completions`. For 10K+ chars, RLM starts to dominate.
- **Small models work**: The REPL loop compensates for smaller context windows and weaker reasoning. A 4B model with 10 iterations often beats a single-shot 30B because it can verify its work.
- **Check the trace**: If `iterations` equals `max_iterations`, the model ran out of budget. Look at the trace to see if it was making progress or stuck in a loop. Bump `max_iterations` or rephrase the query.
- **Sub-calls are expensive**: Each `llm_query()` is a full generation call. Set `sub_max_tokens` low (256-512) to keep them fast. Use `sub_temperature: 0.0` for deterministic sub-answers.
- **Sandbox doesn't have `re` or `json`**: The sandbox blocks `import`. The model has to work with string methods, list comprehensions, and builtins only. This is usually enough. If you need imports, set `"sandbox": false`.
- **The model pins itself**: During an RLM run, the model is pinned in the LRU cache so it can't be evicted by another request. Long-running RLM jobs won't lose their model mid-execution.

## Next steps

See [rlm_advanced.md](./rlm_advanced.md) for composable patterns: pipelines, fan-out/fan-in, structured extraction with validation, streaming progress monitors, retry with escalation, and batch processing.
