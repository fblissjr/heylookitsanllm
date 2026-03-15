# RLM (Recursive Language Model) Guide

The RLM endpoint lets a language model write and execute Python code to work through a problem iteratively. Instead of trying to answer in one shot, the model gets a persistent REPL where it can explore your data, call itself on sub-problems, and build up an answer step by step.

Based on the paper by Zhang, Kraska, and Khattab (arxiv 2512.24601v2), which showed 2x+ improvement over vanilla LLMs on long-context tasks.

## When to use RLM

- **Long documents**: The model can slice, search, and summarize sections instead of cramming everything into one prompt
- **Multi-step reasoning**: Count things, extract structured data, cross-reference sections
- **Data processing**: Parse, filter, aggregate -- anything you'd write a script for
- **Tasks where "show your work" matters**: The trace gives you full visibility into what the model did

When NOT to use it: simple Q&A, creative writing, conversations. Use `/v1/chat/completions` for those.

## Quick start

Start the server, then:

```bash
curl -X POST http://localhost:8080/v1/rlm/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL_ID",
    "context": "Alice has 3 cats. Bob has 5 dogs. Carol has 2 fish.",
    "query": "How many animals does each person have, and what is the total?"
  }'
```

The model will write code like:

```python
# Iteration 1: explore the data
lines = context.split(". ")
for line in lines:
    print(line)
```

Then after seeing the output, it might write:

```python
# Iteration 2: extract and compute
import re
counts = re.findall(r'(\w+) has (\d+) (\w+)', context)
total = sum(int(c[1]) for c in counts)
FINAL(f"Alice: 3 cats, Bob: 5 dogs, Carol: 2 fish. Total: {total} animals.")
```

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

- **Start with low `max_iterations`** (3-5) while testing. The model often converges fast.
- **Use `system` for format instructions**: "Return your answer as JSON" or "Use bullet points".
- **Small models work**: The REPL loop compensates for smaller context windows and weaker reasoning. A 4B model with 10 iterations often beats a single-shot 30B.
- **Check the trace**: If the model is looping without progress, the trace shows exactly where it got stuck.
- **Sub-calls are expensive**: Each `llm_query()` is a full generation. Use `sub_max_tokens` to keep them short.
