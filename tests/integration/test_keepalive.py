#!/usr/bin/env python3
"""Watch the SSE keepalive comments the server emits during a long prefill.

Manual script against a RUNNING server (like the rest of tests/integration) --
it prints a timeline, it does not assert. The thing it exercises is
streaming_utils' 5s `: keepalive` cadence before the first token, which no
unit test can see.

httpx, not aiohttp: aiohttp was the one undeclared dependency in the repo, so
this file could not run after a clean `uv sync`, while httpx is already a dev
dep (the contract tests' TestClient rides on it).
"""

import asyncio
import json
import os
import sys
import time

import httpx

BASE_URL = os.environ.get("HEYLOOK_URL", "http://localhost:1263")


async def _resolve_model(client: httpx.AsyncClient) -> str:
    """The model to probe: $HEYLOOK_MODEL, argv[1], else the server's first.

    This used to be a hardcoded "dolphin-mistral" with a "change to your
    model" comment. That id stopped existing long ago, so the script 404'd
    before it could ever reach the keepalive path it exists to watch -- asking
    the server is the version that cannot go stale.
    """
    named = os.environ.get("HEYLOOK_MODEL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if named:
        return named
    res = await client.get(f"{BASE_URL}/v1/models")
    res.raise_for_status()
    models = res.json().get("data") or []
    if not models:
        sys.exit("No models are enabled on the server -- nothing to probe.")
    return models[0]["id"]


async def test_keepalive():
    """Test that keepalive messages are sent during long prompts."""

    # Create a long prompt that will trigger prompt processing
    long_context = "This is a test. " * 500  # Create a long context

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {long_context}\n\nQuestion: What is 2+2?"}
    ]

    # No read timeout: a long prefill is exactly the silence being measured.
    async with httpx.AsyncClient(timeout=None) as client:
        model = await _resolve_model(client)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": 50
        }

        print(f"Sending request with long prompt (model: {model})...")
        start_time = time.time()

        async with client.stream("POST", f"{BASE_URL}/v1/chat/completions", json=payload) as response:
            print(f"Response status: {response.status_code}")

            # aiter_lines() yields str and splits on line boundaries, so the
            # decode + the byte-level line handling aiohttp needed are gone.
            async for line in response.aiter_lines():
                line = line.strip()

                if line.startswith(':'):
                    # This is a keepalive comment
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.2f}s] Keepalive: {line}")

                elif line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        print("Stream complete")
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and chunk['choices']:
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                print(f"Token: {content}", end='', flush=True)
                    except json.JSONDecodeError:
                        pass

        print(f"\nTotal time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    print("Testing keepalive functionality...")
    print("Make sure the server is running: heylookllm --api openai --log-level DEBUG")
    print()
    asyncio.run(test_keepalive())
