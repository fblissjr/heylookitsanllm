#!/usr/bin/env python3
"""Test keepalive functionality during long prompt processing."""

import asyncio
import aiohttp
import json
import time

async def test_keepalive():
    """Test that keepalive messages are sent during long prompts."""
    
    # Create a long prompt that will trigger prompt processing
    long_context = "This is a test. " * 500  # Create a long context
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {long_context}\n\nQuestion: What is 2+2?"}
    ]
    
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8080/v1/chat/completions"
        payload = {
            "model": "dolphin-mistral",  # Change to your model
            "messages": messages,
            "stream": True,
            "max_tokens": 50
        }
        
        print("Sending request with long prompt...")
        start_time = time.time()
        
        async with session.post(url, json=payload) as response:
            print(f"Response status: {response.status}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
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