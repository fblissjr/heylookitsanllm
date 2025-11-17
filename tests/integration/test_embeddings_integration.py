#!/usr/bin/env python3
"""
Integration test for embeddings API endpoint.
Starts the server and tests the endpoint.
"""

import subprocess
import time
import requests
import sys
import signal
import os

def test_embeddings_with_server():
    """Start server and test embeddings endpoint."""
    
    # Start the server in the background
    print("Starting heylookllm server...")
    server_process = subprocess.Popen(
        ["heylookllm", "--api", "openai", "--port", "8081"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    max_wait = 30  # seconds
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8081/v1/models")
            if response.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    
    if not server_ready:
        print("✗ Server failed to start within 30 seconds")
        server_process.terminate()
        return False
    
    print("✓ Server started successfully")
    
    # Test embeddings endpoint
    try:
        print("\nTesting /v1/embeddings endpoint...")
        
        # Test single embedding
        response = requests.post(
            "http://localhost:8081/v1/embeddings",
            json={
                "input": "Hello world",
                "model": "qwen2.5-coder-1.5b-instruct-4bit"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Embeddings endpoint works!")
            print(f"  Model: {data['model']}")
            print(f"  Dimensions: {len(data['data'][0]['embedding'])}")
            success = True
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Response: {response.text}")
            success = False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        success = False
    
    finally:
        # Stop the server
        print("\nStopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        print("✓ Server stopped")
    
    return success

def main():
    """Run the integration test."""
    print("=" * 60)
    print("Embeddings API Integration Test")
    print("=" * 60)
    
    success = test_embeddings_with_server()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Integration test passed!")
        return 0
    else:
        print("✗ Integration test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())