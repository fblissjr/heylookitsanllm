#!/usr/bin/env python3
"""
Example usage of the new heylookllm command with API selection
"""

print("HeyLookLLM API Selection Examples")
print("=" * 50)

print("\nAvailable Options:")
print("  --api openai   : Only OpenAI-compatible endpoints")
print("  --api ollama   : Only Ollama-compatible endpoints")
print("  --api both     : Both APIs (default)")
print("  --port 11434   : Ollama standard port (default)")
print("  --port 8080    : OpenAI alternative port")

print("\nExample Commands:")

print("\n1. Ollama-only server (default port 11434):")
print("   heylookllm --api ollama")
print("   # Endpoints: /api/tags, /api/chat, /api/generate")

print("\n2. OpenAI-only server (custom port):")
print("   heylookllm --api openai --port 8080")
print("   # Endpoints: /v1/models, /v1/chat/completions")

print("\n3. Both APIs (default):")
print("   heylookllm --api both")
print("   # OpenAI: /v1/models, /v1/chat/completions")
print("   # Ollama: /api/tags, /api/chat, /api/generate")

print("\n4. Public server with both APIs:")
print("   heylookllm --host 0.0.0.0 --port 11434 --api both")

print("\n5. Debug mode:")
print("   heylookllm --log-level DEBUG --api both")

print("\nTesting Commands:")

print("\nTest Ollama API:")
print("   curl -X POST http://localhost:11434/api/chat \\")
print("     -H 'Content-Type: application/json' \\")
print("     -d '{\"model\": \"gemma3n-e4b-it\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")

print("\nTest OpenAI API:")
print("   curl -X POST http://localhost:11434/v1/chat/completions \\")
print("     -H 'Content-Type: application/json' \\")
print("     -d '{\"model\": \"gemma3n-e4b-it\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")

print("\nRun Integration Tests:")
print("   python tests/test_ollama.py")

print("\nBenefits:")
print("  - Drop-in replacement for Ollama server")
print("  - OpenAI API compatibility")
print("  - Vision support with base64 images")
print("  - Single server, no port conflicts")
print("  - Extensible middleware architecture")