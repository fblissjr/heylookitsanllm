#!/usr/bin/env python3
"""
Simple test runner for Ollama API compatibility tests.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_ollama_tests():
    """Run the Ollama integration tests."""
    
    print("🚀 Running Ollama API Compatibility Tests")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run the integration tests
    cmd = [sys.executable, "tests/test_ollama_integration.py"]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_ollama.py")
        print("")
        print("Requirements:")
        print("  1. Start the server: heylookllm --host 0.0.0.0 --port 11434")
        print("  2. Run tests: python tests/test_ollama.py")
        print("")
        print("This will test:")
        print("  - Ollama /api/tags endpoint")
        print("  - Ollama /api/chat endpoint")
        print("  - Ollama /api/generate endpoint")
        print("  - Vision support")
        print("  - Parameter mapping")
        print("  - Error handling")
        return 0
    
    return run_ollama_tests()

if __name__ == "__main__":
    sys.exit(main())
