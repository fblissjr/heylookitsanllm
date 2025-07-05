#!/bin/bash

# MLX Provider Optimization Test Runner
# Runs comprehensive test suite for MLX optimizations

set -e

echo "🚀 MLX Provider Optimization Test Suite"
echo "======================================="

# Check if we're in the right directory
if [ ! -f "models.yaml" ]; then
    echo "❌ Please run from the project root directory"
    exit 1
fi

# Check if server is running
echo "🔍 Checking server status..."
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Server is running on port 8080"
    SERVER_RUNNING=true
else
    echo "⚠️  Server not running on port 8080"
    echo "   Start server: python -m heylook_llm.server --port 8080"
    SERVER_RUNNING=false
fi

echo

# Run tests based on server status
if [ "$SERVER_RUNNING" = true ]; then
    echo "🧪 Running complete test suite..."
    python tests/test_runner.py
else
    echo "🧪 Running unit tests only..."
    echo "Unit Tests:"
    python tests/test_optimizations_unit.py
    echo
    echo "Phase 2 Features:"
    python tests/test_phase2_features.py
fi

echo
echo "📋 Test execution completed"
