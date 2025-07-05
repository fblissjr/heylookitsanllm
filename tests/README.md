# MLX Provider Optimization Tests

This directory contains comprehensive tests for the MLX provider optimizations. All tests are designed to run with `heylookllm` server running on port 8080.

## 🚀 Quick Start

### Prerequisites
1. Start the heylookllm server:
   ```bash
   python -m heylook_llm.server --port 8080
   ```

2. Run all tests:
   ```bash
   python tests/test_runner.py
   ```

## 📁 Test Files

### Unit Tests (No Server Required)
- **`test_optimizations_unit.py`** - Component validation tests
- **`test_phase2_features.py`** - Phase 2 feature testing

### Integration Tests (Server Required)
- **`test_api_quick.py`** - Quick API validation
- **`test_api_integration.py`** - Comprehensive API testing

### Test Runner
- **`test_runner.py`** - Runs all tests in correct order
- **`test_server_connectivity.py`** - Server connectivity validation
- **`run_tests.sh`** - Shell script for easy test execution

## 🧪 Individual Test Usage

### Unit Tests
```bash
# Test core optimizations
python tests/test_optimizations_unit.py

# Test Phase 2 features
python tests/test_phase2_features.py
```

### API Tests (Requires Server)
```bash
# Quick API test
python tests/test_api_quick.py

# Full integration test
python tests/test_api_integration.py
```

### Complete Test Suite
```bash
# Run everything
python tests/test_runner.py

# Or use shell script
./tests/run_tests.sh
# or
bash tests/run_tests.sh
```

### Server Connectivity
```bash
# Test server connectivity
python tests/test_server_connectivity.py
```

## 🎯 What Each Test Validates

### Unit Tests (`test_optimizations_unit.py`)
- ✅ Import validation
- ✅ Config structure
- ✅ Performance monitor
- ✅ Optimization components
- ✅ Provider initialization

### Phase 2 Features (`test_phase2_features.py`)
- ✅ Enhanced VLM generation
- ✅ Advanced sampling integration
- ✅ Speculative decoding support
- ✅ Strategy enhancements
- ✅ Provider integration
- ✅ Backwards compatibility

### API Quick Test (`test_api_quick.py`)
- ✅ VLM text-only path optimization
- ✅ VLM vision path functionality
- ✅ Text-only model performance
- ✅ Path comparison analysis

### API Integration (`test_api_integration.py`)
- ✅ VLM path optimization validation
- ✅ Advanced sampling features
- ✅ Performance monitoring
- ✅ Real-world performance analysis

## 📊 Expected Results

### Performance Improvements
- **10-20% faster** text-only VLM requests
- **15-30% better** text quality
- **Feature parity** between paths
- **Comprehensive monitoring**

### Success Indicators
- VLM text-only path faster than vision path
- Performance metrics included in API responses
- All sampling parameters working
- Error handling functioning correctly

## 🔧 Running with pytest

You can also run tests using pytest:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_optimizations_unit.py

# Run with verbose output
pytest -v tests/

# Run with output capture disabled
pytest -s tests/
```

## 🛠️ Configuration

### Server Configuration
Tests expect the server to be running on:
- **Host**: `localhost`
- **Port**: `8080`
- **Health endpoint**: `/health`
- **API endpoint**: `/v1/chat/completions`

### Model Configuration
Tests use these models from your `models.yaml`:
- **VLM Model**: `gemma3n-e4b-it`
- **Text Model**: `llama-3.1-8b-instruct`

If these models aren't available, tests will use the first available model.

## 📈 Performance Monitoring

### API Response Format
Tests verify that API responses include performance metrics:
```json
{
  "choices": [...],
  "performance": {
    "prompt_tps": 150.5,
    "generation_tps": 45.2,
    "peak_memory_gb": 8.3
  }
}
```

### Performance Comparison
Tests compare response times between:
- VLM text-only path (optimized)
- VLM vision path (standard)
- Text-only model (baseline)

## 🔍 Troubleshooting

### Common Issues

**"Server not running"**
- Ensure server is started: `python -m heylook_llm.server --port 8080`
- Check port 8080 is not in use by another service
- Verify server health: `curl http://localhost:8080/health`

**"No models available"**
- Check `models.yaml` configuration
- Ensure model files exist in `modelzoo/`
- Verify models are enabled in configuration

**"Tests timeout"**
- Increase timeout values in test files
- Check server logs for errors
- Verify sufficient system resources

### Debug Mode
Run tests with debug output:
```bash
# Unit tests with debug
python tests/test_optimizations_unit.py 2>&1 | tee test_debug.log

# API tests with debug
python tests/test_api_quick.py 2>&1 | tee api_debug.log
```
