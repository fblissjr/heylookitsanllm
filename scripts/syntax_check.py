import ast
import sys

def check_syntax(filepath):
    try:
        with open(filepath, "r") as f:
            source = f.read()
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return False

files_to_check = [
    "src/heylook_llm/__init__.py",
    "src/heylook_llm/analytics_config.py",
    "src/heylook_llm/api.py",
    "src/heylook_llm/api_capabilities.py",
    "src/heylook_llm/api_multipart.py",
    "src/heylook_llm/batch_extensions.py",
    "src/heylook_llm/batch_processor.py",
    "src/heylook_llm/config.py",
    "src/heylook_llm/data_endpoint.py",
    "src/heylook_llm/data_loader.py",
    "src/heylook_llm/embeddings.py",
    "src/heylook_llm/metrics_db.py",
    "src/heylook_llm/metrics_db_wrapper.py",
    "src/heylook_llm/model_importer.py",
    "src/heylook_llm/openapi_enhancements.py",
    "src/heylook_llm/openapi_examples.py",
    "src/heylook_llm/queue_manager.py",
    "src/heylook_llm/router.py",
    "src/heylook_llm/server.py",
    "src/heylook_llm/stt_api.py",
    "src/heylook_llm/utils.py",
    "src/heylook_llm/utils_resize.py",
    "src/heylook_llm/hidden_states.py",
    "src/heylook_llm/optimizations/__init__.py",
    "src/heylook_llm/optimizations/fast_image.py",
    "src/heylook_llm/optimizations/fast_json.py",
    "src/heylook_llm/optimizations/mlx_optimizations.py",
    "src/heylook_llm/optimizations/status.py",
    "src/heylook_llm/providers/__init__.py",
    "src/heylook_llm/providers/base.py",
    "src/heylook_llm/providers/mlx_stt_provider.py",
    "src/heylook_llm/providers/llama_cpp_provider.py",
    "src/heylook_llm/providers/mlx_batch_text.py",
    "src/heylook_llm/providers/mlx_batch_vision.py",
    "src/heylook_llm/providers/mlx_provider.py",
    "src/heylook_llm/providers/mlx_stt_provider.py",
    "src/heylook_llm/providers/common/__init__.py",
    "src/heylook_llm/providers/common/batch_vision.py",
    "src/heylook_llm/providers/common/cache_helpers.py",
    "src/heylook_llm/providers/common/performance_monitor.py",
    "src/heylook_llm/providers/common/prompt_cache.py",
    "src/heylook_llm/providers/common/samplers.py",
    "src/heylook_llm/providers/common/vlm_generation.py"
]

success = True
for file in files_to_check:
    if not check_syntax(file):
        success = False

if success:
    print("All files passed syntax check.")
    sys.exit(0)
else:
    print("Some files failed syntax check.")
    sys.exit(1)
