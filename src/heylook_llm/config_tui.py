# src/heylook_llm/config_tui.py
"""Terminal UI components for interactive model configuration.

Provides reusable TUI components for:
- Editing sampler parameters (temperature, top_p, etc.)
- Editing KV cache settings (cache_type, kv_bits, etc.)
- Selecting models from lists
- Confirming configuration changes with diffs
"""

import questionary
from typing import Dict, Any, List, Optional
from questionary import Style


# Custom style for consistent UI
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),       # Question mark color
    ('question', 'bold'),                # Question text
    ('answer', 'fg:#f44336 bold'),      # Answer text
    ('pointer', 'fg:#673ab7 bold'),     # Selection pointer
    ('highlighted', 'fg:#673ab7 bold'), # Highlighted choice
    ('selected', 'fg:#cc5454'),         # Selected items
    ('separator', 'fg:#cc5454'),        # Separators
    ('instruction', ''),                # Instructions
    ('text', ''),                       # Normal text
])


class ConfigEditor:
    """Reusable configuration editing components."""

    def __init__(self):
        self.style = custom_style

    def edit_sampler_params(self, current_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Interactive editor for sampler parameters.

        Args:
            current_values: Current parameter values (defaults if None)

        Returns:
            Dict of updated sampler parameters
        """
        if current_values is None:
            current_values = {
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 50,
                'min_p': 0.05,
                'max_tokens': 512,
                'repetition_penalty': 1.0,
                'repetition_context_size': 20
            }

        print("\n=== Sampler Configuration ===")
        print("Configure default sampling parameters for models")
        print()

        # Ask if user wants to customize
        customize = questionary.confirm(
            "Customize sampler parameters?",
            default=False,
            style=self.style
        ).ask()

        if not customize:
            return current_values

        result = {}

        # Temperature
        temp = questionary.text(
            "Temperature (0.0-2.0, controls randomness):",
            default=str(current_values.get('temperature', 0.7)),
            validate=lambda x: self._validate_float_range(x, 0.0, 2.0),
            style=self.style
        ).ask()
        result['temperature'] = float(temp)

        # Top-p
        top_p = questionary.text(
            "Top-p (0.0-1.0, nucleus sampling):",
            default=str(current_values.get('top_p', 0.95)),
            validate=lambda x: self._validate_float_range(x, 0.0, 1.0),
            style=self.style
        ).ask()
        result['top_p'] = float(top_p)

        # Top-k
        top_k = questionary.text(
            "Top-k (1-100, top-k sampling):",
            default=str(current_values.get('top_k', 50)),
            validate=lambda x: self._validate_int_range(x, 1, 100),
            style=self.style
        ).ask()
        result['top_k'] = int(top_k)

        # Min-p
        min_p = questionary.text(
            "Min-p (0.0-1.0, minimum probability):",
            default=str(current_values.get('min_p', 0.05)),
            validate=lambda x: self._validate_float_range(x, 0.0, 1.0),
            style=self.style
        ).ask()
        result['min_p'] = float(min_p)

        # Max tokens
        max_tokens = questionary.text(
            "Max tokens (1-32768):",
            default=str(current_values.get('max_tokens', 512)),
            validate=lambda x: self._validate_int_range(x, 1, 32768),
            style=self.style
        ).ask()
        result['max_tokens'] = int(max_tokens)

        # Repetition penalty
        rep_penalty = questionary.text(
            "Repetition penalty (1.0-2.0):",
            default=str(current_values.get('repetition_penalty', 1.0)),
            validate=lambda x: self._validate_float_range(x, 1.0, 2.0),
            style=self.style
        ).ask()
        result['repetition_penalty'] = float(rep_penalty)

        # Repetition context size
        rep_context = questionary.text(
            "Repetition context size (tokens):",
            default=str(current_values.get('repetition_context_size', 20)),
            validate=lambda x: self._validate_int_range(x, 1, 1000),
            style=self.style
        ).ask()
        result['repetition_context_size'] = int(rep_context)

        return result

    def edit_kv_cache_params(
        self,
        current_values: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Interactive editor for KV cache parameters (MLX only).

        Args:
            current_values: Current parameter values
            model_info: Model information (size, type, etc.) for recommendations

        Returns:
            Dict of updated KV cache parameters
        """
        if model_info is None:
            model_info = {}

        size_gb = model_info.get('size_gb', 0)
        is_large = size_gb > 30

        print("\n=== KV Cache Configuration (MLX Models) ===")
        if is_large:
            print(f"Model size: {size_gb:.1f}GB (LARGE - quantization recommended)")
        elif size_gb > 0:
            print(f"Model size: {size_gb:.1f}GB")
        print()

        # Ask if user wants to customize
        customize = questionary.confirm(
            "Configure KV cache settings?",
            default=is_large,  # Default yes for large models
            style=self.style
        ).ask()

        if not customize:
            # Return smart defaults
            if is_large:
                return {
                    'cache_type': 'quantized',
                    'kv_bits': 4,
                    'kv_group_size': 32,
                    'quantized_kv_start': 512,
                    'max_kv_size': 2048
                }
            else:
                return {'cache_type': 'standard'}

        result = {}

        # Cache type
        cache_type = questionary.select(
            "Cache type:",
            choices=[
                questionary.Choice("standard", "Standard (full precision)"),
                questionary.Choice("quantized", "Quantized (save memory)")
            ],
            default="quantized" if is_large else "standard",
            style=self.style
        ).ask()
        result['cache_type'] = cache_type

        # If quantized, ask for quantization settings
        if cache_type == 'quantized':
            # KV bits
            kv_bits = questionary.select(
                "KV cache quantization bits:",
                choices=[
                    questionary.Choice("4", "4-bit (75% memory savings, slight quality loss)"),
                    questionary.Choice("8", "8-bit (50% memory savings, minimal quality loss)")
                ],
                default="4" if is_large else "8",
                style=self.style
            ).ask()
            result['kv_bits'] = int(kv_bits)

            # KV group size
            kv_group = questionary.select(
                "KV group size:",
                choices=[
                    questionary.Choice("32", "32 (faster, less precise)"),
                    questionary.Choice("64", "64 (balanced)"),
                    questionary.Choice("128", "128 (slower, more precise)")
                ],
                default="32" if is_large else "64",
                style=self.style
            ).ask()
            result['kv_group_size'] = int(kv_group)

            # Quantized start
            quant_start = questionary.text(
                "Start quantizing after N tokens:",
                default="512" if is_large else "1024",
                validate=lambda x: self._validate_int_range(x, 0, 10000),
                style=self.style
            ).ask()
            result['quantized_kv_start'] = int(quant_start)

            # Max KV size
            set_max = questionary.confirm(
                "Set maximum KV cache size?",
                default=is_large,
                style=self.style
            ).ask()

            if set_max:
                max_size = questionary.text(
                    "Maximum KV cache size (tokens):",
                    default="2048" if is_large else "4096",
                    validate=lambda x: self._validate_int_range(x, 512, 32768),
                    style=self.style
                ).ask()
                result['max_kv_size'] = int(max_size)

        return result

    def select_model(self, models: List[Dict[str, Any]]) -> str:
        """Interactive model selector.

        Args:
            models: List of model dicts with 'id', 'provider', 'enabled' keys

        Returns:
            Selected model ID
        """
        print("\n=== Select Model ===")
        print()

        choices = []
        for model in models:
            enabled_str = "[enabled]" if model.get('enabled', True) else "[disabled]"
            provider_str = f"[{model.get('provider', 'unknown')}]"
            vision_str = "[vision]" if model.get('config', {}).get('vision') else ""

            label = f"{model['id']:40} {provider_str:10} {enabled_str:12} {vision_str}"
            choices.append(questionary.Choice(model['id'], label))

        selected = questionary.select(
            "Select a model:",
            choices=choices,
            style=self.style
        ).ask()

        return selected

    def confirm_changes(self, before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        """Show diff and confirm changes.

        Args:
            before: Original configuration
            after: Updated configuration

        Returns:
            True if user confirms, False otherwise
        """
        print("\n=== Configuration Changes ===")
        print()

        # Find changes
        changes = []
        all_keys = set(before.keys()) | set(after.keys())

        for key in sorted(all_keys):
            old_val = before.get(key)
            new_val = after.get(key)

            if old_val != new_val:
                if old_val is None:
                    changes.append(f"  + {key}: {new_val}")
                elif new_val is None:
                    changes.append(f"  - {key}: {old_val}")
                else:
                    changes.append(f"  ~ {key}: {old_val} → {new_val}")

        if not changes:
            print("No changes detected.")
            return False

        for change in changes:
            print(change)

        print()
        return questionary.confirm(
            "Apply these changes?",
            default=True,
            style=self.style
        ).ask()

    # Validation helpers
    def _validate_float_range(self, value: str, min_val: float, max_val: float) -> bool:
        """Validate float in range."""
        try:
            val = float(value)
            if min_val <= val <= max_val:
                return True
            return f"Value must be between {min_val} and {max_val}"
        except ValueError:
            return "Please enter a valid number"

    def _validate_int_range(self, value: str, min_val: int, max_val: int) -> bool:
        """Validate integer in range."""
        try:
            val = int(value)
            if min_val <= val <= max_val:
                return True
            return f"Value must be between {min_val} and {max_val}"
        except ValueError:
            return "Please enter a valid integer"
