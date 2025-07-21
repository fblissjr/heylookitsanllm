# Updated ShrugPrompter with Batch Processing Support

import hashlib
import json
import torch
from typing import List, Dict, Any, Optional, Union

try:
    from ..utils import tensors_to_base64_list, run_async
    from ..shrug_router import send_request
except ImportError:
    from utils import tensors_to_base64_list, run_async
    from shrug_router import send_request

class ShrugPrompterBatch:
    """
    Enhanced ShrugPrompter that supports batch processing.
    Can handle multiple images as either:
    1. Single inference with multiple images (default/legacy)
    2. Multiple separate inferences (batch mode)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("*",),
                "system_prompt": ("STRING", {"multiline": True}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 32000}),
                "temperature": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.00, "max": 1.00, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "metadata": ("STRING", {"default": "{}"}),
                "template_vars": ("STRING", {"multiline": True, "default": "{}"}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "debug_mode": ("BOOLEAN", {"default": False}),
                "batch_mode": ("BOOLEAN", {"default": False}),
                "processing_mode": (["single", "sequential", "sequential_with_context"], {"default": "sequential"}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "debug_info")
    FUNCTION = "execute_prompt"
    CATEGORY = "Shrug Nodes/Logic"
    OUTPUT_IS_LIST = (False, True)

    def __init__(self):
        self._cache = {}
        self._cache_max_size = 50

    def execute_prompt(self, context, system_prompt, user_prompt, max_tokens, temperature, top_p,
                      images=None, mask=None, metadata="{}", template_vars="{}", 
                      use_cache=True, debug_mode=False, batch_mode=False, processing_mode="sequential"):

        debug_info = []
        context["vlm_metadata"] = metadata
        context["debug_info"] = debug_info

        try:
            provider_config = context.get("provider_config")
            if not provider_config:
                raise ValueError("A `provider_config` is required.")

            template_variables = json.loads(template_vars) if template_vars.strip() else {}
            processed_system = system_prompt.format(**template_variables)
            processed_user = user_prompt.format(**template_variables)

            # Process images
            image_b64_list = self._process_images(images)
            mask_b64 = self._process_mask(mask)

            # Determine processing approach
            if batch_mode and image_b64_list and len(image_b64_list) > 1:
                # Batch processing - multiple separate inferences
                if debug_mode:
                    debug_info.append(f"Batch mode: Processing {len(image_b64_list)} images separately")
                
                response_data = self._build_and_execute_batch_request(
                    provider_config, processed_system, processed_user, 
                    image_b64_list, mask_b64, max_tokens, temperature, top_p,
                    processing_mode
                )
                
                # Store multiple responses
                context["llm_responses"] = response_data.get("completions", [])
                context["batch_mode"] = True
                context["batch_size"] = len(image_b64_list)
                
            else:
                # Single inference (legacy mode or single image)
                if debug_mode:
                    debug_info.append(f"Single mode: Processing {len(image_b64_list)} images in one request")
                
                cache_key = self._create_cache_key(
                    provider_config, processed_system, processed_user, 
                    max_tokens, temperature, top_p, images, mask
                )
                
                if use_cache and cache_key in self._cache:
                    context["llm_response"] = self._cache[cache_key]
                    return (context, ["\n".join(debug_info)])

                response_data = self._build_and_execute_request(
                    provider_config, processed_system, processed_user, 
                    image_b64_list, mask_b64, max_tokens, temperature, top_p
                )
                
                context["llm_response"] = response_data
                context["batch_mode"] = False

                if use_cache and "error" not in response_data:
                    self._cache[cache_key] = response_data
                    self._cleanup_cache()

        except Exception as e:
            context["llm_response"] = {"error": {"message": f"Critical error in ShrugPrompter: {e}"}}
            if debug_mode:
                debug_info.append(f"Error: {str(e)}")

        return (context, ["\n".join(debug_info)])

    def _build_and_execute_batch_request(self, provider_config, system, user, images, mask, 
                                       max_tokens, temp, top_p, processing_mode):
        """Execute batch request with multiple separate inferences"""
        
        # Build messages for batch processing
        messages = []
        
        for i, img_b64 in enumerate(images):
            # Add system prompt
            messages.append({"role": "system", "content": system})
            
            # Add user message with single image
            user_content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
            messages.append({"role": "user", "content": user_content})
            
            # Add conversation boundary (except for last)
            if i < len(images) - 1:
                messages.append({
                    "role": "system",
                    "content": "___CONVERSATION_BOUNDARY___"
                })
        
        # Send batch request
        kwargs = {
            "provider": provider_config["provider"],
            "base_url": provider_config["base_url"],
            "api_key": provider_config["api_key"],
            "llm_model": provider_config["llm_model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "processing_mode": processing_mode,
            "return_individual": True,
            "mask": mask
        }
        
        return run_async(send_request(**kwargs))

    def _build_and_execute_request(self, provider_config, system, user, images, mask, 
                                  max_tokens, temp, top_p):
        """Execute single request (legacy mode)"""
        user_content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} 
            for img_b64 in images
        ]
        messages = [
            {"role": "system", "content": system}, 
            {"role": "user", "content": user_content}
        ]
        kwargs = {
            "provider": provider_config["provider"],
            "base_url": provider_config["base_url"],
            "api_key": provider_config["api_key"],
            "llm_model": provider_config["llm_model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "mask": mask
        }
        return run_async(send_request(**kwargs))

    def _create_cache_key(self, provider_config, system, user, max_tokens, temp, top_p, images, mask):
        data = {
            "provider": provider_config.get("provider"),
            "model": provider_config.get("llm_model"),
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "images_shape": str(images.shape) if images is not None else "None",
            "mask_shape": str(mask.shape) if mask is not None else "None"
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _cleanup_cache(self):
        if len(self._cache) > self._cache_max_size:
            for key in list(self._cache.keys())[:len(self._cache) - self._cache_max_size]:
                del self._cache[key]

    def _process_images(self, images):
        return tensors_to_base64_list(images) if images is not None else []

    def _process_mask(self, mask):
        if mask is None:
            return None
        masks = tensors_to_base64_list(mask)
        return masks[0] if masks else None


# Updated Response Parser for Batch Support
class ShrugResponseParserBatch:
    """
    Enhanced response parser that handles both single and batch responses.
    """
    OUTPUT_IS_LIST = (True, True, True, True)  # All outputs as lists for batch support

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"context": ("*",)},
            "optional": {
                "original_image": ("IMAGE",),
                "mask_size": ("INT", {"default": 256, "min": 64, "max": 2048}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("OPTIMIZED_PROMPTS", "DETECTED_MASKS", "DETECTED_LABELS", "DEBUG_INFO")
    FUNCTION = "parse_response"
    CATEGORY = "Shrug Nodes/Parsing"

    def parse_response(self, context, original_image=None, mask_size=256, 
                      confidence_threshold=0.5, debug_mode=False):
        
        default_prompt = "A cinematic scene with dynamic camera movement"
        
        try:
            if context is None or not isinstance(context, dict):
                return ([default_prompt], [self._create_empty_mask(mask_size, mask_size)], 
                       [""], ["Invalid context"])

            # Check if batch mode
            if context.get("batch_mode", False) and "llm_responses" in context:
                # Handle multiple responses
                responses = context["llm_responses"]
                prompts, masks, labels, debug_infos = [], [], [], []
                
                for i, response in enumerate(responses):
                    prompt, mask, label, debug = self._parse_single_response(
                        response, original_image, mask_size, 
                        confidence_threshold, debug_mode, i
                    )
                    prompts.append(prompt)
                    masks.append(mask)
                    labels.append(label)
                    debug_infos.append(debug)
                
                return (prompts, masks, labels, debug_infos)
            
            else:
                # Handle single response (legacy)
                llm_response = context.get("llm_response")
                prompt, mask, label, debug = self._parse_single_response(
                    llm_response, original_image, mask_size, 
                    confidence_threshold, debug_mode
                )
                return ([prompt], [mask], [label], [debug])

        except Exception as e:
            return ([default_prompt], [self._create_empty_mask(mask_size, mask_size)], 
                   [""], [f"Exception: {str(e)}"])

    def _parse_single_response(self, response, original_image, mask_size, 
                              confidence_threshold, debug_mode, index=0):
        """Parse a single response"""
        default_prompt = "A cinematic scene with dynamic camera movement"
        
        # Extract text content
        response_text = self._extract_response_content(response)
        
        if not response_text or not isinstance(response_text, str) or response_text.strip() == "":
            response_text = default_prompt
        
        response_text = self._sanitize_prompt(response_text)
        
        # Try to parse detection
        mask, label = self._try_parse_detection(
            response_text, original_image, mask_size, confidence_threshold
        )
        
        debug_info = f"Response {index + 1}: {response_text[:100]}..." if debug_mode else ""
        
        return response_text, mask, label, debug_info

    def _extract_response_content(self, resp: Any) -> str:
        """Extract text content from response"""
        default_fallback = "A cinematic scene with dynamic camera movement"
        
        if resp is None:
            return default_fallback
        
        if isinstance(resp, str):
            result = resp.strip()
            return result if result else default_fallback
        
        if not isinstance(resp, dict):
            try:
                result = str(resp).strip()
                return result if result else default_fallback
            except:
                return default_fallback
        
        # Handle OpenAI-style responses
        try:
            choices = resp.get("choices", [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                choice = choices[0]
                if isinstance(choice, dict) and "message" in choice:
                    message = choice["message"]
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        if content and isinstance(content, str):
                            return content.strip() or default_fallback
        except:
            pass
        
        # Try common response keys
        for key in ["content", "text", "response", "output", "message", "result"]:
            try:
                if key in resp and resp[key]:
                    value = resp[key]
                    return str(value).strip() or default_fallback
            except:
                continue
        
        return default_fallback

    def _sanitize_prompt(self, text: str) -> str:
        """Clean and validate text"""
        if not isinstance(text, str):
            return "A cinematic scene"
        
        text = text.replace('\x00', '').strip()
        
        if not text:
            return "A cinematic scene"
        
        return text

    def _try_parse_detection(self, text: str, image: torch.Tensor, size: int, thresh: float) -> Tuple[torch.Tensor, str]:
        """Try to parse detection JSON"""
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "box" in data and "label" in data:
                h, w = (image.shape[-2], image.shape[-1]) if image is not None else (size, size)
                box = data.get("box", [])
                label = data.get("label", "unknown")
                conf = data.get("confidence", 1.0)
                if conf >= thresh and len(box) == 4:
                    mask = self._create_detection_mask(box, h, w)
                    return (mask, f"{label} ({conf:.2f})")
        except:
            pass
        return (self._create_empty_mask(size, size), "")

    def _create_empty_mask(self, h: int, w: int) -> torch.Tensor:
        return torch.zeros((1, h, w), dtype=torch.float32, device="cpu")

    def _create_detection_mask(self, box: list, h: int, w: int) -> torch.Tensor:
        mask = self._create_empty_mask(h, w)
        try:
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                mask[0, y1:y2, x1:x2] = 1.0
        except:
            pass
        return mask