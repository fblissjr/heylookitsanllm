import torch
import json
from typing import Any, List, Tuple, Union

class ShrugBatchResponseParser:
    """
    Enhanced VLM Response Parser that handles both single and batch responses.
    
    Can process:
    1. Single response (standard OpenAI format)
    2. Batch responses (multiple completions from batch API)
    3. List of responses from multiple API calls
    """
    OUTPUT_IS_LIST = (True, True, True, True)  # All outputs are lists for batch support

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"context": ("*",)},
            "optional": {
                "original_images": ("IMAGE",),  # Now expects batch of images
                "mask_size": ("INT", {"default": 256, "min": 64, "max": 2048}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "debug_mode": ("BOOLEAN", {"default": False}),
                "batch_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("OPTIMIZED_PROMPTS", "DETECTED_MASKS", "DETECTED_LABELS", "DEBUG_INFO")
    FUNCTION = "parse_response"
    CATEGORY = "Shrug Nodes/Parsing"

    def parse_response(self, context, original_images=None, mask_size=256, 
                      confidence_threshold=0.5, debug_mode=False, batch_mode=True):
        
        # Default fallback prompt
        default_prompt = "A cinematic scene with dynamic camera movement, professional lighting, and compelling visual storytelling with smooth transitions"
        
        try:
            # Validate context
            if context is None or not isinstance(context, dict):
                return ([default_prompt], [self._create_empty_mask(mask_size, mask_size)], 
                       [""], ["Invalid context, using default prompt"])
            
            # Check if we have batch responses or single response
            llm_responses = self._extract_all_responses(context, batch_mode)
            
            # Process each response
            prompts, masks, labels, debug_infos = [], [], [], []
            
            for i, response in enumerate(llm_responses):
                # Extract text from this response
                response_text = self._extract_response_content(response)
                
                # Validate and clean
                if not response_text or not isinstance(response_text, str) or response_text.strip() == "":
                    response_text = default_prompt
                
                response_text = self._sanitize_prompt(response_text)
                
                # Get corresponding image if available
                image = None
                if original_images is not None and len(original_images) > i:
                    image = original_images[i:i+1]  # Get single image from batch
                
                # Try to parse detection
                mask, label = self._try_parse_detection(response_text, image, mask_size, confidence_threshold)
                
                # Collect results
                prompts.append(response_text)
                masks.append(mask)
                labels.append(label)
                
                if debug_mode:
                    debug_infos.append(f"Response {i+1}: {response_text[:100]}...")
            
            # Ensure we always return lists
            if not prompts:
                prompts = [default_prompt]
                masks = [self._create_empty_mask(mask_size, mask_size)]
                labels = [""]
                debug_infos = ["No responses found"]
            
            return (prompts, masks, labels, debug_infos)
            
        except Exception as e:
            # Exception handler - return single item lists
            return ([default_prompt], [self._create_empty_mask(mask_size, mask_size)], 
                   [""], [f"Exception in response parser: {str(e)}"])
    
    def _extract_all_responses(self, context: dict, batch_mode: bool) -> List[Any]:
        """Extract all responses from context, handling various formats"""
        
        # Check for batch responses first (from new batch API)
        if batch_mode and "llm_responses" in context:
            # Multiple responses from batch API
            responses = context["llm_responses"]
            if isinstance(responses, list):
                return responses
            else:
                return [responses]
        
        # Check for single response (standard format)
        if "llm_response" in context:
            response = context["llm_response"]
            
            # Check if it's a batch response with completions
            if isinstance(response, dict) and "completions" in response:
                return response["completions"]
            
            # Single response
            return [response]
        
        # Fallback - look for any response-like data
        for key in ["responses", "results", "outputs"]:
            if key in context:
                value = context[key]
                if isinstance(value, list):
                    return value
                else:
                    return [value]
        
        return []
    
    def _extract_response_content(self, resp: Any) -> str:
        """Extract text content from VLM response with robust fallback handling"""
        default_fallback = "A cinematic scene with dynamic camera movement, professional lighting, and compelling visual storytelling with smooth transitions"
        
        if resp is None:
            return default_fallback
        
        # Handle string responses
        if isinstance(resp, str):
            result = resp.strip()
            return result if result else default_fallback
        
        # Handle non-dict responses
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
                            result = content.strip()
                            return result if result else default_fallback
        except:
            pass
        
        # Try common response keys
        for key in ["content", "text", "response", "output", "message", "result"]:
            try:
                if key in resp and resp[key]:
                    value = resp[key]
                    result = str(value).strip()
                    return result if result else default_fallback
            except:
                continue
        
        return default_fallback
    
    def _sanitize_prompt(self, text: str) -> str:
        """Clean and validate text response for downstream processing"""
        if not isinstance(text, str):
            return "A cinematic scene with dynamic camera movement"
        
        text = text.replace('\x00', '').strip()
        
        if not text:
            return "A cinematic scene with dynamic camera movement"
        
        return text
    
    def _try_parse_detection(self, text: str, image: torch.Tensor, size: int, thresh: float) -> Tuple[torch.Tensor, str]:
        """Attempt to parse detection JSON and create corresponding mask"""
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "box" in data and "label" in data:
                h, w = (image.shape[-2], image.shape[-1]) if image is not None else (size, size)
                box, label, conf = data.get("box", []), data.get("label", "unknown"), data.get("confidence", 1.0)
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