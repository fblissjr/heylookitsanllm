# Modified _build_and_execute_request for ShrugPrompter to optionally use batch mode

def _build_and_execute_request_with_batch(self, provider_config, system, user, images, mask, max_tokens, temp, top_p, use_batch=False):
    """
    Build and execute request with optional batch processing support.
    
    If use_batch=True and multiple images are provided, it will send all as separate
    inferences in a single batch request.
    """
    
    if use_batch and images and len(images) > 1:
        # Build batch request with multiple independent inferences
        messages = []
        
        for i, img_b64 in enumerate(images):
            # Add system prompt for this inference
            messages.append({"role": "system", "content": system})
            
            # Add user prompt with image
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
            "processing_mode": "sequential",  # Process each independently
            "return_individual": True,        # Get separate responses
            "mask": mask
        }
        
        response = run_async(send_request(**kwargs))
        
        # Extract individual completions
        if "completions" in response:
            # Return list of responses
            return [comp["choices"][0]["message"]["content"] for comp in response["completions"]]
        else:
            # Fallback to single response
            return response
    
    else:
        # Standard single inference request (current behavior)
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


# Alternative: Create a dedicated batch node
class ShrugBatchPrompter:
    """
    Batch version of ShrugPrompter that processes multiple inputs in one API call.
    Perfect for workflows with multiple images that need independent analysis.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "contexts": ("*",),  # List of contexts
                "system_prompt": ("STRING", {"multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "max_tokens": ("INT", {"default": 512}),
                "temperature": ("FLOAT", {"default": 1.00}),
                "top_p": ("FLOAT", {"default": 0.95}),
            },
            "optional": {
                "images": ("IMAGE",),  # Batch of images
                "processing_mode": (["sequential", "sequential_with_context"], {"default": "sequential"}),
                "use_batch_api": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("contexts",)
    FUNCTION = "execute_batch_prompt"
    CATEGORY = "Shrug Nodes/Logic"
    OUTPUT_IS_LIST = (True,)
    
    def execute_batch_prompt(self, contexts, system_prompt, user_prompt, max_tokens, 
                           temperature, top_p, images=None, processing_mode="sequential", 
                           use_batch_api=True):
        
        if not isinstance(contexts, list):
            contexts = [contexts]
        
        # Get provider config from first context
        provider_config = contexts[0].get("provider_config")
        if not provider_config:
            raise ValueError("A `provider_config` is required.")
        
        # Process images
        image_b64_list = tensors_to_base64_list(images) if images is not None else []
        
        if use_batch_api and len(image_b64_list) > 1:
            # Build batch request
            messages = []
            
            for i, img_b64 in enumerate(image_b64_list):
                # Add system prompt
                messages.append({"role": "system", "content": system_prompt})
                
                # Add user prompt with image
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
                messages.append({"role": "user", "content": user_content})
                
                # Add boundary between conversations (except last)
                if i < len(image_b64_list) - 1:
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
                "temperature": temperature,
                "top_p": top_p,
                "processing_mode": processing_mode,
                "return_individual": True
            }
            
            response = run_async(send_request(**kwargs))
            
            # Process batch response
            if "completions" in response:
                # Update each context with its response
                for i, (context, completion) in enumerate(zip(contexts, response["completions"])):
                    context["llm_response"] = completion
                    context["batch_index"] = i
                    context["batch_total"] = len(response["completions"])
            else:
                # Single response - assign to first context
                contexts[0]["llm_response"] = response
        
        else:
            # Fallback: Process individually
            for i, (context, img_b64) in enumerate(zip(contexts, image_b64_list)):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ]
                
                kwargs = {
                    "provider": provider_config["provider"],
                    "base_url": provider_config["base_url"],
                    "api_key": provider_config["api_key"],
                    "llm_model": provider_config["llm_model"],
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
                
                response = run_async(send_request(**kwargs))
                context["llm_response"] = response
        
        return (contexts,)