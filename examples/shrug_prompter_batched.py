# Example: Modified ShrugPrompter that batches requests

class ShrugBatchPrompter:
    """
    Batched version of ShrugPrompter that processes multiple 
    independent inferences in a single API call.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("*",),
                "max_tokens": ("INT", {"default": 512}),
                "temperature": ("FLOAT", {"default": 1.00}),
                "top_p": ("FLOAT", {"default": 0.95}),
            },
            "optional": {
                # Expects lists of prompts and images
                "system_prompts": ("STRING_LIST",),  # List of system prompts
                "user_prompts": ("STRING_LIST",),    # List of user prompts  
                "images": ("IMAGE",),                 # Batch of images
                "use_batch_api": ("BOOLEAN", {"default": True}),
            },
        }
    
    def execute_prompt(self, context, max_tokens, temperature, top_p,
                      system_prompts=None, user_prompts=None, images=None, 
                      use_batch_api=True):
        
        provider_config = context.get("provider_config")
        if not provider_config:
            raise ValueError("A `provider_config` is required.")
        
        # Convert images to base64
        image_b64_list = tensors_to_base64_list(images) if images is not None else []
        
        # Ensure we have matching counts
        num_inferences = len(image_b64_list)
        if len(system_prompts) != num_inferences or len(user_prompts) != num_inferences:
            raise ValueError(f"Mismatched counts: {len(system_prompts)} systems, "
                           f"{len(user_prompts)} users, {num_inferences} images")
        
        if use_batch_api and num_inferences > 1:
            # Use batch API for multiple inferences
            responses = self._batch_inference(
                provider_config, system_prompts, user_prompts, 
                image_b64_list, max_tokens, temperature, top_p
            )
        else:
            # Fall back to individual requests
            responses = self._individual_inference(
                provider_config, system_prompts, user_prompts,
                image_b64_list, max_tokens, temperature, top_p
            )
        
        context["llm_responses"] = responses
        return (context,)
    
    def _batch_inference(self, provider_config, system_prompts, user_prompts, 
                        images, max_tokens, temperature, top_p):
        """Send all inferences in a single batched request"""
        
        # Build messages with conversation boundaries
        all_messages = []
        
        for i, (system, user, image_b64) in enumerate(zip(system_prompts, user_prompts, images)):
            # Add boundary marker between conversations (except first)
            if i > 0:
                all_messages.append({
                    "role": "system",
                    "content": "___CONVERSATION_BOUNDARY___"
                })
            
            # Add this conversation
            all_messages.append({"role": "system", "content": system})
            all_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            })
        
        # Send batch request
        kwargs = {
            "provider": provider_config["provider"],
            "base_url": provider_config["base_url"],
            "api_key": provider_config["api_key"],
            "llm_model": provider_config["llm_model"],
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "processing_mode": "sequential",
            "return_individual": True
        }
        
        response = run_async(send_request(**kwargs))
        
        # Extract individual completions
        if "completions" in response:
            return [comp["choices"][0]["message"]["content"] for comp in response["completions"]]
        else:
            # Fallback if batch API not available
            return [response.get("choices", [{}])[0].get("message", {}).get("content", "Error")]
    
    def _individual_inference(self, provider_config, system_prompts, user_prompts,
                            images, max_tokens, temperature, top_p):
        """Fall back to individual requests (current behavior)"""
        responses = []
        
        for system, user, image_b64 in zip(system_prompts, user_prompts, images):
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
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
            responses.append(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        
        return responses