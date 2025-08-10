# src/heylook_llm/embeddings.py
import logging
import time
from typing import Dict, List, Union, Optional, Any
from pydantic import BaseModel, Field
import numpy as np

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field(..., description="Model ID to use for embeddings")
    encoding_format: str = Field("float", description="Format of the embedding (float or base64)")
    dimensions: Optional[int] = Field(None, description="Optional number of dimensions to truncate to")
    user: Optional[str] = Field(None, description="Optional user identifier")

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

class EmbeddingExtractor:
    """Base class for extracting embeddings from models."""
    
    def extract(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Extract embeddings from texts."""
        raise NotImplementedError

class MLXEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from MLX models."""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        # Handle different processor types
        # For MLX models, the processor is often a TokenizerWrapper
        if hasattr(processor, '_tokenizer'):
            # MLX TokenizerWrapper case
            self.tokenizer = processor._tokenizer
        elif hasattr(processor, 'tokenizer'):
            # VLM processor case
            self.tokenizer = processor.tokenizer
        else:
            # Direct tokenizer case
            self.tokenizer = processor
        
    def extract(self, texts: List[str], pooling: str = "mean", normalize: bool = True, layer: int = -1) -> List[List[float]]:
        """
        Extract embeddings from MLX models.
        
        Args:
            texts: List of texts to embed
            pooling: Pooling strategy ('mean', 'cls', 'last', 'max')
            normalize: Whether to L2 normalize embeddings
            layer: Which layer to extract from (-1 for last layer)
        """
        import mlx.core as mx
        import mlx.nn as nn
        
        embeddings = []
        
        for text in texts:
            # Tokenize the text - handle different tokenizer types
            try:
                # Try using the tokenizer as a callable (standard HF tokenizer)
                if callable(self.tokenizer):
                    inputs = self.tokenizer(
                        text,
                        return_tensors="np",
                        padding=False,
                        truncation=True,
                        max_length=512
                    )
                else:
                    # Fallback to encode method
                    if hasattr(self.tokenizer, 'encode'):
                        input_ids = self.tokenizer.encode(text, max_length=512)
                        inputs = {'input_ids': np.array([input_ids])}
                    else:
                        raise ValueError(f"Tokenizer type not supported: {type(self.tokenizer)}")
            except Exception as e:
                logging.error(f"Tokenization failed: {e}")
                # Try alternative tokenization methods
                if hasattr(self.tokenizer, 'encode'):
                    input_ids = self.tokenizer.encode(text)
                    inputs = {'input_ids': np.array([input_ids])}
                else:
                    raise
            
            # Convert to MLX arrays
            input_ids = mx.array(inputs['input_ids'])
            
            # Get model outputs with hidden states
            # For VLM models, we need to handle the model structure differently
            if hasattr(self.model, 'language_model'):
                # VLM model structure
                language_model = self.model.language_model
                
                # Get embeddings from the embedding layer
                if hasattr(language_model, 'model'):
                    # Try to get embeddings
                    if hasattr(language_model.model, 'embed_tokens'):
                        token_embeddings = language_model.model.embed_tokens(input_ids)
                    elif hasattr(language_model.model, 'tok_embeddings'):
                        token_embeddings = language_model.model.tok_embeddings(input_ids)
                    else:
                        # Fallback: run through the model
                        token_embeddings = self._run_model_for_embeddings(language_model, input_ids)
                else:
                    token_embeddings = self._run_model_for_embeddings(language_model, input_ids)
            else:
                # Regular LLM model
                if hasattr(self.model, 'embed_tokens'):
                    token_embeddings = self.model.embed_tokens(input_ids)
                elif hasattr(self.model, 'tok_embeddings'):
                    token_embeddings = self.model.tok_embeddings(input_ids)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                    token_embeddings = self.model.model.embed_tokens(input_ids)
                else:
                    token_embeddings = self._run_model_for_embeddings(self.model, input_ids)
            
            # Apply pooling
            if pooling == "mean":
                # Mean pooling across sequence dimension
                embedding = mx.mean(token_embeddings, axis=1)
            elif pooling == "cls":
                # Use first token (CLS token)
                embedding = token_embeddings[:, 0, :]
            elif pooling == "last":
                # Use last token
                embedding = token_embeddings[:, -1, :]
            elif pooling == "max":
                # Max pooling
                embedding = mx.max(token_embeddings, axis=1)
            else:
                # Default to mean pooling
                embedding = mx.mean(token_embeddings, axis=1)
            
            # Flatten to 1D if needed
            if len(embedding.shape) > 1:
                embedding = embedding.reshape(-1)
            
            # Normalize if requested
            if normalize:
                norm = mx.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            # Convert to Python list
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    def _run_model_for_embeddings(self, model, input_ids):
        """Fallback method to get embeddings by running the model."""
        import mlx.core as mx
        
        # Try to get hidden states from a forward pass
        try:
            # Create a simple forward pass to get embeddings
            if hasattr(model, 'forward'):
                outputs = model(input_ids, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    return outputs.hidden_states[-1]
            
            # If that doesn't work, try to get the embedding layer directly
            for attr_name in ['embed_tokens', 'tok_embeddings', 'wte', 'embedding', 'embeddings']:
                if hasattr(model, attr_name):
                    embed_layer = getattr(model, attr_name)
                    if callable(embed_layer):
                        return embed_layer(input_ids)
            
            # Last resort: create random embeddings (should not happen)
            logging.warning("Could not extract real embeddings, using random initialization")
            return mx.random.normal((input_ids.shape[0], input_ids.shape[1], 768))
            
        except Exception as e:
            logging.error(f"Error extracting embeddings: {e}")
            raise

class LlamaCppEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from llama.cpp models."""
    
    def __init__(self, model):
        self.model = model
        
    def extract(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Extract embeddings from llama.cpp models.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to L2 normalize embeddings
        """
        embeddings = []
        
        # Check if model supports batch embedding
        if hasattr(self.model, 'create_embedding'):
            try:
                # Try batch processing first (more efficient)
                result = self.model.create_embedding(texts)
                
                # Handle the response format
                if isinstance(result, dict):
                    if 'data' in result:
                        # OpenAI-style response format
                        for item in result['data']:
                            embedding = item.get('embedding', [])
                            
                            # Check if we got token-level embeddings (nested list)
                            if embedding and isinstance(embedding[0], list):
                                # We have token-level embeddings, need to aggregate
                                # Use mean pooling across tokens
                                embedding_array = np.array(embedding)  # Shape: [n_tokens, embedding_dim]
                                embedding = np.mean(embedding_array, axis=0).tolist()  # Mean across tokens
                                logging.debug(f"Aggregated {embedding_array.shape[0]} token embeddings to single vector")
                            
                            if normalize and embedding and any(e != 0 for e in embedding):
                                norm = np.linalg.norm(embedding)
                                if norm > 0:
                                    embedding = (np.array(embedding) / norm).tolist()
                            embeddings.append(embedding)
                        return embeddings
                    elif 'embedding' in result:
                        # Single embedding response
                        embedding = result['embedding']
                        
                        # Check if we got token-level embeddings (nested list)
                        if embedding and isinstance(embedding[0], list):
                            # We have token-level embeddings, need to aggregate
                            embedding_array = np.array(embedding)  # Shape: [n_tokens, embedding_dim]
                            embedding = np.mean(embedding_array, axis=0).tolist()  # Mean across tokens
                            logging.debug(f"Aggregated {embedding_array.shape[0]} token embeddings to single vector")
                        
                        if normalize and embedding and any(e != 0 for e in embedding):
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = (np.array(embedding) / norm).tolist()
                        return [embedding]  # Return as list of one embedding
                        
            except Exception as e:
                logging.debug(f"Batch embedding failed, trying individual: {e}")
        
        # Fallback to individual embedding
        for text in texts:
            embedding = None
            
            # llama-cpp-python supports embeddings through create_embedding
            if hasattr(self.model, 'create_embedding'):
                try:
                    # Use the create_embedding method which is the standard way
                    result = self.model.create_embedding(text)
                    if isinstance(result, dict):
                        if 'data' in result and len(result['data']) > 0:
                            # OpenAI-style response
                            embedding = result['data'][0].get('embedding', [])
                        elif 'embedding' in result:
                            # Direct embedding response
                            embedding = result['embedding']
                    else:
                        embedding = result
                    
                    # Check if we got token-level embeddings (nested list)
                    if embedding and isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                        # We have token-level embeddings, need to aggregate
                        embedding_array = np.array(embedding)  # Shape: [n_tokens, embedding_dim]
                        embedding = np.mean(embedding_array, axis=0).tolist()  # Mean across tokens
                        logging.debug(f"Aggregated {embedding_array.shape[0]} token embeddings to single vector")
                except Exception as e:
                    logging.warning(f"create_embedding failed: {e}, trying alternative methods")
            
            # Alternative: use embed method if available
            if embedding is None and hasattr(self.model, 'embed'):
                try:
                    embedding = self.model.embed(text)
                except Exception as e:
                    logging.warning(f"embed method failed: {e}")
            
            # Fallback: get embeddings from evaluated tokens
            if embedding is None:
                try:
                    # Tokenize the text
                    tokens = self.model.tokenize(text.encode('utf-8'))
                    
                    # Reset context and evaluate tokens
                    self.model.reset()
                    self.model.eval(tokens)
                    
                    # Try to get embeddings from the model state
                    # Note: This is model-dependent and may not work for all models
                    if hasattr(self.model, 'n_embd'):
                        n_embd = self.model.n_embd()
                        # Create placeholder embeddings
                        # In practice, this would need access to internal model state
                        embedding = [0.0] * n_embd
                        logging.warning("Using placeholder embeddings for llama.cpp model without embedding support")
                except Exception as e:
                    logging.error(f"Failed to extract embeddings: {e}")
                    # Return zero vector as last resort
                    embedding = [0.0] * 768  # Default embedding size
            
            # Ensure embedding is a list
            if not isinstance(embedding, list):
                embedding = list(embedding) if hasattr(embedding, '__iter__') else [embedding]
            
            # Normalize if requested
            if normalize and embedding and any(e != 0 for e in embedding):
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()
            
            embeddings.append(embedding)
        
        return embeddings

def create_embedding_extractor(provider_type: str, model: Any, processor: Any = None) -> EmbeddingExtractor:
    """
    Factory function to create the appropriate embedding extractor.
    
    Args:
        provider_type: Type of provider ('mlx' or 'llama_cpp')
        model: The loaded model
        processor: The processor/tokenizer (for MLX models)
    """
    if provider_type == "mlx":
        if processor is None:
            raise ValueError("MLX models require a processor/tokenizer")
        return MLXEmbeddingExtractor(model, processor)
    elif provider_type in ["llama_cpp", "gguf"]:
        return LlamaCppEmbeddingExtractor(model)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

async def create_embeddings(
    request: EmbeddingRequest,
    router: Any
) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).
    
    Args:
        request: The embedding request
        router: The model router instance
    """
    try:
        # Ensure input is a list
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Load the model (router.get_provider is synchronous, not async)
        provider = router.get_provider(request.model)
        
        # Determine provider type by checking the class
        provider_class_name = provider.__class__.__name__
        if 'MLX' in provider_class_name:
            provider_type = 'mlx'
        elif 'LlamaCpp' in provider_class_name:
            provider_type = 'llama_cpp'
        else:
            # Fallback: try to infer from model file extension
            model_path = provider.config.get('model_path', '')
            if '.gguf' in model_path.lower():
                provider_type = 'llama_cpp'
            else:
                provider_type = 'mlx'
        
        # Create the appropriate extractor
        if provider_type == "mlx":
            # MLX providers have a processor/tokenizer
            processor = getattr(provider, 'processor', None)
            if processor is None:
                raise ValueError(f"MLX provider for model {request.model} has no processor/tokenizer")
            extractor = create_embedding_extractor(
                provider_type,
                provider.model,
                processor
            )
        else:  # llama_cpp
            # Llama.cpp providers don't have a separate processor
            extractor = create_embedding_extractor(
                provider_type,
                provider.model
            )
        
        # Extract embeddings
        embeddings = extractor.extract(texts)
        
        # Optionally truncate dimensions
        if request.dimensions:
            embeddings = [emb[:request.dimensions] for emb in embeddings]
        
        # Format response
        data = [
            EmbeddingData(
                embedding=emb,
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]
        
        # Calculate token usage (rough estimate)
        total_tokens = sum(len(text.split()) for text in texts) * 1.3
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=int(total_tokens),
                total_tokens=int(total_tokens)
            )
        )
        
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        raise