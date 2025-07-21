# examples/comfyui_duckdb_node.py
"""
Example DuckDB nodes for ComfyUI workflows.

This would go in ComfyUI/custom_nodes/ComfyUI-DuckDB/
"""

import duckdb
import json
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import io
import base64


class DuckDBLoader:
    """Load data from DuckDB queries into ComfyUI workflows."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {
                    "multiline": True,
                    "default": "SELECT prompt, negative_prompt, seed FROM prompts LIMIT 10"
                }),
                "db_path": ("STRING", {
                    "default": "comfyui_data.duckdb"
                }),
                "output_type": (["LIST", "BATCH", "SINGLE"],),
            }
        }
    
    RETURN_TYPES = ("DUCKDB_RESULT",)
    FUNCTION = "load_data"
    CATEGORY = "data/duckdb"
    
    def load_data(self, query: str, db_path: str, output_type: str):
        conn = duckdb.connect(db_path)
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.execute(query).description]
        
        # Convert to list of dicts
        data = [dict(zip(columns, row)) for row in result]
        
        if output_type == "SINGLE":
            return (data[0] if data else {},)
        elif output_type == "BATCH":
            return (data,)
        else:  # LIST
            return (data,)


class DuckDBImageLogger:
    """Log generated images and metadata to DuckDB for analysis."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "db_path": ("STRING", {"default": "comfyui_outputs.duckdb"}),
                "table_name": ("STRING", {"default": "generated_images"}),
            },
            "optional": {
                "prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
                "sampler": ("STRING", {"default": ""}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 7.0}),
                "seed": ("INT", {"default": -1}),
                "metadata": ("STRING", {"default": "{}"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "log_images"
    CATEGORY = "data/duckdb"
    OUTPUT_NODE = True
    
    def log_images(self, images, db_path, table_name, **kwargs):
        conn = duckdb.connect(db_path)
        
        # Create table if not exists
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP DEFAULT current_timestamp,
                image_data BLOB,
                width INTEGER,
                height INTEGER,
                prompt TEXT,
                negative_prompt TEXT,
                model VARCHAR,
                sampler VARCHAR,
                steps INTEGER,
                cfg FLOAT,
                seed BIGINT,
                metadata JSON,
                workflow_hash VARCHAR
            )
        """)
        
        # Convert images to bytes and store
        for i, image in enumerate(images):
            # Convert tensor to PIL Image
            img_array = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            # Insert into database
            conn.execute(f"""
                INSERT INTO {table_name} 
                (image_data, width, height, prompt, negative_prompt, model, sampler, steps, cfg, seed, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                img_bytes,
                pil_image.width,
                pil_image.height,
                kwargs.get('prompt', ''),
                kwargs.get('negative_prompt', ''),
                kwargs.get('model', ''),
                kwargs.get('sampler', ''),
                kwargs.get('steps', 20),
                kwargs.get('cfg', 7.0),
                kwargs.get('seed', -1),
                kwargs.get('metadata', '{}')
            ])
        
        conn.commit()
        conn.close()
        
        return (images,)


class DuckDBAnalytics:
    """Run analytics queries on ComfyUI data."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "db_path": ("STRING", {"default": "comfyui_outputs.duckdb"}),
                "analysis_type": ([
                    "top_prompts",
                    "model_comparison", 
                    "parameter_analysis",
                    "error_rate",
                    "generation_speed",
                    "custom_query"
                ],),
                "custom_query": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "data/duckdb"
    
    def analyze(self, db_path: str, analysis_type: str, custom_query: str):
        conn = duckdb.connect(db_path)
        
        queries = {
            "top_prompts": """
                SELECT prompt, COUNT(*) as count
                FROM generated_images
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY prompt
                ORDER BY count DESC
                LIMIT 20
            """,
            "model_comparison": """
                SELECT 
                    model,
                    COUNT(*) as images,
                    AVG(cfg) as avg_cfg,
                    AVG(steps) as avg_steps
                FROM generated_images
                GROUP BY model
                ORDER BY images DESC
            """,
            "parameter_analysis": """
                SELECT 
                    sampler,
                    AVG(steps) as avg_steps,
                    AVG(cfg) as avg_cfg,
                    COUNT(*) as count
                FROM generated_images
                GROUP BY sampler
                HAVING COUNT(*) > 10
            """,
            "error_rate": """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as total,
                    SUM(CASE WHEN metadata::json->>'error' IS NOT NULL THEN 1 ELSE 0 END) as errors
                FROM generated_images
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                GROUP BY 1
                ORDER BY 1
            """,
            "generation_speed": """
                SELECT 
                    model,
                    sampler,
                    AVG(CAST(metadata::json->>'generation_time' AS FLOAT)) as avg_time
                FROM generated_images
                WHERE metadata::json->>'generation_time' IS NOT NULL
                GROUP BY model, sampler
                ORDER BY avg_time
            """
        }
        
        if analysis_type == "custom_query":
            query = custom_query
        else:
            query = queries.get(analysis_type, "SELECT 'Invalid analysis type'")
        
        try:
            result = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.execute(query).description]
            
            # Format as readable text
            output = f"Analysis: {analysis_type}\n"
            output += "-" * 50 + "\n"
            output += " | ".join(columns) + "\n"
            output += "-" * 50 + "\n"
            
            for row in result[:50]:  # Limit output
                output += " | ".join(str(v) for v in row) + "\n"
            
            return (output,)
            
        except Exception as e:
            return (f"Query error: {str(e)}",)


class DuckDBPromptDataset:
    """Load prompts from datasets for batch generation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_source": ([
                    "huggingface",
                    "local_parquet",
                    "local_jsonl",
                    "duckdb_table"
                ],),
                "source_path": ("STRING", {
                    "default": "lambdalabs/pokemon-blip-captions"
                }),
                "prompt_column": ("STRING", {"default": "text"}),
                "limit": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "shuffle": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    FUNCTION = "load_prompts"
    CATEGORY = "data/duckdb"
    
    def load_prompts(self, dataset_source, source_path, prompt_column, limit, shuffle):
        conn = duckdb.connect(':memory:')
        conn.execute("INSTALL httpfs; LOAD httpfs")
        
        # Build query based on source
        if dataset_source == "huggingface":
            # Try common HF dataset locations
            query = f"""
                SELECT {prompt_column} 
                FROM read_parquet('https://huggingface.co/datasets/{source_path}/resolve/main/train.parquet')
            """
        elif dataset_source == "local_parquet":
            query = f"SELECT {prompt_column} FROM read_parquet('{source_path}')"
        elif dataset_source == "local_jsonl":
            query = f"SELECT {prompt_column} FROM read_json_auto('{source_path}', format='newline_delimited')"
        else:  # duckdb_table
            query = f"SELECT {prompt_column} FROM {source_path}"
        
        # Add shuffle and limit
        if shuffle:
            query += " ORDER BY RANDOM()"
        query += f" LIMIT {limit}"
        
        try:
            prompts = [row[0] for row in conn.execute(query).fetchall()]
            return (prompts,)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return ([f"Error loading dataset: {str(e)}"],)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DuckDBLoader": DuckDBLoader,
    "DuckDBImageLogger": DuckDBImageLogger,
    "DuckDBAnalytics": DuckDBAnalytics,
    "DuckDBPromptDataset": DuckDBPromptDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DuckDBLoader": "DuckDB Query Loader",
    "DuckDBImageLogger": "DuckDB Image Logger",
    "DuckDBAnalytics": "DuckDB Analytics",
    "DuckDBPromptDataset": "DuckDB Prompt Dataset",
}