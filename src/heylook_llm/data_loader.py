# src/heylook_llm/data_loader.py
"""DuckDB-powered data loader for benchmarking and batch processing."""

import duckdb
from typing import List, Dict, Any, Optional, Generator
import logging
from pathlib import Path


class DataLoader:
    """
    Universal data loader using DuckDB for various formats.
    
    Supports:
    - Hugging Face datasets (via HTTP)
    - Local/remote Parquet files
    - JSONL files
    - CSV files
    - Even other databases
    """
    
    def __init__(self):
        self.conn = duckdb.connect(':memory:')
        
        # Install and load httpfs extension for remote data
        self.conn.execute("INSTALL httpfs")
        self.conn.execute("LOAD httpfs")
    
    def load_huggingface_dataset(self, dataset_name: str, split: str = 'train', limit: Optional[int] = None) -> duckdb.DuckDBPyRelation:
        """
        Load a Hugging Face dataset directly via DuckDB.
        
        Example:
            loader.load_huggingface_dataset('Open-Orca/OpenOrca', limit=1000)
        """
        # Hugging Face datasets are often available as parquet
        hf_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{split}.parquet"
        
        query = f"SELECT * FROM read_parquet('{hf_url}')"
        if limit:
            query += f" LIMIT {limit}"
            
        logging.info(f"Loading HF dataset: {dataset_name}/{split}")
        return self.conn.execute(query)
    
    def load_jsonl(self, path: str, columns: Optional[List[str]] = None) -> duckdb.DuckDBPyRelation:
        """Load JSONL file with optional column selection."""
        cols = ', '.join(columns) if columns else '*'
        return self.conn.execute(f"""
            SELECT {cols} 
            FROM read_json_auto('{path}', format='newline_delimited')
        """)
    
    def load_parquet(self, path: str, filters: Optional[str] = None) -> duckdb.DuckDBPyRelation:
        """Load parquet file(s) with optional SQL filters."""
        query = f"SELECT * FROM read_parquet('{path}')"
        if filters:
            query += f" WHERE {filters}"
        return self.conn.execute(query)
    
    def create_benchmark_dataset(self, name: str, prompts: List[str], expected_outputs: Optional[List[str]] = None):
        """Create a benchmark dataset for testing."""
        data = [{'prompt': p, 'expected': e} for p, e in zip(prompts, expected_outputs or [None]*len(prompts))]
        
        self.conn.execute(f"""
            CREATE TABLE {name} AS 
            SELECT * FROM (VALUES {','.join([f"('{d['prompt']}', '{d['expected'] or ''}')" for d in data])}) 
            AS t(prompt, expected)
        """)
    
    def prepare_inference_batch(self, 
                              source: str, 
                              prompt_column: str = 'prompt',
                              system_prompt: Optional[str] = None,
                              max_tokens: int = 512,
                              batch_size: int = 10) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Prepare batches for inference from any data source.
        
        Example:
            for batch in loader.prepare_inference_batch('dataset.parquet', prompt_column='question'):
                results = await process_batch(batch)
        """
        # Load data
        if source.endswith('.parquet'):
            data = self.load_parquet(source)
        elif source.endswith('.jsonl'):
            data = self.load_jsonl(source)
        else:
            # Assume it's a table name
            data = self.conn.execute(f"SELECT * FROM {source}")
        
        # Convert to inference format in batches
        rows = data.fetchall()
        columns = [desc[0] for desc in data.description]
        
        for i in range(0, len(rows), batch_size):
            batch = []
            for row in rows[i:i+batch_size]:
                row_dict = dict(zip(columns, row))
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": row_dict[prompt_column]})
                
                batch.append({
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "metadata": {k: v for k, v in row_dict.items() if k != prompt_column}
                })
            
            yield batch
    
    def analyze_inference_results(self, results_table: str) -> Dict[str, Any]:
        """Analyze inference results for performance and quality metrics."""
        return {
            'performance': self.conn.execute(f"""
                SELECT 
                    AVG(inference_time_ms) as avg_time_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY inference_time_ms) as p95_time_ms,
                    AVG(tokens_per_second) as avg_tps,
                    COUNT(*) as total_requests
                FROM {results_table}
            """).fetchone(),
            
            'token_stats': self.conn.execute(f"""
                SELECT 
                    AVG(prompt_tokens) as avg_prompt_tokens,
                    AVG(completion_tokens) as avg_completion_tokens,
                    SUM(prompt_tokens + completion_tokens) as total_tokens
                FROM {results_table}
            """).fetchone(),
            
            'errors': self.conn.execute(f"""
                SELECT 
                    error_type,
                    COUNT(*) as count
                FROM {results_table}
                WHERE error_type IS NOT NULL
                GROUP BY error_type
            """).fetchall()
        }
    
    def export_results(self, table: str, output_path: str, format: str = 'parquet'):
        """Export results to file."""
        if format == 'parquet':
            self.conn.execute(f"COPY {table} TO '{output_path}' (FORMAT PARQUET)")
        elif format == 'csv':
            self.conn.execute(f"COPY {table} TO '{output_path}' (FORMAT CSV, HEADER)")
        elif format == 'jsonl':
            self.conn.execute(f"COPY {table} TO '{output_path}' (FORMAT JSON, ARRAY false)")


# Example usage for benchmarking
def create_benchmark_script():
    """Example of using DataLoader for benchmarking."""
    loader = DataLoader()
    
    # Load a dataset (e.g., OpenOrca subset)
    print("Loading dataset...")
    dataset = loader.load_jsonl('orca_subset.jsonl', columns=['question', 'response'])
    
    # Or load from Hugging Face
    # dataset = loader.load_huggingface_dataset('tatsu-lab/alpaca', limit=100)
    
    # Prepare for inference
    print("Preparing batches...")
    for batch_num, batch in enumerate(loader.prepare_inference_batch('orca_subset.jsonl', 
                                                                    prompt_column='question',
                                                                    batch_size=5)):
        print(f"Processing batch {batch_num + 1} with {len(batch)} prompts...")
        
        # Here you would call your inference endpoint
        # results = await process_batch(batch)
        
        # Store results back in DuckDB for analysis
        # loader.conn.execute("INSERT INTO results VALUES ...")
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = loader.analyze_inference_results('results')
    print(f"Average inference time: {analysis['performance'][0]:.2f}ms")
    print(f"Average tokens/second: {analysis['performance'][2]:.1f}")


if __name__ == "__main__":
    create_benchmark_script()