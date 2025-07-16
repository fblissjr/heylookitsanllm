# src/heylook_llm/data_endpoint.py
"""Data query endpoint for DuckDB-powered analytics and data loading."""

from fastapi import HTTPException
from typing import Optional, List, Dict, Any
import duckdb
import logging
from pydantic import BaseModel


class DataQuery(BaseModel):
    """SQL query request."""
    query: str
    limit: Optional[int] = 1000
    format: Optional[str] = "json"  # json, csv, parquet


class DataLoadRequest(BaseModel):
    """Request to load data from various sources."""
    source_type: str  # 'huggingface', 'parquet', 'jsonl', 'csv'
    source_path: str
    table_name: Optional[str] = "loaded_data"
    options: Optional[Dict[str, Any]] = {}


def add_data_endpoints(app):
    """Add data query and loading endpoints to the FastAPI app."""
    
    # Initialize connection pool
    data_conn = duckdb.connect(':memory:')
    data_conn.execute("INSTALL httpfs")
    data_conn.execute("LOAD httpfs")
    
    @app.post("/v1/data/query")
    async def query_data(request: DataQuery):
        """
        Execute SQL query on loaded data or metrics.
        
        Example:
            POST /v1/data/query
            {
                "query": "SELECT model, AVG(total_time_ms) FROM metrics.request_logs GROUP BY model",
                "limit": 100
            }
        """
        try:
            # Safety check - only allow SELECT queries
            if not request.query.strip().upper().startswith('SELECT'):
                raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
            
            # Add limit if not present
            query = request.query
            if request.limit and 'LIMIT' not in query.upper():
                query += f" LIMIT {request.limit}"
            
            result = data_conn.execute(query)
            
            if request.format == "json":
                return {
                    "columns": [desc[0] for desc in result.description],
                    "data": result.fetchall(),
                    "row_count": len(result.fetchall())
                }
            else:
                # For other formats, return as downloadable file
                # Implementation depends on your needs
                pass
                
        except Exception as e:
            logging.error(f"Query failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/v1/data/load")
    async def load_data(request: DataLoadRequest):
        """
        Load data from various sources into DuckDB for querying.
        
        Example:
            POST /v1/data/load
            {
                "source_type": "huggingface",
                "source_path": "Open-Orca/OpenOrca",
                "table_name": "orca_data",
                "options": {"limit": 1000}
            }
        """
        try:
            if request.source_type == "huggingface":
                # Load from Hugging Face
                hf_url = f"https://huggingface.co/datasets/{request.source_path}/resolve/main/train.parquet"
                query = f"CREATE TABLE {request.table_name} AS SELECT * FROM read_parquet('{hf_url}')"
                if request.options.get('limit'):
                    query += f" LIMIT {request.options['limit']}"
                    
            elif request.source_type == "parquet":
                query = f"CREATE TABLE {request.table_name} AS SELECT * FROM read_parquet('{request.source_path}')"
                
            elif request.source_type == "jsonl":
                query = f"CREATE TABLE {request.table_name} AS SELECT * FROM read_json_auto('{request.source_path}', format='newline_delimited')"
                
            elif request.source_type == "csv":
                query = f"CREATE TABLE {request.table_name} AS SELECT * FROM read_csv_auto('{request.source_path}')"
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported source type: {request.source_type}")
            
            data_conn.execute(query)
            
            # Get table info
            info = data_conn.execute(f"SELECT COUNT(*) as row_count FROM {request.table_name}").fetchone()
            schema = data_conn.execute(f"DESCRIBE {request.table_name}").fetchall()
            
            return {
                "status": "success",
                "table_name": request.table_name,
                "row_count": info[0],
                "schema": [{"column": s[0], "type": s[1]} for s in schema]
            }
            
        except Exception as e:
            logging.error(f"Data load failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/v1/data/tables")
    async def list_tables():
        """List all available tables."""
        tables = data_conn.execute("SHOW TABLES").fetchall()
        return {
            "tables": [t[0] for t in tables]
        }
    
    @app.post("/v1/data/prepare_batch")
    async def prepare_batch(table_name: str, prompt_column: str = "prompt", batch_size: int = 10):
        """
        Prepare a batch of prompts from a loaded dataset for inference.
        
        This is perfect for benchmarking or batch processing workflows.
        """
        try:
            # Get batch of data
            batch_data = data_conn.execute(f"""
                SELECT * FROM {table_name} 
                LIMIT {batch_size}
            """).fetchall()
            
            columns = [desc[0] for desc in data_conn.execute(f"SELECT * FROM {table_name} LIMIT 0").description]
            
            # Format for inference
            requests = []
            for row in batch_data:
                row_dict = dict(zip(columns, row))
                requests.append({
                    "messages": [{"role": "user", "content": row_dict[prompt_column]}],
                    "max_tokens": 512,
                    "metadata": {k: v for k, v in row_dict.items() if k != prompt_column}
                })
            
            return {
                "requests": requests,
                "count": len(requests)
            }
            
        except Exception as e:
            logging.error(f"Batch preparation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))