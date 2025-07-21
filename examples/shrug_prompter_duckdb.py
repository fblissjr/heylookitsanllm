#!/usr/bin/env python3
"""
DuckDB-powered shrug-prompter integration for advanced prompt engineering.
"""

import duckdb
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests


class ShrugPrompterDB:
    """
    DuckDB backend for shrug-prompter workflows.
    
    Features:
    - Prompt template library with versioning
    - A/B testing framework for prompts
    - Performance tracking across models
    - Automatic result analysis
    """
    
    def __init__(self, db_path: str = "shrug_prompter.duckdb"):
        self.conn = duckdb.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema for prompt engineering."""
        
        # Prompt templates with versioning
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                category VARCHAR,
                template TEXT,
                variables JSON,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp,
                parent_id VARCHAR,
                performance_score FLOAT,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Prompt execution history
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_runs (
                run_id VARCHAR PRIMARY KEY,
                template_id VARCHAR,
                model VARCHAR,
                provider VARCHAR,
                
                -- Inputs
                prompt_text TEXT,
                variables_used JSON,
                images JSON,
                
                -- Outputs
                response TEXT,
                response_metadata JSON,
                
                -- Performance
                total_time_ms INTEGER,
                tokens_used INTEGER,
                cost_estimate FLOAT,
                
                -- Quality metrics
                user_rating INTEGER,
                auto_score FLOAT,
                flags JSON,
                
                timestamp TIMESTAMP DEFAULT current_timestamp
            )
        """)
        
        # A/B test campaigns
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                hypothesis TEXT,
                variants JSON,  -- List of template_ids
                target_metric VARCHAR,
                sample_size INTEGER,
                status VARCHAR DEFAULT 'active',
                created_at TIMESTAMP DEFAULT current_timestamp,
                completed_at TIMESTAMP
            )
        """)
        
        # Model performance profiles
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_profiles (
                model VARCHAR PRIMARY KEY,
                provider VARCHAR,
                
                -- Performance characteristics
                avg_response_time_ms FLOAT,
                avg_tokens_per_second FLOAT,
                error_rate FLOAT,
                
                -- Best use cases
                strengths JSON,
                weaknesses JSON,
                optimal_prompt_length INTEGER,
                
                -- Cost
                cost_per_1k_tokens FLOAT,
                
                last_updated TIMESTAMP DEFAULT current_timestamp
            )
        """)
    
    def save_prompt_template(self, 
                           name: str, 
                           template: str, 
                           category: str = "general",
                           variables: Optional[Dict] = None) -> str:
        """Save a new prompt template or create a new version."""
        template_id = hashlib.md5(f"{name}:{template}".encode()).hexdigest()[:12]
        
        # Check if exists
        existing = self.conn.execute(
            "SELECT version FROM prompt_templates WHERE name = ? ORDER BY version DESC LIMIT 1",
            [name]
        ).fetchone()
        
        version = (existing[0] + 1) if existing else 1
        
        self.conn.execute("""
            INSERT INTO prompt_templates 
            (id, name, category, template, variables, version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [template_id, name, category, template, json.dumps(variables or {}), version])
        
        return template_id
    
    def get_best_prompt_for_task(self, category: str, model: str) -> Optional[Dict]:
        """Get the best performing prompt template for a task/model combination."""
        result = self.conn.execute("""
            WITH prompt_performance AS (
                SELECT 
                    pt.id,
                    pt.name,
                    pt.template,
                    pt.variables,
                    AVG(pr.auto_score) as avg_score,
                    COUNT(pr.run_id) as run_count
                FROM prompt_templates pt
                JOIN prompt_runs pr ON pt.id = pr.template_id
                WHERE pt.category = ?
                  AND pr.model = ?
                  AND pr.auto_score IS NOT NULL
                GROUP BY pt.id, pt.name, pt.template, pt.variables
                HAVING COUNT(pr.run_id) >= 5
            )
            SELECT * FROM prompt_performance
            ORDER BY avg_score DESC
            LIMIT 1
        """, [category, model]).fetchone()
        
        if result:
            return {
                'id': result[0],
                'name': result[1],
                'template': result[2],
                'variables': json.loads(result[3]),
                'avg_score': result[4],
                'run_count': result[5]
            }
        return None
    
    def log_prompt_run(self, 
                      template_id: str,
                      model: str,
                      prompt_text: str,
                      response: str,
                      metrics: Dict[str, Any]) -> str:
        """Log a prompt execution for analysis."""
        run_id = hashlib.md5(f"{template_id}:{datetime.now()}".encode()).hexdigest()[:12]
        
        self.conn.execute("""
            INSERT INTO prompt_runs
            (run_id, template_id, model, prompt_text, response, 
             total_time_ms, tokens_used, auto_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id, template_id, model, prompt_text, response,
            metrics.get('time_ms', 0),
            metrics.get('tokens', 0),
            metrics.get('score', None)
        ])
        
        # Update template usage count
        self.conn.execute("""
            UPDATE prompt_templates 
            SET usage_count = usage_count + 1
            WHERE id = ?
        """, [template_id])
        
        return run_id
    
    def analyze_prompt_performance(self, template_id: str) -> Dict[str, Any]:
        """Analyze performance of a prompt template across models."""
        
        # Performance by model
        model_perf = self.conn.execute("""
            SELECT 
                model,
                COUNT(*) as runs,
                AVG(auto_score) as avg_score,
                AVG(total_time_ms) as avg_time,
                AVG(tokens_used) as avg_tokens
            FROM prompt_runs
            WHERE template_id = ?
            GROUP BY model
            ORDER BY avg_score DESC
        """, [template_id]).fetchall()
        
        # Performance over time
        time_perf = self.conn.execute("""
            SELECT 
                DATE_TRUNC('day', timestamp) as day,
                AVG(auto_score) as avg_score,
                COUNT(*) as runs
            FROM prompt_runs
            WHERE template_id = ?
            GROUP BY 1
            ORDER BY 1
        """, [template_id]).fetchall()
        
        return {
            'by_model': [
                {
                    'model': row[0],
                    'runs': row[1],
                    'avg_score': row[2],
                    'avg_time_ms': row[3],
                    'avg_tokens': row[4]
                }
                for row in model_perf
            ],
            'over_time': [
                {
                    'date': row[0],
                    'avg_score': row[1],
                    'runs': row[2]
                }
                for row in time_perf
            ]
        }
    
    def create_ab_test(self, 
                      name: str,
                      hypothesis: str,
                      template_ids: List[str],
                      target_metric: str = "auto_score",
                      sample_size: int = 100) -> str:
        """Create an A/B test for prompt templates."""
        test_id = hashlib.md5(f"{name}:{datetime.now()}".encode()).hexdigest()[:12]
        
        self.conn.execute("""
            INSERT INTO ab_tests
            (test_id, name, hypothesis, variants, target_metric, sample_size)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [test_id, name, hypothesis, json.dumps(template_ids), target_metric, sample_size])
        
        return test_id
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get results of an A/B test."""
        test = self.conn.execute(
            "SELECT * FROM ab_tests WHERE test_id = ?", [test_id]
        ).fetchone()
        
        if not test:
            return None
        
        variants = json.loads(test[3])
        
        # Get performance for each variant
        results = []
        for variant_id in variants:
            stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as runs,
                    AVG(auto_score) as avg_score,
                    STDDEV(auto_score) as stddev_score,
                    AVG(total_time_ms) as avg_time,
                    AVG(tokens_used) as avg_tokens
                FROM prompt_runs
                WHERE template_id = ?
                  AND timestamp >= (SELECT created_at FROM ab_tests WHERE test_id = ?)
            """, [variant_id, test_id]).fetchone()
            
            results.append({
                'template_id': variant_id,
                'runs': stats[0],
                'avg_score': stats[1],
                'stddev_score': stats[2],
                'avg_time_ms': stats[3],
                'avg_tokens': stats[4]
            })
        
        # Determine winner (simple t-test would be better)
        winner = max(results, key=lambda x: x['avg_score'] if x['avg_score'] else 0)
        
        return {
            'test_id': test_id,
            'name': test[1],
            'hypothesis': test[2],
            'status': test[6],
            'results': results,
            'winner': winner
        }
    
    def suggest_prompt_improvements(self, template_id: str) -> List[str]:
        """Suggest improvements based on performance data."""
        suggestions = []
        
        # Get template info
        template = self.conn.execute(
            "SELECT template, category FROM prompt_templates WHERE id = ?", 
            [template_id]
        ).fetchone()
        
        if not template:
            return []
        
        # Analyze token usage
        token_stats = self.conn.execute("""
            SELECT 
                AVG(tokens_used) as avg_tokens,
                MIN(tokens_used) as min_tokens,
                MAX(tokens_used) as max_tokens
            FROM prompt_runs
            WHERE template_id = ?
        """, [template_id]).fetchone()
        
        if token_stats[2] > token_stats[1] * 2:
            suggestions.append(
                "High token variance detected. Consider adding length constraints."
            )
        
        # Check performance variance across models
        model_variance = self.conn.execute("""
            SELECT 
                STDDEV(avg_score) as score_variance
            FROM (
                SELECT model, AVG(auto_score) as avg_score
                FROM prompt_runs
                WHERE template_id = ?
                GROUP BY model
            )
        """, [template_id]).fetchone()
        
        if model_variance[0] > 0.2:
            suggestions.append(
                "High variance across models. Consider model-specific variants."
            )
        
        # Compare to category best
        category_best = self.conn.execute("""
            SELECT MAX(performance_score)
            FROM prompt_templates
            WHERE category = ?
        """, [template[1]]).fetchone()
        
        current_score = self.conn.execute("""
            SELECT AVG(auto_score)
            FROM prompt_runs
            WHERE template_id = ?
        """, [template_id]).fetchone()
        
        if category_best[0] and current_score[0] and current_score[0] < category_best[0] * 0.8:
            suggestions.append(
                f"Performance is {(1 - current_score[0]/category_best[0])*100:.0f}% "
                "below category best. Consider studying top performers."
            )
        
        return suggestions


# Example ComfyUI node that uses this
class ShrugPrompterDBNode:
    """ComfyUI node for database-driven prompt selection."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["best_for_task", "ab_test", "analyze", "save_template"],),
                "category": ("STRING", {"default": "general"}),
                "model": ("STRING", {"default": ""}),
            },
            "optional": {
                "template_name": ("STRING", {"default": ""}),
                "template_text": ("STRING", {"multiline": True}),
                "variables": ("STRING", {"default": "{}"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "DICT")
    FUNCTION = "execute"
    CATEGORY = "shrug-prompter/database"
    
    def __init__(self):
        self.db = ShrugPrompterDB()
    
    def execute(self, mode, category, model, **kwargs):
        if mode == "best_for_task":
            result = self.db.get_best_prompt_for_task(category, model)
            if result:
                return (result['template'], result)
            return ("No optimal prompt found", {})
        
        elif mode == "save_template":
            template_id = self.db.save_prompt_template(
                kwargs.get('template_name', 'unnamed'),
                kwargs.get('template_text', ''),
                category,
                json.loads(kwargs.get('variables', '{}'))
            )
            return (f"Saved as {template_id}", {"template_id": template_id})
        
        # ... other modes


if __name__ == "__main__":
    # Example usage
    db = ShrugPrompterDB()
    
    # Save a template
    template_id = db.save_prompt_template(
        "image_analysis_v1",
        "Analyze this image and describe: {focus_areas}",
        category="vision",
        variables={"focus_areas": "color, composition, style"}
    )
    
    # Log some runs
    db.log_prompt_run(
        template_id,
        "gpt-4-vision",
        "Analyze this image and describe: color, composition, style",
        "The image shows a vibrant sunset...",
        {"time_ms": 1200, "tokens": 150, "score": 0.85}
    )
    
    # Analyze performance
    perf = db.analyze_prompt_performance(template_id)
    print(f"Performance analysis: {perf}")
    
    # Get suggestions
    suggestions = db.suggest_prompt_improvements(template_id)
    print(f"Suggestions: {suggestions}")