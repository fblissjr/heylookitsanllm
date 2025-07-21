#!/usr/bin/env python3
"""
Simple CLI dashboard for heylook_llm metrics.

Usage:
    python scripts/metrics_dashboard.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.heylook_llm.metrics_db import MetricsDB
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from datetime import datetime


console = Console()


@click.command()
@click.option('--db-path', default='~/.heylook_llm/metrics.duckdb', help='Path to metrics database')
@click.option('--hours', default=24, help='Hours of history to show')
def dashboard(db_path: str, hours: int):
    """Display performance metrics dashboard."""
    db = MetricsDB(db_path)
    
    # Header
    console.print(Panel.fit(
        f"[bold blue]HeylookLLM Performance Dashboard[/bold blue]\n"
        f"Last {hours} hours | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="blue"
    ))
    
    # Model Performance Comparison
    console.print("\n[bold]Model Performance Comparison[/bold]")
    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Type", style="yellow")
    model_table.add_column("Requests", justify="right")
    model_table.add_column("Avg Time (ms)", justify="right")
    model_table.add_column("First Token (ms)", justify="right")
    model_table.add_column("Tokens/sec", justify="right")
    model_table.add_column("Errors", justify="right", style="red")
    
    for row in db.get_model_comparison().fetchall():
        model_table.add_row(
            row[0],  # model
            row[1],  # request_type
            str(row[2]),  # requests
            f"{row[3]:.0f}" if row[3] else "N/A",  # avg_time_ms
            f"{row[4]:.0f}" if row[4] else "N/A",  # avg_first_token_ms
            f"{row[5]:.1f}" if row[5] else "N/A",  # avg_tps
            str(row[6]) if row[6] > 0 else "0"  # errors
        )
    
    console.print(model_table)
    
    # Slow Requests
    console.print("\n[bold]Slowest Requests (>1s)[/bold]")
    slow_table = Table(show_header=True, header_style="bold red")
    slow_table.add_column("Time", style="dim")
    slow_table.add_column("Model", style="cyan")
    slow_table.add_column("Type", style="yellow")
    slow_table.add_column("Duration (s)", justify="right")
    slow_table.add_column("Tokens", justify="right")
    
    for row in db.get_slow_requests(1000, 10).fetchall():
        slow_table.add_row(
            row[0].strftime("%H:%M:%S"),  # timestamp
            row[1],  # model
            row[2],  # request_type
            f"{row[3]/1000:.1f}",  # total_time_ms -> seconds
            str(row[4]) if row[4] else "N/A"  # total_tokens
        )
    
    console.print(slow_table)
    
    # Recent Errors
    errors = db.get_error_analysis().fetchall()
    if errors:
        console.print("\n[bold]Recent Errors[/bold]")
        error_table = Table(show_header=True, header_style="bold red")
        error_table.add_column("Model", style="cyan")
        error_table.add_column("Error Type", style="yellow")
        error_table.add_column("Count", justify="right")
        error_table.add_column("Last Seen", style="dim")
        error_table.add_column("Sample", style="dim", max_width=50)
        
        for row in errors[:5]:
            error_table.add_row(
                row[0],  # model
                row[1],  # error_type
                str(row[2]),  # count
                row[3].strftime("%H:%M:%S"),  # last_seen
                row[4]  # sample_error
            )
        
        console.print(error_table)
    
    # SQL Query Examples
    console.print("\n[bold]Example Queries:[/bold]")
    console.print("""
    [dim]# Find requests with high token generation but slow overall time
    SELECT * FROM request_logs 
    WHERE tokens_per_second > 50 AND total_time_ms > 2000
    
    # Compare vision vs text performance by model
    SELECT model, request_type, 
           AVG(total_time_ms) as avg_ms,
           COUNT(*) as requests
    FROM request_logs
    GROUP BY model, request_type
    
    # Memory usage patterns
    SELECT time_bucket(INTERVAL '1 hour', timestamp) as hour,
           MAX(memory_used_gb) as peak_memory_gb
    FROM request_logs
    GROUP BY 1
    ORDER BY 1[/dim]
    """)


if __name__ == "__main__":
    dashboard()