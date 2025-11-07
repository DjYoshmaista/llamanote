#!/usr/bin/env python3
"""
Test ProgressManager Class
Validates the new progress tracking system.
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.progress_tracking import ProgressManager
from rich.console import Console

console = Console()


def test_basic_usage():
    """Test basic ProgressManager usage"""
    console.print("\n[bold cyan]Test 1: Basic Usage[/bold cyan]")

    stage_weights = {
        "extract": 0.5,
        "preprocess": 0.5,
        "chunk": 0.3,
        "process": 45.0,
        "filter": 2.0,
        "format": 0.2,
        "save": 0.5,
        "audio": 35.0,
    }

    with ProgressManager(stage_weights=stage_weights) as pm:
        # Add pipeline progress
        pm.add_pipeline_progress()

        # Simulate extract stage
        pm.add_stage_progress("extract", total_items=1)
        time.sleep(0.5)
        pm.update_stage("extract", completed=1)
        pm.update_pipeline(stage_weights["extract"])
        time.sleep(0.5)
        pm.remove_stage("extract")

        # Simulate process stage (the heavy one)
        pm.add_stage_progress("process", total_items=50)
        for i in range(50):
            time.sleep(0.05)
            pm.update_stage("process", completed=1, advance=True)

            # Update pipeline based on weighted progress
            stage_progress = (i + 1) / 50
            completed_weight = stage_weights["extract"] + (stage_weights["process"] * stage_progress)
            pm.update_pipeline(completed_weight)

        pm.remove_stage("process")

    console.print("[green]✓ Basic usage test passed[/green]")


def test_with_batches():
    """Test ProgressManager with batch-level progress"""
    console.print("\n[bold cyan]Test 2: With Batch Progress[/bold cyan]")

    stage_weights = {"process": 45.0, "audio": 35.0}

    with ProgressManager(stage_weights=stage_weights) as pm:
        pm.add_pipeline_progress()

        # Add stage and batches
        pm.add_stage_progress("process", total_items=50)
        pm.add_batch_progress(batch_id=0, total_items=25)
        pm.add_batch_progress(batch_id=1, total_items=25)

        # Simulate parallel batch processing
        for i in range(50):
            time.sleep(0.03)

            # Update batches (alternate)
            if i < 25:
                pm.update_batch(0, completed=1, advance=True)
            else:
                pm.update_batch(1, completed=1, advance=True)

            # Update stage
            pm.update_stage("process", completed=1, advance=True)

            # Update pipeline
            stage_progress = (i + 1) / 50
            completed_weight = stage_weights["process"] * stage_progress
            pm.update_pipeline(completed_weight)

        pm.remove_batch(0)
        pm.remove_batch(1)
        pm.remove_stage("process")

    console.print("[green]✓ Batch progress test passed[/green]")


def test_checkpoint_notification():
    """Test checkpoint notification display"""
    console.print("\n[bold cyan]Test 3: Checkpoint Notification[/bold cyan]")

    stage_weights = {"process": 45.0}

    with ProgressManager(stage_weights=stage_weights) as pm:
        pm.add_pipeline_progress()
        pm.add_stage_progress("process", total_items=30)

        for i in range(30):
            time.sleep(0.1)
            pm.update_stage("process", completed=1, advance=True)

            # Update pipeline
            stage_progress = (i + 1) / 30
            completed_weight = stage_weights["process"] * stage_progress
            pm.update_pipeline(completed_weight)

            # Show checkpoint notification at intervals
            if i > 0 and i % 10 == 0:
                checkpoint_info = {
                    'filename': f"test_pdf-deepseek_r1_1.5b-ms_speecht5-chk{i:04d}-{i*3}pct.ckpt",
                    'chunk_index': i,
                    'total_chunks': 30,
                    'uncompressed_size_mb': 245.3,
                    'compressed_size_mb': 89.2,
                    'compression_ratio': 0.636,
                    'hash': 'a3f9c2d1e8b4567890abcdef',
                    'timestamp': '2025-11-07 14:32:18'
                }
                pm.display_checkpoint_info(checkpoint_info)

                # Wait to show notification
                time.sleep(2)

        pm.remove_stage("process")

    console.print("[green]✓ Checkpoint notification test passed[/green]")


def test_multi_stage_pipeline():
    """Test complete multi-stage pipeline"""
    console.print("\n[bold cyan]Test 4: Multi-Stage Pipeline[/bold cyan]")

    stage_weights = {
        "extract": 0.5,
        "preprocess": 0.5,
        "chunk": 0.3,
        "process": 45.0,
        "filter": 2.0,
        "audio": 35.0,
    }

    stages = [
        ("extract", 1),
        ("preprocess", 1),
        ("chunk", 1),
        ("process", 30),
        ("filter", 30),
        ("audio", 30),
    ]

    with ProgressManager(stage_weights=stage_weights) as pm:
        pm.add_pipeline_progress()

        completed_weight = 0.0

        for stage_name, stage_total in stages:
            pm.add_stage_progress(stage_name, total_items=stage_total)

            for i in range(stage_total):
                time.sleep(0.05)
                pm.update_stage(stage_name, completed=1, advance=True)

                # Update pipeline with weighted progress
                stage_progress = (i + 1) / stage_total
                current_stage_weight = stage_weights[stage_name] * stage_progress
                pm.update_pipeline(completed_weight + current_stage_weight)

            # Stage complete
            completed_weight += stage_weights[stage_name]
            pm.update_pipeline(completed_weight)
            pm.remove_stage(stage_name)

    console.print("[green]✓ Multi-stage pipeline test passed[/green]")


if __name__ == "__main__":
    console.print("[bold yellow]" + "="*60 + "[/bold yellow]")
    console.print("[bold yellow]ProgressManager Test Suite[/bold yellow]")
    console.print("[bold yellow]" + "="*60 + "[/bold yellow]")

    try:
        test_basic_usage()
        test_with_batches()
        test_checkpoint_notification()
        test_multi_stage_pipeline()

        console.print("\n[bold green]" + "="*60 + "[/bold green]")
        console.print("[bold green]✅ All ProgressManager tests passed![/bold green]")
        console.print("[bold green]" + "="*60 + "[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
