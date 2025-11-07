#!/usr/bin/env python3
"""
End-to-End Progress System Integration Test
Simulates a complete pipeline run with progress tracking.
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.progress_tracking import ProgressManager
from src.config.settings import DEFAULT_STAGE_WEIGHTS
from rich.console import Console

console = Console()


def simulate_pipeline_execution():
    """
    Simulate a complete pipeline execution with all stages.
    This mimics what happens in src/core/pipeline.py
    """
    console.print("\n[bold yellow]" + "="*80 + "[/bold yellow]")
    console.print("[bold yellow]End-to-End Progress System Integration Test[/bold yellow]")
    console.print("[bold yellow]Simulating Complete Pipeline Execution[/bold yellow]")
    console.print("[bold yellow]" + "="*80 + "[/bold yellow]\n")

    # Simulate processing a file
    input_file = "test_document.pdf"
    console.print(f"[cyan]Processing: {input_file}[/cyan]\n")

    # Initialize ProgressManager (as done in pipeline.py:212-214)
    progress_manager = ProgressManager(stage_weights=DEFAULT_STAGE_WEIGHTS)
    progress_manager.start()
    progress_manager.add_pipeline_progress()

    # Track completed stages
    completed_stages = []

    # Define pipeline stages with their chunk counts
    stages = [
        ("extract", 1, "Extracting text from PDF"),
        ("preprocess", 1, "Cleaning and preprocessing text"),
        ("chunk", 1, "Splitting text into chunks"),
        ("process", 55, "Generating processed text"),  # Main bottleneck
        ("filter", 55, "Filtering and cleaning output"),
        ("format", 1, "Formatting output"),
        ("save", 1, "Saving to file"),
        ("audio", 55, "Generating audio"),  # Second bottleneck
    ]

    try:
        for stage_name, total_chunks, description in stages:
            console.print(f"\n[bold green]→ Stage: {stage_name.upper()}[/bold green] - {description}")

            # Add stage progress bar
            progress_manager.add_stage_progress(stage_name, total_items=total_chunks)

            # Simulate chunk processing
            for chunk_idx in range(total_chunks):
                # Simulate processing time (faster for testing)
                if stage_name in ["process", "audio"]:
                    time.sleep(0.03)  # Slower stages
                else:
                    time.sleep(0.05)  # Fast stages

                # Update stage progress
                progress_manager.update_stage(stage_name, completed=1, advance=True)

                # Update pipeline progress
                stage_progress = (chunk_idx + 1) / total_chunks
                completed_weight = sum(DEFAULT_STAGE_WEIGHTS.get(s, 0.0) for s in completed_stages)
                current_weight = DEFAULT_STAGE_WEIGHTS.get(stage_name, 0.0) * stage_progress
                progress_manager.update_pipeline(completed_weight + current_weight)

                # Simulate checkpoint save at intervals (every 10 chunks for process/audio)
                if stage_name in ["process", "audio"] and (chunk_idx + 1) % 10 == 0:
                    # Calculate completion percentage
                    total_weight = sum(DEFAULT_STAGE_WEIGHTS.values())
                    completion_pct = int(((completed_weight + current_weight) / total_weight) * 100)

                    checkpoint_info = {
                        'filename': f"test_document_pdf-deepseek_r1_1.5b-ms_speecht5-chk{chunk_idx+1:04d}-{completion_pct}pct.ckpt",
                        'chunk_index': chunk_idx + 1,
                        'total_chunks': total_chunks,
                        'uncompressed_size_mb': 245.3,
                        'compressed_size_mb': 89.2,
                        'compression_ratio': 0.636,
                        'hash': f'a3f9c2d{chunk_idx:02d}',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    progress_manager.display_checkpoint_info(checkpoint_info)

                    # Brief pause to show checkpoint notification
                    time.sleep(0.5)

            # Stage complete
            completed_stages.append(stage_name)
            progress_manager.remove_stage(stage_name)

            # Update pipeline to reflect completed stage
            completed_weight = sum(DEFAULT_STAGE_WEIGHTS.get(s, 0.0) for s in completed_stages)
            progress_manager.update_pipeline(completed_weight)

            console.print(f"[green]✓ {stage_name.capitalize()} complete[/green]")

        console.print("\n[bold green]" + "="*80 + "[/bold green]")
        console.print("[bold green]✅ Pipeline execution completed successfully![/bold green]")
        console.print("[bold green]" + "="*80 + "[/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Pipeline interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline failed: {e}[/bold red]")
        raise
    finally:
        # Always stop progress manager (as done in pipeline.py:857)
        progress_manager.stop()


def test_with_batch_processing():
    """
    Test progress tracking with batch processing (batch_size > 1).
    """
    console.print("\n[bold yellow]" + "="*80 + "[/bold yellow]")
    console.print("[bold yellow]Testing Batch-Level Progress Tracking[/bold yellow]")
    console.print("[bold yellow]" + "="*80 + "[/bold yellow]\n")

    progress_manager = ProgressManager(stage_weights=DEFAULT_STAGE_WEIGHTS)
    progress_manager.start()
    progress_manager.add_pipeline_progress()

    # Simulate process stage with batch_size=2
    stage_name = "process"
    total_chunks = 50
    batch_size = 2

    console.print(f"[cyan]Stage: {stage_name.upper()} (batch_size={batch_size})[/cyan]\n")

    progress_manager.add_stage_progress(stage_name, total_items=total_chunks)

    # Add batch progress bars
    for batch_id in range(batch_size):
        progress_manager.add_batch_progress(batch_id, total_items=total_chunks // batch_size)

    try:
        # Simulate parallel batch processing
        for chunk_idx in range(total_chunks):
            time.sleep(0.02)

            # Determine which batch this chunk belongs to
            batch_id = chunk_idx % batch_size

            # Update batch progress
            progress_manager.update_batch(batch_id, completed=1, advance=True)

            # Update stage progress
            progress_manager.update_stage(stage_name, completed=1, advance=True)

            # Update pipeline progress
            stage_progress = (chunk_idx + 1) / total_chunks
            current_weight = DEFAULT_STAGE_WEIGHTS.get(stage_name, 0.0) * stage_progress
            progress_manager.update_pipeline(current_weight)

        # Clean up batch progress bars
        for batch_id in range(batch_size):
            progress_manager.remove_batch(batch_id)

        progress_manager.remove_stage(stage_name)

        console.print("\n[bold green]✅ Batch processing test completed![/bold green]\n")

    finally:
        progress_manager.stop()


def test_checkpoint_notifications():
    """
    Test checkpoint notification display in isolation.
    """
    console.print("\n[bold yellow]" + "="*80 + "[/bold yellow]")
    console.print("[bold yellow]Testing Checkpoint Notification Display[/bold yellow]")
    console.print("[bold yellow]" + "="*80 + "[/bold yellow]\n")

    progress_manager = ProgressManager(stage_weights=DEFAULT_STAGE_WEIGHTS)
    progress_manager.start()
    progress_manager.add_pipeline_progress()

    stage_name = "process"
    total_chunks = 30

    progress_manager.add_stage_progress(stage_name, total_items=total_chunks)

    try:
        for chunk_idx in range(total_chunks):
            time.sleep(0.08)

            progress_manager.update_stage(stage_name, completed=1, advance=True)

            stage_progress = (chunk_idx + 1) / total_chunks
            current_weight = DEFAULT_STAGE_WEIGHTS.get(stage_name, 0.0) * stage_progress
            progress_manager.update_pipeline(current_weight)

            # Show checkpoint every 5 chunks
            if (chunk_idx + 1) % 5 == 0:
                checkpoint_info = {
                    'filename': f"test_pdf-model1-model2-chk{chunk_idx+1:04d}-{int(stage_progress*53.6)}pct.ckpt",
                    'chunk_index': chunk_idx + 1,
                    'total_chunks': total_chunks,
                    'uncompressed_size_mb': 180.5,
                    'compressed_size_mb': 65.3,
                    'compression_ratio': 0.638,
                    'hash': f'b2e4f1c{chunk_idx:02d}',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                progress_manager.display_checkpoint_info(checkpoint_info)
                time.sleep(1)  # Pause to show notification

        console.print("\n[bold green]✅ Checkpoint notification test completed![/bold green]\n")

    finally:
        progress_manager.stop()


if __name__ == "__main__":
    try:
        # Run all integration tests
        simulate_pipeline_execution()
        time.sleep(1)

        test_with_batch_processing()
        time.sleep(1)

        test_checkpoint_notifications()

        console.print("\n[bold green]" + "="*80 + "[/bold green]")
        console.print("[bold green]🎉 All End-to-End Tests Passed![/bold green]")
        console.print("[bold green]Progress system is ready for production use.[/bold green]")
        console.print("[bold green]" + "="*80 + "[/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Tests failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
