#!/usr/bin/env python3
# src/utils/progress_tracking.py
"""
Progress Tracking Module for LlamaNote
Provides hierarchical, dynamic progress tracking using the rich library.
Supports pipeline-level, stage-level, and batch-level progress bars.
"""

import time
from typing import Dict, Optional, Any
from threading import RLock  # Changed from Lock to RLock for reentrant locking
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskID
)
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.console import Console, Group
from rich.table import Table

from .logger import get_logger_conf

logger = get_logger_conf(__name__)


class ProgressManager:
    """
    Manages hierarchical progress tracking for the LlamaNote pipeline.

    Supports three levels of progress bars:
    1. Pipeline-level: Overall weighted progress across all stages
    2. Stage-level: Progress within individual pipeline stages
    3. Batch-level: Progress for parallel batch processing (when batch_size > 1)

    Thread-safe for concurrent updates.
    """

    def __init__(self, stage_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ProgressManager.

        Args:
            stage_weights: Dictionary mapping stage names to their relative weights.
                          Used for calculating pipeline-level progress.
                          If None, uses equal weights for all stages.
        """
        self.stage_weights = stage_weights or {}
        self.total_weight = sum(self.stage_weights.values()) if self.stage_weights else 1.0
        self.completed_weight = 0.0

        # Rich components
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            expand=True
        )

        # Live display (don't create layout yet - create on start)
        self.live: Optional[Live] = None
        self.layout: Optional[Layout] = None

        # Task tracking
        self.pipeline_task: Optional[TaskID] = None
        self.stage_tasks: Dict[str, TaskID] = {}
        self.batch_tasks: Dict[int, TaskID] = {}

        # Thread safety - using RLock for reentrant locking
        self.lock = RLock()

        # Checkpoint notification state
        self.checkpoint_info: Optional[Dict[str, Any]] = None
        self.checkpoint_display_time: Optional[float] = None

        # Active status
        self.is_started = False

    def start(self):
        """Start the progress display."""
        with self.lock:
            if self.is_started:
                logger.warning("ProgressManager already started")
                return

            # Create and start Live display entirely within the lock
            self.live = Live(
                self.progress,
                console=self.console,
                refresh_per_second=10,
                transient=False
            )
            self.live.start()  # This starts the background refresh thread
            self.is_started = True
            logger.info("Progress tracking started")

    def stop(self):
        """Stop the progress display."""
        with self.lock:
            if not self.is_started:
                return
            
            if self.live:
                self.live.stop()
            
            self.live = None
            self.is_started = False
            logger.info("Progress tracking stopped")

    def add_pipeline_progress(self, total_weight: Optional[float] = None):
        """
        Add the top-level pipeline progress bar.

        Args:
            total_weight: Total weight for the pipeline (sum of all stage weights).
                         If None, uses self.total_weight.
        """
        # Check state INSIDE lock
        with self.lock:
            if self.pipeline_task is not None:
                logger.warning("Pipeline progress already exists")
                return
            if total_weight is not None:
                self.total_weight = total_weight

        # Call Rich library OUTSIDE lock
        task_id = self.progress.add_task(
            "[bold magenta]Pipeline Progress",
            total=100
        )

        # Store task ID INSIDE lock
        with self.lock:
            self.pipeline_task = task_id

        logger.debug("Added pipeline progress bar")

    def add_stage_progress(self, stage_name: str, total_items: int):
        """
        Add a stage-level progress bar.

        Args:
            stage_name: Name of the stage (e.g., "extract", "process", "audio")
            total_items: Total number of items/chunks in this stage
        """
        # Check if already exists INSIDE lock
        with self.lock:
            if stage_name in self.stage_tasks:
                logger.warning(f"Stage progress for '{stage_name}' already exists")
                return

        # Call Rich library OUTSIDE lock
        task_id = self.progress.add_task(
            f"[cyan]  ├─ {stage_name.capitalize()}",
            total=total_items
        )

        # Store task ID INSIDE lock
        with self.lock:
            self.stage_tasks[stage_name] = task_id

        logger.debug(f"Added stage progress: {stage_name} ({total_items} items)")

    def add_batch_progress(self, batch_id: int, total_items: int):
        """
        Add a batch-level progress bar (for parallel processing).

        Args:
            batch_id: Unique identifier for the batch
            total_items: Total number of items in this batch
        """
        # Check if already exists INSIDE lock
        with self.lock:
            if batch_id in self.batch_tasks:
                logger.warning(f"Batch progress for batch {batch_id} already exists")
                return

        # Call Rich library OUTSIDE lock
        is_last = False  # TODO: Detect if this is the last batch
        prefix = "└─" if is_last else "├─"
        task_id = self.progress.add_task(
            f"[yellow]  │  {prefix} Batch {batch_id}",
            total=total_items
        )

        # Store task ID INSIDE lock
        with self.lock:
            self.batch_tasks[batch_id] = task_id

        logger.debug(f"Added batch progress: batch {batch_id} ({total_items} items)")

    def update_pipeline(self, completed_weight: float):
        """
        Update the pipeline progress based on completed weight.

        Args:
            completed_weight: Total weight completed so far
        """
        logger.info(f"=== ProgressManager.update_pipeline ENTRY: weight={completed_weight} ===")
        # Get task ID and calculate percentage INSIDE lock
        logger.info(f"=== ProgressManager.update_pipeline: Acquiring lock ===")
        with self.lock:
            logger.info(f"=== ProgressManager.update_pipeline: Lock acquired ===")
            if self.pipeline_task is None:
                logger.warning("Pipeline progress not initialized")
                logger.info(f"=== ProgressManager.update_pipeline: pipeline_task is None, returning ===")
                return

            self.completed_weight = completed_weight
            # Convert weight to percentage (0-100)
            percentage = (completed_weight / self.total_weight * 100) if self.total_weight > 0 else 0
            percentage = min(100, max(0, percentage))  # Clamp to 0-100
            task_id = self.pipeline_task
            logger.info(f"=== ProgressManager.update_pipeline: Calculated percentage={percentage:.2f}% ===")

        logger.info(f"=== ProgressManager.update_pipeline: Lock released, calling progress.update ===")
        # Call Rich library OUTSIDE lock to avoid nested locking
        self.progress.update(task_id, completed=percentage)
        logger.info(f"=== ProgressManager.update_pipeline EXIT ===")


    def update_stage(self, stage_name: str, completed: int, advance: bool = False):
        """
        Update stage progress.

        Args:
            stage_name: Name of the stage to update
            completed: Number of items completed (or amount to advance if advance=True)
            advance: If True, increment by 'completed'. If False, set to 'completed'.
        """
        # Get task ID INSIDE lock
        with self.lock:
            if stage_name not in self.stage_tasks:
                logger.warning(f"Stage '{stage_name}' not found in progress tracking")
                return
            task_id = self.stage_tasks[stage_name]

        # Call Rich library OUTSIDE lock to avoid nested locking
        if advance:
            self.progress.update(task_id, advance=completed)
        else:
            self.progress.update(task_id, completed=completed)

    def update_batch(self, batch_id: int, completed: int, advance: bool = False):
        """
        Update batch progress.

        Args:
            batch_id: ID of the batch to update
            completed: Number of items completed (or amount to advance if advance=True)
            advance: If True, increment by 'completed'. If False, set to 'completed'.
        """
        # Get task ID INSIDE lock
        with self.lock:
            if batch_id not in self.batch_tasks:
                logger.warning(f"Batch {batch_id} not found in progress tracking")
                return
            task_id = self.batch_tasks[batch_id]

        # Call Rich library OUTSIDE lock to avoid nested locking
        if advance:
            self.progress.update(task_id, advance=completed)
        else:
            self.progress.update(task_id, completed=completed)

    def remove_stage(self, stage_name: str):
        """
        Remove a completed stage progress bar.

        Args:
            stage_name: Name of the stage to remove
        """
        logger.info(f"=== ProgressManager.remove_stage ENTRY: stage={stage_name} ===")
        # Get task ID and remove from dict INSIDE lock
        logger.info(f"=== ProgressManager.remove_stage: Acquiring lock ===")
        with self.lock:
            logger.info(f"=== ProgressManager.remove_stage: Lock acquired ===")
            if stage_name not in self.stage_tasks:
                logger.info(f"=== ProgressManager.remove_stage: {stage_name} not in stage_tasks, returning ===")
                return
            task_id = self.stage_tasks[stage_name]
            logger.info(f"=== ProgressManager.remove_stage: Deleting {stage_name} from stage_tasks ===")
            del self.stage_tasks[stage_name]
            logger.info(f"=== ProgressManager.remove_stage: Deleted ===")

        logger.info(f"=== ProgressManager.remove_stage: Lock released, calling progress.update ===")
        # Call Rich library OUTSIDE lock to avoid nested locking
        self.progress.update(task_id, visible=False)
        logger.info(f"=== ProgressManager.remove_stage: progress.update returned ===")
        logger.debug(f"Removed stage progress: {stage_name}")
        logger.info(f"=== ProgressManager.remove_stage EXIT: stage={stage_name} ===")

    def remove_batch(self, batch_id: int):
        """
        Remove a completed batch progress bar.

        Args:
            batch_id: ID of the batch to remove
        """
        # Get task ID and remove from dict INSIDE lock
        with self.lock:
            if batch_id not in self.batch_tasks:
                return
            task_id = self.batch_tasks[batch_id]
            del self.batch_tasks[batch_id]

        # Call Rich library OUTSIDE lock to avoid nested locking
        self.progress.update(task_id, visible=False)
        logger.debug(f"Removed batch progress: {batch_id}")

    def display_checkpoint_info(self, checkpoint_info: Dict[str, Any], display_duration: float = 5.0):
        """
        Display checkpoint save information above the progress bars.

        Args:
            checkpoint_info: Dictionary containing checkpoint metadata:
                - filename: Checkpoint filename
                - chunk_index: Chunk number saved
                - total_chunks: Total chunks
                - uncompressed_size_mb: Size before compression
                - compressed_size_mb: Size after compression
                - compression_ratio: Compression ratio (0-1)
                - hash: Checkpoint hash (first 8 chars)
                - timestamp: Save timestamp
            display_duration: How long to display the notification (seconds)
        """
        with self.lock:
            self.checkpoint_info = checkpoint_info
            self.checkpoint_display_time = time.time()

            # Build panel content
            filename = checkpoint_info.get('filename', 'Unknown')
            chunk_idx = checkpoint_info.get('chunk_index', 0)
            total_chunks = checkpoint_info.get('total_chunks', 0)
            uncomp_size = checkpoint_info.get('uncompressed_size_mb', 0)
            comp_size = checkpoint_info.get('compressed_size_mb', 0)
            comp_ratio = checkpoint_info.get('compression_ratio', 0) * 100
            ckpt_hash = checkpoint_info.get('hash', 'N/A')[:8]
            timestamp = checkpoint_info.get('timestamp', 'N/A')

            content = (
                f"[bold green]File:[/bold green] {filename}\n"
                f"[bold green]Chunk:[/bold green] {chunk_idx}/{total_chunks} │ "
                f"[bold green]Size:[/bold green] {uncomp_size:.1f}MB → {comp_size:.1f}MB ({comp_ratio:.1f}% compression)\n"
                f"[bold green]Hash:[/bold green] {ckpt_hash} │ "
                f"[bold green]Saved:[/bold green] {timestamp}"
            )

            panel = Panel(
                content,
                title="💾 [bold green]Checkpoint Saved[/bold green]",
                border_style="green",
                expand=False
            )

            # Update layout to show notification
            self.layout["notification"].size = 7
            self.layout["notification"].update(panel)

            logger.debug(f"Displaying checkpoint notification: {filename}")

            # Note: Auto-hide is handled in update cycle

    def hide_checkpoint_info(self):
        """Hide the checkpoint notification panel."""
        with self.lock:
            self.checkpoint_info = None
            self.checkpoint_display_time = None
            self.layout["notification"].size = 0
            self.layout["notification"].update("")

    def update_cycle(self):
        """
        Update cycle for automatic checkpoint notification hiding.
        Should be called periodically if auto-hiding is desired.
        """
        with self.lock:
            # Auto-hide checkpoint notification after duration
            if self.checkpoint_display_time is not None:
                elapsed = time.time() - self.checkpoint_display_time
                if elapsed >= 5.0:  # 5 second display duration
                    self.hide_checkpoint_info()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
