# llamanote/utils/decorators.py
"""
Common Decorators for LlamaNote Enhanced
Includes decorators for logging execution time, resource usage, handling errors,
and managing checkpoints.
"""

import time
import sys
import traceback
from functools import wraps
from typing import Optional, Callable, Any

# Assuming logger setup is done elsewhere and we can get logger instances
from .logger import get_logger_conf, ConsoleOutput

# Attempt optional imports for resource usage
try:
    import torch
    TORCH_AVAILABLE_FOR_DECORATOR = True
except ImportError:
    TORCH_AVAILABLE_FOR_DECORATOR = False
    torch = None # Define for type hints

try:
    import psutil
    PSUTIL_AVAILABLE_FOR_DECORATOR = True
except ImportError:
    PSUTIL_AVAILABLE_FOR_DECORATOR = False
    psutil = None


def log_execution_time(logger_name: Optional[str] = None):
    """Decorator to log function/method execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger_conf(logger_name or func.__module__)
            start_time = time.time()
            func_qualname = f"{func.__module__}.{func.__name__}" # Get qualified name
            log.debug(f"Starting {func_qualname}...")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log.info(f"{func_qualname} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                # Include class name if it's a method
                log.error(
                    f"{func_qualname} failed after {elapsed:.3f}s: {e}",
                    exc_info=True,
                    save_context=True # Assuming ContextLogger handles this
                )
                raise
        return wrapper
    return decorator


def log_resource_usage(logger_name: Optional[str] = None):
    """Decorator to log RAM and GPU memory usage delta for a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PSUTIL_AVAILABLE_FOR_DECORATOR:
                # If psutil is not available, just run the function
                return func(*args, **kwargs)

            log = get_logger_conf(logger_name or func.__module__)
            func_qualname = f"{func.__module__}.{func.__name__}"

            # Initial state
            process = psutil.Process()
            initial_ram_mb = process.memory_info().rss / (1024 * 1024)
            initial_gpu_mb = 0
            gpu_peak_mb = 0
            if TORCH_AVAILABLE_FOR_DECORATOR and torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                    initial_gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception as e:
                     log.warning(f"Could not get initial GPU stats for {func_qualname}: {e}")


            # Execute function
            result = func(*args, **kwargs)

            # Final state
            final_ram_mb = process.memory_info().rss / (1024 * 1024)
            ram_delta_mb = final_ram_mb - initial_ram_mb
            gpu_delta_mb = 0

            log_msg = f"{func_qualname} resource usage: RAM Δ={ram_delta_mb:+.1f}MB (Current: {final_ram_mb:.1f}MB)"

            if TORCH_AVAILABLE_FOR_DECORATOR and torch.cuda.is_available():
                 try:
                    final_gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    gpu_delta_mb = final_gpu_mb - initial_gpu_mb
                    log_msg += f" | GPU Δ={gpu_delta_mb:+.1f}MB (Peak: {gpu_peak_mb:.1f}MB)"
                 except Exception as e:
                      log.warning(f"Could not get final GPU stats for {func_qualname}: {e}")


            log.info(log_msg)
            return result
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay_seconds: float = 1.0, backoff_factor: float = 2.0,
          exceptions_to_catch: tuple = (Exception,), logger_name: Optional[str] = None):
    """Decorator to automatically retry a function on specified exceptions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger_conf(logger_name or func.__module__)
            func_qualname = f"{func.__module__}.{func.__name__}"
            attempts = 0
            current_delay = delay_seconds

            while attempts < max_attempts:
                attempts += 1
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    log.warning(f"Attempt {attempts}/{max_attempts} failed for {func_qualname}: {e}")
                    if attempts == max_attempts:
                        log.error(f"{func_qualname} failed permanently after {attempts} attempts.", exc_info=True)
                        raise # Re-raise the last exception
                    else:
                        log.info(f"Retrying {func_qualname} in {current_delay:.2f} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor # Exponential backoff
        return wrapper
    return decorator


def with_checkpoint(stage_name: str, result_attr: Optional[str] = None):
    """
    Decorator to handle checkpoint save/load for pipeline stages (refactored).

    Assumes the decorated method is part of a class with attributes:
    - `config`: Containing `enable_checkpoints` boolean.
    - `checkpoint_manager`: An instance of CheckpointManager.
    - `logger`: A logger instance.
    - `stages_completed`: A list to track completed stages.

    Args:
        stage_name: The unique name for this stage's checkpoint.
        result_attr: If provided, store the result in `self.result_attr`.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            # Access attributes from the instance the method belongs to
            config = getattr(instance, 'config', None)
            manager = getattr(instance, 'checkpoint_manager', None)
            log = getattr(instance, 'logger', get_logger_conf(func.__module__)) # Fallback logger
            stages_completed_list = getattr(instance, 'stages_completed', [])

            if not all([config, manager]):
                log.warning(f"@with_checkpoint used on {func.__name__} without required instance attributes (config, checkpoint_manager). Skipping checkpoint.")
                return func(instance, *args, **kwargs)

            setattr(instance, 'current_stage', stage_name) # Track current stage
            ConsoleOutput.section(f"Stage: {stage_name.title()}")

            # Checkpoint Load
            if config.enable_checkpoints:
                checkpoint_data = manager.load(stage_name)
                if checkpoint_data is not None:
                    log.info(f"Loaded {stage_name} checkpoint")
                    if stage_name not in stages_completed_list:
                        stages_completed_list.append(stage_name)
                    if result_attr:
                        setattr(instance, result_attr, checkpoint_data)
                    return checkpoint_data # Return loaded data

            # Execute stage function
            result = func(instance, *args, **kwargs)

            # Checkpoint Save
            if config.enable_checkpoints and result is not None:
                manager.save(stage_name, result)

            if stage_name not in stages_completed_list:
                 stages_completed_list.append(stage_name)

            if result_attr:
                setattr(instance, result_attr, result)

            # Optional: Add memory check after stage execution
            if hasattr(instance, 'memory_monitor') and callable(getattr(instance.memory_monitor, 'check', None)):
                 instance.memory_monitor.check(f"after stage {stage_name}")


            return result
        return wrapper
    return decorator
