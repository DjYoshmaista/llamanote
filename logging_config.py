"""
Logging Configuration Module
Provides centralized logging setup and utilities for the LlamaNote system
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import json
import traceback
from functools import wraps
import time

from config import LOGGING_CONFIG, LOG_DIR, SAVE_ERROR_CONTEXT

# Initialize logging configuration
logging.config.dictConfig(LOGGING_CONFIG)


class ContextLogger:
    """Enhanced logger with context tracking and performance monitoring"""
    
    def __init__(self, name: str, enable_performance: bool = True):
        self.logger = logging.getLogger(f"llamanote.{name}")
        self.enable_performance = enable_performance
        self.context: Dict[str, Any] = {}
        self.timers: Dict[str, float] = {}
        
    def set_context(self, **kwargs):
        """Set context variables that will be included in all log messages"""
        self.context.update(kwargs)
        
    def clear_context(self):
        """Clear all context variables"""
        self.context.clear()
        
    def _format_message(self, message: str) -> str:
        """Format message with context information"""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} | Context: [{context_str}]"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(self._format_message(message), **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(self._format_message(message), **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(self._format_message(message), **kwargs)
        
    def error(self, message: str, exc_info: bool = False, save_context: bool = True, **kwargs):
        """Log error message with context and optionally save error context"""
        formatted_message = self._format_message(message)
        self.logger.error(formatted_message, exc_info=exc_info, **kwargs)
        
        if save_context and SAVE_ERROR_CONTEXT:
            self._save_error_context(message, exc_info)
            
    def critical(self, message: str, exc_info: bool = True, save_context: bool = True, **kwargs):
        """Log critical message with context"""
        formatted_message = self._format_message(message)
        self.logger.critical(formatted_message, exc_info=exc_info, **kwargs)
        
        if save_context and SAVE_ERROR_CONTEXT:
            self._save_error_context(message, exc_info)
            
    def _save_error_context(self, message: str, include_traceback: bool):
        """Save error context to file for debugging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = LOG_DIR / f"error_context_{timestamp}.json"
            
            error_data = {
                "timestamp": timestamp,
                "message": message,
                "context": self.context,
                "traceback": traceback.format_exc() if include_traceback else None,
                "timers": self.timers
            }
            
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2, default=str)
                
            self.logger.debug(f"Error context saved to {error_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save error context: {e}")
            
    def start_timer(self, name: str):
        """Start a named timer for performance monitoring"""
        if self.enable_performance:
            self.timers[name] = time.time()
            self.debug(f"Timer '{name}' started")
            
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a named timer and return elapsed time"""
        if self.enable_performance and name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            self.info(f"Timer '{name}' stopped: {elapsed:.3f} seconds")
            return elapsed
        return None
    
    def log_performance(self, operation: str, start_time: float, **metrics):
        """Log performance metrics for an operation"""
        if self.enable_performance:
            elapsed = time.time() - start_time
            metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
            self.info(f"Performance: {operation} completed in {elapsed:.3f}s | {metrics_str}")


def get_logger(name: str, enable_performance: bool = True) -> ContextLogger:
    """Factory function to create a context logger"""
    return ContextLogger(name, enable_performance)


def log_execution_time(logger: Optional[ContextLogger] = None):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logger or get_logger(func.__module__)
            func_logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                func_logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                func_logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}",
                    exc_info=True,
                    save_context=True
                )
                raise
                
        return wrapper
    return decorator


def log_resource_usage(logger: Optional[ContextLogger] = None):
    """Decorator to log resource usage (memory, GPU) for a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            
            try:
                import torch
                import psutil
                
                # Get initial resource state
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    initial_gpu = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                else:
                    initial_gpu = 0
                    
                # Execute function
                result = func(*args, **kwargs)
                
                # Get final resource state
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = final_memory - initial_memory
                
                if torch.cuda.is_available():
                    final_gpu = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                    gpu_delta = final_gpu - initial_gpu
                    
                    func_logger.info(
                        f"{func.__name__} resource usage: "
                        f"RAM delta={memory_delta:.1f}MB, "
                        f"GPU delta={gpu_delta:.1f}MB, "
                        f"GPU peak={peak_gpu:.1f}MB"
                    )
                else:
                    func_logger.info(
                        f"{func.__name__} resource usage: "
                        f"RAM delta={memory_delta:.1f}MB"
                    )
                    
                return result
                
            except ImportError:
                # If psutil or torch not available, just run the function
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


class LoggingProgress:
    """Context manager for logging progress of long-running operations"""
    
    def __init__(self, logger: ContextLogger, operation: str, total: Optional[int] = None):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.current = 0
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}" + 
                        (f" (total: {self.total})" if self.total else ""))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {elapsed:.3f}s" +
                (f" ({self.current}/{self.total} items)" if self.total else "")
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {elapsed:.3f}s: {exc_val}",
                exc_info=True
            )
            
    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress counter"""
        self.current += increment
        
        if self.total:
            percentage = (self.current / self.total) * 100
            progress_msg = f"{self.operation}: {self.current}/{self.total} ({percentage:.1f}%)"
        else:
            progress_msg = f"{self.operation}: {self.current} items processed"
            
        if message:
            progress_msg += f" - {message}"
            
        self.logger.debug(progress_msg)


class MemoryMonitor:
    """Monitor and log memory usage throughout execution"""
    
    def __init__(self, logger: ContextLogger, threshold_mb: float = 1000):
        self.logger = logger
        self.threshold_mb = threshold_mb
        self.baseline_memory = None
        
    def start(self):
        """Start memory monitoring"""
        try:
            import psutil
            process = psutil.Process()
            self.baseline_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory monitoring started. Baseline: {self.baseline_memory:.1f}MB")
        except ImportError:
            self.logger.warning("psutil not available, memory monitoring disabled")
            
    def check(self, operation: str = "current operation"):
        """Check current memory usage and log if threshold exceeded"""
        if self.baseline_memory is None:
            return
            
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            delta = current_memory - self.baseline_memory
            
            if delta > self.threshold_mb:
                self.logger.warning(
                    f"High memory usage during {operation}: "
                    f"Current={current_memory:.1f}MB, "
                    f"Delta={delta:.1f}MB"
                )
                
                # Try to get GPU memory as well
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
                        self.logger.warning(
                            f"GPU memory: Allocated={gpu_memory:.1f}MB, "
                            f"Cached={gpu_cached:.1f}MB"
                        )
                except:
                    pass
                    
        except Exception as e:
            self.logger.debug(f"Memory check failed: {e}")


# Console output utilities for user interaction
class ConsoleOutput:
    """Utilities for formatted console output"""
    
    @staticmethod
    def header(text: str, width: int = 80):
        """Print a formatted header"""
        print("\n" + "=" * width)
        print(text.center(width))
        print("=" * width)
        
    @staticmethod
    def section(text: str, width: int = 60):
        """Print a section divider"""
        print("\n" + "-" * width)
        print(text)
        print("-" * width)
        
    @staticmethod
    def progress_bar(current: int, total: int, width: int = 50, prefix: str = "Progress"):
        """Print a progress bar"""
        percentage = (current / total) * 100 if total > 0 else 0
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        print(f"\r{prefix}: |{bar}| {percentage:.1f}% ({current}/{total})", end="", flush=True)
        if current == total:
            print()  # New line when complete
            
    @staticmethod
    def success(message: str):
        """Print a success message"""
        print(f"✅ {message}")
        
    @staticmethod
    def warning(message: str):
        """Print a warning message"""
        print(f"⚠️  {message}")
        
    @staticmethod
    def error(message: str):
        """Print an error message"""
        print(f"❌ {message}")
        
    @staticmethod
    def info(message: str):
        """Print an info message"""
        print(f"ℹ️  {message}")
