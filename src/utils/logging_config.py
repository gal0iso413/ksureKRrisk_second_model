"""
Logging configuration for the second model.

This module provides standardized logging configuration for the second model,
including log formatting, handlers, and utility functions.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        log_format: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return setup_logger(name)

class LoggingContext:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize logging context.
        
        Args:
            logger: Logger instance
            level: Log level to set temporarily
        """
        self.logger = logger
        self.level = level
        self.old_level = logger.level
        
    def __enter__(self):
        """Set new log level when entering context."""
        self.logger.setLevel(self.level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log level when exiting context."""
        self.logger.setLevel(self.old_level) 