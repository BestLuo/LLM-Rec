# utils/logger.py
import logging
import sys
import os
from datetime import datetime

def setup_logger(name='LGHRecLogger', log_file=None, level=logging.INFO):
    """
    Set up a logger.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Log file name.
        level (int): Log level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler_list = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handler_list.append(console_handler)

    # File handler
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a') # 'a' for append
        file_handler.setFormatter(formatter)
        handler_list.append(file_handler)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    for handler in handler_list:
        logger.addHandler(handler)
    
    logger.propagate = False 

    return logger

