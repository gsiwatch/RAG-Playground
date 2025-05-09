import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(
        f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
