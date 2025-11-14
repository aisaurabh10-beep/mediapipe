import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(config):
    """Configures the root logger."""
    log_level = config.get('Logging', 'level', fallback='INFO').upper()
    log_file = config.get('Paths', 'log_file')
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove default handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logging.info("Logging configured.")
    return logger