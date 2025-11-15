# import logging
# import sys
# from logging.handlers import RotatingFileHandler

# def setup_logging(config):
#     """Configures the root logger."""
#     log_level = config.get('Logging', 'level', fallback='INFO').upper()
#     log_file = config.get('Paths', 'log_file')
    
#     logger = logging.getLogger()
#     logger.setLevel(log_level)
    
#     # Remove default handlers
#     for h in logger.handlers[:]:
#         logger.removeHandler(h)

#     # File handler - logs everything at configured level
#     file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
#     file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(file_formatter)
#     logger.addHandler(file_handler)
    
#     # Console handler - ONLY show WARNING and above (attendance marks + errors)
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.WARNING)  # Only WARNING, ERROR, CRITICAL
#     console_formatter = logging.Formatter('%(levelname)s: %(message)s')
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(console_handler)
    
#     logging.info("Logging configured - console shows WARNING+ only")
#     return logger

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

    # File handler - logs everything at configured level
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - Show INFO and above (attendance marks + errors, but not DEBUG/WARNING spam)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Show INFO, ERROR, CRITICAL
    
    # Filter out specific WARNING messages we don't want
    class RejectWarningFilter(logging.Filter):
        def filter(self, record):
            # Block WARNING messages about rejections/validation
            if record.levelno == logging.WARNING:
                if 'failed geometric validation' in record.getMessage():
                    return False
                if 'REJECTED' in record.getMessage():
                    return False
            return True
    
    console_handler.addFilter(RejectWarningFilter())
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logging.info("Logging configured - console shows attendance logs, hides rejection spam")
    return logger