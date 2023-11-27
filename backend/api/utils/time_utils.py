import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_timestamp_string():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d_%H-%M-%S-%f")

def start_timer():
    """Start timer to measure execution time."""
    global start_time
    start_time = time.time()
    
    
def stop_timer(text):
    """Stop timer and log execution time."""
    logger.info(f"Execution time of {text}: {time.time() - start_time} seconds")