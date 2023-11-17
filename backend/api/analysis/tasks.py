from api.analysis.services.analysis_service import AnalysisService
from celery import shared_task

import logging
logger = logging.getLogger(__name__)

@shared_task
def process_image(files, file_paths):
    try:
        return AnalysisService.process_images(files, file_paths)
    except Exception as e:
        logger.error(f"Error in process_image task: {str(e)}")
        return None, str(e)