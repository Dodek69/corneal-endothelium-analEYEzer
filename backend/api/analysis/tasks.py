from api.analysis.services.analysis_service import AnalysisService
from celery import shared_task

import logging
logger = logging.getLogger(__name__)

@shared_task
def process_image(files, file_paths, masks, predictionsPath, overlayedPath, areaParameter, generateLabelledImages, labelledImagePath, model, model_file_extension):
    try:
        return AnalysisService.process(files, file_paths, masks, predictionsPath, overlayedPath, areaParameter, generateLabelledImages, labelledImagePath, model, model_file_extension)
    except Exception as e:
        logger.error(f"Error in process_image task: {str(e)}")
        return None, str(e)