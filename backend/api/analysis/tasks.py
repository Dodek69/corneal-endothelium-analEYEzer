from api.analysis.services.analysis_service import AnalysisService
from api.celery import app

import logging
logger = logging.getLogger(__name__)

@app.task(bind=True)
def process_image(self, input_images, input_images_paths, input_masks, predictions_path, overlayed_path, area_per_pixel, generate_labelled_images, labelled_images_path, model, model_file_extension, pipeline_type, target_dimensions, downsampling_factor):
    try:
        return AnalysisService.process(self.request.id, input_images, input_images_paths, input_masks, predictions_path, overlayed_path, area_per_pixel, generate_labelled_images, labelled_images_path, model, model_file_extension, pipeline_type, target_dimensions, downsampling_factor)
    except Exception as e:
        logger.error(f"Error in process_image task: {str(e)}")
        return None, str(e)