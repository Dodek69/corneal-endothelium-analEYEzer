from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
import logging
import zipfile
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
import io
from api.analysis.tasks import process_image

logger = logging.getLogger(__name__)

class AnalysisView(APIView):
    parser_classes = (MultiPartParser,)
    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist('files')
        file_paths = request.data.getlist('paths')
        masks = request.data.getlist('masks')
        logger.debug(f'Length of masks: {len(masks)}')
        for i, mask in enumerate(masks):
            print("====================================")
            print(files[i])
            print(mask)
            if mask == None:
                print("mask is none")
            if not mask:
                print("mask is empty")
            print("====================================")
            
            
        predictionsPath = request.data.get('predictionsPath')
        overlayedPath = request.data.get('overlayedPath')
        areaParameter = float(request.data.get('areaParameter'))
        generateLabelledImages = request.data.get('generateLabelledImages')
        logger.debug(f"raw generate labelled images: {generateLabelledImages}")
        if generateLabelledImages == 'true':
            generateLabelledImages = True
        else:
            generateLabelledImages = False
        logger.debug(f"not raw generate labelled images: {generateLabelledImages}")
        labelledImagesPath = request.data.get('labelledImagesPath')
        
        file_bytes_list = [file.read() for file in files]
        
        # Convert masks to bytes if they are not None
        mask_bytes_list = [mask.read() if mask else None for mask in masks]
        logger.debug(f'Length of mask_bytes_list: {len(mask_bytes_list)}')
        
        logger.debug(f'Creating task to process {len(files)} images')
        try:
            task = process_image.delay(file_bytes_list, file_paths, mask_bytes_list, predictionsPath, overlayedPath, areaParameter, generateLabelledImages, labelledImagesPath)
        except Exception as e:
            logger.error(f'Error while processing images: {e}')
            return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
        
        #while task.ready() is False:
        #    logger.debug(f'Task not ready')
        #    task.wait(timeout=5)
        processed_data, error = task.get()
        if error:
            logger.error(f'Error while processing images: {error}')
            return Response(error, status=status.HTTP_400_BAD_REQUEST)
        logger.debug(f'Task returned {len(processed_data)} images')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for data, filename in processed_data:
                zip_file.writestr(filename, data)

        response = HttpResponse(buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=images.zip'
        return response