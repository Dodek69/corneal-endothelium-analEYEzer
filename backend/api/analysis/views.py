from api.analysis.services.analysis_service import AnalysisService
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
import logging
import zipfile
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
import io

logger = logging.getLogger(__name__)

class AnalysisView(APIView):
    parser_classes = (MultiPartParser,)
    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist('files')
        file_paths = request.data.getlist('paths')

        processed_data, error = AnalysisService.process_images(files, file_paths)

        if error:
            return Response(error, status=status.HTTP_400_BAD_REQUEST)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for data, filename in processed_data:
                zip_file.writestr(filename, data)

        response = HttpResponse(buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=images.zip'
        return response