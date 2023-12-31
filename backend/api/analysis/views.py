from api.analysis.model_wrappers.tensorflow_model_wrapper import TensorFlowModelWrapper
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
import logging
import zipfile
from rest_framework.response import Response
from rest_framework import status
from api.analysis.tasks import process_image
from django.http import JsonResponse
from api.analysis.services.analysis_service import pipelines_registry
import zipfile
import os
import tempfile
import uuid
from api.analysis.repositories.minio_repository import MinioRepository

def generate_object_name(filename):
    unique_id = uuid.uuid4()
    return f"{unique_id}-{filename}"

def unzip_model(zip_file_path):
    # Create a temporary directory to extract the files
    extract_path = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    return extract_path

minio_repo = MinioRepository(
    endpoint_url='http://minio:9000',
    access_key='minio',
    secret_key='minio123',
    bucket_name='corneal-endothelium-analeyezer'
)

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
        
        
        uploaded_model = request.FILES.get('uploadedModel')
        if uploaded_model != None:
            model_file_name, model_file_extension = os.path.splitext(uploaded_model.name)
        else:
            model_file_name = None
            model_file_extension = None
        logger.debug(f"uploaded_model: {uploaded_model}")
        logger.debug(f"uploadedModel name is: {uploaded_model.name}")
        logger.debug(f"uploadedModel extension: {model_file_extension}")
        
        predictionsPath = request.data.get('predictionsPath')
        overlayedPath = request.data.get('overlayedPath')
        areaParameter = float(request.data.get('areaParameter'))
        generateLabelledImages = request.data.get('generateLabelledImages')
        logger.debug(f"raw generate labelled images: {generateLabelledImages}")
        generateLabelledImages = generateLabelledImages == 'true'
        logger.debug(f"not raw generate labelled images: {generateLabelledImages}")
        labelledImagesPath = request.data.get('labelledImagesPath')
        selectedModel = request.data.get('model')
        logger.debug(f"selectedModel: {selectedModel}")
        try:
            logger.debug(f"uploaded_model name: {uploaded_model.name}")
        except:
            pass
        
        #if selectedModel not in ['', 'deeplab']:
        #    return Response("No such model", status=status.HTTP_400_BAD_REQUEST)
        # check if model is in the registry of models
        if selectedModel not in pipelines_registry:
            logger.debug(f"model not in pipelines_registry")
            logger.debug(f"model: {selectedModel}")
            # validate uploaded model
            if uploaded_model == None:
                logger.info(f"uploaded_model is None")
                return Response("No such model", status=status.HTTP_400_BAD_REQUEST)
            logger.debug(f"unpacking uploaded_model: {uploaded_model}")
            object_name = generate_object_name(uploaded_model.name)
            
            minio_repo.upload_file_directly(uploaded_model, object_name)

            model = object_name
            logger.debug(f"model: {model}")
        

                # Clean up
                #shutil.rmtree(temp_dir)
            #uploaded_model = uploaded_model[0]
            #file_path = default_storage.save('tmp/' + uploaded_model.name, uploaded_model)
            #model_directory = unzip_model(file_path)
            #model = TensorFlowModelWrapper(model_directory)
        else:
            model = selectedModel
            
        logger.debug(f"selected model: {selectedModel}")
        for file in files:
            logger.debug(f"file: {file}")
        
        file_bytes_list = [file.read() for file in files]
        
        # Convert masks to bytes if they are not None
        mask_bytes_list = [mask.read() if mask else None for mask in masks]
        logger.debug(f'Length of mask_bytes_list: {len(mask_bytes_list)}')
        
        logger.debug(f'Creating task to process {len(files)} images')
        try:
            task = process_image.delay(file_bytes_list, file_paths, mask_bytes_list, predictionsPath, overlayedPath, areaParameter, generateLabelledImages, labelledImagesPath, model, model_file_extension)
        except Exception as e:
            logger.error(f'Error while processing images: {e}')
            return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
        
        processed_data, error = task.get()
        if error:
            logger.error(f'Error while processing images: {error}')
            return Response(error, status=status.HTTP_400_BAD_REQUEST)
        
        logger.debug(f'Task returned {len(processed_data)} images')
        
        json_response_data = [
            {
                'filename': filename,
                'data': data  # data is already in bytes
            } 
            for data, filename in processed_data
        ]
        
        return JsonResponse(json_response_data, safe=False)
    
    def get(self, request, format=None):
        model_names = list(pipelines_registry.keys())
        logger.debug(f"model names: {model_names}")
        return Response(model_names)
    