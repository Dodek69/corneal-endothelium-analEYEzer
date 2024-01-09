from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
import logging
import zipfile
from rest_framework.response import Response
from rest_framework import status
from api.analysis.tasks import process_image
from django.http import JsonResponse
from api.analysis.registers import pipelines_registry
import zipfile
import os
import tempfile
import uuid
from api.analysis.repositories.minio_repository import MinioRepository
import base64
from django.core import signing
from api.celery import app
from rest_framework.permissions import AllowAny
from api.analysis.serializers import AnalysisRequestSerializer

def generate_object_name():
    unique_id = uuid.uuid4()
    return f"{unique_id}"

def jsend_success(data, status=status.HTTP_200_OK):
    return Response({
        "status": "success",
        "data": data
    }, status=status)
    
def jsend_fail(data, status=status.HTTP_400_BAD_REQUEST):
    return Response({
        "status": "fail",
        "data": data
    }, status=status)

def jsend_error(message = "Internal server error", status=500):
    return Response({
        "status": "error",
        "message": message
    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        serializer = AnalysisRequestSerializer(data=request.data, context={'request': request})
        if not serializer.is_valid():
            logger.debug(f"validation error: {serializer.errors}")
            return jsend_fail(serializer.errors)
        
        area_per_pixel = serializer.validated_data['area_per_pixel']
        generate_labelled_images = serializer.validated_data['generate_labelled_images']
        predictions_output_path = serializer.validated_data['predictions_output_path']
        overlayed_output_path = serializer.validated_data['overlayed_output_path']
            
        input_images = serializer.validated_data['input_images']
        input_paths = serializer.validated_data['input_paths']
        masks = serializer.validated_data['masks']

        """
        for i, mask in enumerate(masks):
            print("====================================")
            print(files[i])
            print(mask)
            if mask == None:
                print("mask is none")
            if not mask:
                print("mask is empty")
            print("====================================")
        """
        labelled_output_path = serializer.validated_data['labelled_output_path'] if generate_labelled_images else None
 
        if 'pipeline' in serializer.validated_data:
            pipeline = serializer.validated_data['pipeline']
            custom_model_object_name = None
            custom_model_extension = None
            custom_model_pipeline = None
            threshold = None
            target_height = None
            target_width = None
            downsampling_factor = None
        else:
            pipeline = None
            custom_model = serializer.validated_data['custom_model']
            custom_model_pipeline = serializer.validated_data['custom_model_pipeline']
            threshold = serializer.validated_data['threshold']
            target_height = serializer.validated_data['target_height']
            target_width = serializer.validated_data['target_width']
            downsampling_factor = serializer.validated_data['downsampling_factor']

            logger.debug(f"custom_model: {custom_model}")
            
            logger.debug(f"custom_model name is: {custom_model.name}")
            custom_model_name, custom_model_extension = os.path.splitext(custom_model.name)
            
            logger.debug(f"custom_model_name: {custom_model_name}")
            logger.debug(f"custom_model_extension: {custom_model_extension}")
            
            custom_model_object_name = generate_object_name()
            minio_repo.upload_file_directly(custom_model, custom_model_object_name)
        
        file_bytes_list = [file.read() for file in input_images]
        
        # Convert masks to bytes if they are not None
        mask_bytes_list = [mask.read() if mask else None for mask in masks]
        
        #logger.debug(f"file_paths: {file_paths}")
        #logger.debug(f"mask_bytes_list: {mask_bytes_list}")
        logger.debug(f"len(file_bytes_list): {len(file_bytes_list)}")
        logger.debug(f"len(mask_bytes_list): {len(mask_bytes_list)}")
        logger.debug(f"input_paths: {input_paths}")
        logger.debug(f"predictions_output_path: {predictions_output_path}")
        logger.debug(f"overlayed_output_path: {overlayed_output_path}")
        logger.debug(f"labelled_output_path: {labelled_output_path}")
        logger.debug(f"area_per_pixel: {area_per_pixel}")
        logger.debug(f"generate_labelled_images: {generate_labelled_images}")
        logger.debug(f"pipeline: {pipeline}")
        logger.debug(f"custom_model_object_name: {custom_model_object_name}")
        logger.debug(f"custom_model_extension: {custom_model_extension}")
        logger.debug(f"custom_model_pipeline: {custom_model_pipeline}")
        logger.debug(f"target_height: {target_height}")
        logger.debug(f"target_width: {target_width}")
        logger.debug(f"downsampling_factor: {downsampling_factor}")
        logger.debug(f"threshold: {threshold}")
        
        logger.debug(f'Creating task to process {len(input_images)} images')
        try:
            task = process_image.delay(file_bytes_list, input_paths, mask_bytes_list, predictions_output_path, overlayed_output_path, area_per_pixel, generate_labelled_images, labelled_output_path, pipeline, custom_model_object_name, custom_model_extension, custom_model_pipeline, (target_height, target_width), downsampling_factor, threshold)
        except Exception as e:
            logger.error(f'Error while starting task: {str(e)}')
            return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
        
        user_id = request.user.id
        logger.debug(f'started task id: {task.id}')
        logger.debug(f'type of started task id: {type(task.id)}')
        logger.debug(f'user id: {user_id}')
        logger.debug(f'type of user id: {type(user_id)}')
        try:
            signed_task_id = signing.dumps(f"{task.id}:{user_id}")
        except Exception as e:
            logger.error(f'Error while signing task ID: {str(e)}')
        
        #return Response(error, status=status.HTTP_400_BAD_REQUEST)
        logger.debug(f"task id: {signed_task_id}")
        response_data = {
            "polling_endpoint": f"http://localhost:8000/task-status/{signed_task_id}/",
            "polling_interval": 5
        }

        return jsend_success(response_data, status=status.HTTP_202_ACCEPTED)
        
    
class ModelsView(APIView):
    permission_classes = [AllowAny]
    def get(self, request, format=None):
        model_names = list(pipelines_registry.keys())
        logger.debug(f"model names: {model_names}")
        return Response(model_names)
    
class TaskStatusView(APIView):
    def get(self, request, task_id):
        try:
            # Attempt to load the original values from the signed ID
            original_value = signing.loads(task_id)
            task_id, original_user_id = original_value.split(':')
        except signing.BadSignature:
            logger.debug(f"bad signature")
            return JsonResponse({'status': 'error'}, status=status.HTTP_400_BAD_REQUEST)
        
        logger.debug(f"task_id: {task_id}")
        logger.debug(f"type of task_id: {type(task_id)}")
        logger.debug(f"original_user_id: {original_user_id}")
        logger.debug(f"type of original_user_id: {type(original_user_id)}")
        
        try:
            original_user_id = int(original_user_id)
        except ValueError:
             logger.debug("original_user_id is not an integer")
             return JsonResponse({'status': 'error'}, status=status.HTTP_400_BAD_REQUEST)
         
        if request.user.id != original_user_id:
            logger.debug(f"user id does not match original user id")
            return JsonResponse({'status': 'error'}, status=status.HTTP_400_BAD_REQUEST)
        
        task_status = app.AsyncResult(task_id)
        
        state = task_status.state
        logger.debug(f"task state: {state}")
        
        result = task_status.result
        if result:
            #might be [None, None], [None, error], [[data, filename], None], [[data, filename], None]
            logger.debug(f"task result exist")
        else:
            # is None
            logger.debug(f"task result does not exist")
            logger.debug(f"task result {result}")
            
        if state == 'SUCCESS':
            (processed_data, metrics), error = result
            if processed_data == None and error == None:
                try:
                    files = minio_repo.list_files()
                    task_files = [f for f in files if f.startswith(f"{task_id}\\")]
                    
                    logger.debug(f"found {len(task_files)} task_files")
                    logger.debug(f"task_files: {task_files}")
                        
                    task_files.sort()  # Sorting to maintain order
            
                    results = []
                    for file_name in task_files:
                        logger.debug(f"Downloading {file_name}...")
                        response = minio_repo.s3_client.get_object(Bucket=minio_repo.bucket_name, Key=file_name)
                        file_content = response['Body'].read()
                        encoded_data = base64.b64encode(file_content).decode('utf-8')
                        results.append({
                            'filename': file_name.split('\\')[-1],  # Just the filename
                            'data': encoded_data  # Base64 encoded data
                        })
                        
                        logger.debug(file_name.split('\\')[-1])
                        
                            
                    if len(results) != len(task_files):
                        logger.error(f"results length does not match task_files length")
                        return JsonResponse({'status': 'error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                except Exception as e:
                    logger.error(f"Error while getting result from minio: {str(e)}")
                    return Response({"error": "error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                try:
                    logger.debug(f"getting result from task_status.result")
                    
                    if error:
                        logger.debug(f"recieved error from task_status.result: {error}")
                        return Response({"error": "error while processing images"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
                    logger.debug(f"error not exist")
                    
                    if not processed_data:
                        logger.debug(f"both processed_data and error does not exist")
                        return Response({"error": "error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
                    logger.debug(f"processed_data exist")
                    
                    results = [
                    {
                        'filename': filename,
                        'data': data
                    } 
                    
                    for data, filename in processed_data]
                except Exception as e:
                    logger.error(f"Error while getting result from task_status.result: {str(e)}")
                    return Response({"error": "error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            metrics_string = [
            {
                'num_labels': num_labels,
                'area': area,
                'cell_density': cell_density,
                'std_areas': std_areas,
                'mean_areas': mean_areas,
                'coefficient_value': coefficient_value,
                'num_hexagonal': num_hexagonal,
                'hexagonal_cell_ratio': hexagonal_cell_ratio,
            }
            
            for num_labels, area, cell_density, std_areas, mean_areas, coefficient_value, num_hexagonal, hexagonal_cell_ratio in metrics]
            
            return JsonResponse({'status': 'completed', 'results': results, 'metrics': metrics_string})
        
        elif state == 'PENDING':
            logger.debug(f"task state is pending")
            return Response({"task_id": task_id, "status": state})
        elif state == 'FAILURE':
            logger.debug(f"task state is failure")
            return Response({"task_id": task_id, "status": state}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        elif state == 'REVOKED':
            logger.debug(f"task state is revoked")
            return Response({"task_id": task_id, "status": state}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        elif state == 'STARTED':
            logger.debug(f"task state is started")
            return Response({"task_id": task_id, "status": state})
        elif state == 'RETRY':
            logger.debug(f"task state is retry")
            return Response({"task_id": task_id, "status": state})
        elif state == 'RECEIVED':
            logger.debug(f"task state is received")
            return Response({"task_id": task_id, "status": state})
        else:
            return Response({"error": "error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)