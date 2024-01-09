from rest_framework import serializers
from api.analysis.registers import pipelines_registry, available_pipelines

class NullableFileField(serializers.FileField):
    def to_internal_value(self, data):
        if data == 'none' or data is None:
            return None
        return super().to_internal_value(data)

# Define a serializer
class AnalysisRequestSerializer(serializers.Serializer):
    input_images = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False
    )
    
    masks = serializers.ListField(
        child=NullableFileField(),
        allow_empty=True,
        required=False
    )

    input_paths = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=False
    )
    
    generate_labelled_images = serializers.BooleanField()
    predictions_output_path = serializers.CharField()
    overlayed_output_path = serializers.CharField()
    labelled_output_path = serializers.CharField(required=False)
    area_per_pixel = serializers.FloatField(min_value=1e-6)
    
    pipeline = serializers.CharField(required=False)
    custom_model = serializers.FileField(required=False)
    custom_model_pipeline = serializers.CharField(required=False)
    
    threshold = serializers.FloatField(required=False, min_value=0, max_value=1)
    target_height = serializers.IntegerField(required=False, min_value=1, max_value=1024)
    target_width = serializers.IntegerField(required=False, min_value=1, max_value=1024)
    downsampling_factor = serializers.IntegerField(required=False, min_value=1, max_value=512)

    def validate_predictions_output_path(self, value):
        if len(value) < 1 or len(value) > 255:
            raise serializers.ValidationError("Invalid predictions output path")
        return value
        
    def validate_overlayed_output_path(self, value):
        if len(value) < 1 or len(value) > 255:
            raise serializers.ValidationError("Invalid overlayed output path")
        return value
        
    def validate_labelled_output_path(self, value):
        if len(value) < 1 or len(value) > 255:
            raise serializers.ValidationError("Invalid labelled output path")
        return value

    def validate_downsampling_factor(self, value):
        if (value & (value - 1)) != 0:
            raise serializers.ValidationError("Downsampling factor must be a power of 2")
        return value

    def validate(self, data):
        pipeline = data.get('pipeline')
        custom_model = data.get('custom_model')
        custom_model_pipeline = data.get('custom_model_pipeline')
        target_height = data.get('target_height')
        target_width = data.get('target_width')
        downsampling_factor = data.get('downsampling_factor')
        generate_labelled_images = data.get('generate_labelled_images')
        labelled_output_path = data.get('labelled_output_path')
        threshold = data.get('threshold')
        
        if generate_labelled_images and not labelled_output_path:
            raise serializers.ValidationError({"labelled_output_path": "Output path for labelled images must be provided when generating labelled images"})
        
        if pipeline:
            if custom_model:
                raise serializers.ValidationError({"custom_model": "Cannot upload model when also pipeline is selected"})
            
            if pipeline not in pipelines_registry:
                raise serializers.ValidationError({"pipeline": "Invalid pipeline"})
        else:
            if not custom_model:
                raise serializers.ValidationError({"pipeline": "No model uploaded"})

            if not custom_model_pipeline:
                raise serializers.ValidationError({"pipeline": "No custom model pipeline selected"})
            
            if not custom_model_pipeline in available_pipelines:
                raise serializers.ValidationError({"custom_model_pipeline": "Invalid custom model pipeline"})
            
            if available_pipelines[custom_model_pipeline] == 'Resizing with padding' or available_pipelines[custom_model_pipeline] == 'Tiling':
                if not target_height or not target_width:
                    raise serializers.ValidationError({"target_height": "Target height and width must be provided"})
            
            elif available_pipelines[custom_model_pipeline] == 'Dynamic resizing with padding':
                if not downsampling_factor:
                    raise serializers.ValidationError({"downsampling_factor": "Downsampling factor must be provided"})
            if not threshold:
                raise serializers.ValidationError({"threshold": "Threshold must be provided"})
            
        input_images = data.get('input_images', [])
        masks = data.get('masks', [])
        input_paths = data.get('input_paths', [])
        
        if len(input_images) == 0:
            raise serializers.ValidationError({"input_images": "No input images"})
        
        if len(input_images) != len(masks) or len(input_images) != len(input_paths):
            raise serializers.ValidationError({"input_images": "The number of input images must match the number of masks and paths"})

        return data