from api.analysis.model_wrappers.tensorflow_model_wrapper import TensorFlowModelWrapper
from api.analysis.pipelines.tiling_pipeline import TilingPipeline
from api.analysis.pipelines.resize_with_pad_pipeline import ResizeWithPadPipeline
from api.analysis.pipelines.dynamic_resize_with_pad_pipeline import DynamicResizeWithPadPipeline
from api.analysis.pipelines.variable_shape_pipeline import VariableShapePipeline
from api.analysis.model_wrappers.binarization_wrapper import BinarizationWrapper
from api.analysis.model_wrappers.ragged_binarization_wrapper import RaggedBinarizationWrapper

model_paths = {
    'custom': 'api/analysis/models/custom',
    'sm_unet-fixed': 'api/analysis/models/cea_sm_unet_fixed.keras',
    'sm_unet-variable': 'api/analysis/models/cea_sm_unet_variable.keras',
}

models = {
    'custom': BinarizationWrapper(TensorFlowModelWrapper(model_paths['custom']).load_model(), threshold=0.1),
    'sm_unet-fixed': BinarizationWrapper(TensorFlowModelWrapper(model_paths['sm_unet-fixed']).load_model(), threshold=0.15),
    'sm_unet-variable': BinarizationWrapper(TensorFlowModelWrapper(model_paths['sm_unet-variable']).load_model(), threshold=0.15),
    'sm-unet-variable-ragged': RaggedBinarizationWrapper(TensorFlowModelWrapper(model_paths['sm_unet-variable']).load_model(), threshold=0.15),
}

pipelines_registry = {
    'custom-resize_with_pad 128x128': ResizeWithPadPipeline(models['custom'], target_dimensions=(128, 128)),
    'custom-tiling 128x128': TilingPipeline(models['custom'], patch_size=(128, 128, 3)),
    
    'sm_unet-resize_with_pad 512x512': ResizeWithPadPipeline(models['sm_unet-fixed'], target_dimensions=(512, 512)),
    'sm_unet-tiling 512x512': TilingPipeline(models['sm_unet-fixed'], patch_size=(512, 512, 3)),

    'sm_unet-variable-resize_with_pad 1024x1024': ResizeWithPadPipeline(models['sm_unet-variable'], target_dimensions=(1024, 1024)),
    'sm_unet-variable-resize_with_pad 512x512': ResizeWithPadPipeline(models['sm_unet-variable'], target_dimensions=(512, 512)),
    'sm_unet-variable-resize_with_pad 128x128': ResizeWithPadPipeline(models['sm_unet-variable'], target_dimensions=(128, 128)),
    'sm_unet-variable-dynamic_resize_with_pad': DynamicResizeWithPadPipeline(models['sm_unet-variable'], downsampling_factor=64),
    'sm_unet-variable-variable': VariableShapePipeline(models['sm-unet-variable-ragged'], downsampling_factor=64),
    'sm_unet-variable-tiling 128x128': TilingPipeline(models['sm_unet-variable'], patch_size=(128, 128, 3)),
    'sm_unet-variable-tiling 512x512': TilingPipeline(models['sm_unet-variable'], patch_size=(512, 512, 3)),
    'sm_unet-variable-tiling 1024x1024': TilingPipeline(models['sm_unet-variable'], patch_size=(1024, 1024, 3)),
}

available_pipelines = {
    "Tiling": TilingPipeline,
    "Resizing with padding": ResizeWithPadPipeline,
    "Dynamic resizing with padding": DynamicResizeWithPadPipeline
}