"""
UNINA-DLA: DLA-Native Object Detection Framework for NVIDIA Jetson Orin.
"""

from unina_dla.model.unina_dla import UNINA_DLA

__all__ = [
    "UNINA_DLA",
    "UNINADetectionPredictor",
    "UNINAYOLO",
    "build_ultralytics_metadata",
    "build_unina_deploy_model",
    "convert_checkpoint_to_ultralytics_pt",
    "embed_metadata_in_engine",
]

try:
    from .ultralytics_wrapper import (
        UNINAYOLO,
        UNINADetectionPredictor,
        build_ultralytics_metadata,
        build_unina_deploy_model,
        convert_checkpoint_to_ultralytics_pt,
        embed_metadata_in_engine,
    )
except ImportError:
    # Optional ultralytics integration; core PyTorch model remains operable standalone.
    UNINADetectionPredictor = None
    UNINAYOLO = None
    build_ultralytics_metadata = None
    build_unina_deploy_model = None
    convert_checkpoint_to_ultralytics_pt = None
    embed_metadata_in_engine = None
