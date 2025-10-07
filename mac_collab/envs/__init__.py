# This file can be used to register custom environments if needed in the future.
# For now, we are loading datasets directly from Hugging Face.

from .huggingface_dataset_env import HuggingFaceDatasetEnv

REGISTRY = {
    "huggingface_dataset_env": HuggingFaceDatasetEnv,
} 