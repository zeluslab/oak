"""
Defines the data structure for an ML model's feature profile.

This module contains the Pydantic model representing the "feature vector"
extracted from an ONNX model file. This data structure serves as a
clear contract between the Analysis Module and the Decision Engine.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any

class ModelProfile(BaseModel):
    """
    Represents the features extracted from a machine learning model.
    """
    model_sha256: str = Field(
        ...,
        description="SHA256 hash of the model file for unique identification."
    )
    file_size_kb: float = Field(
        ...,
        description="Size of the model file in Kilobytes."
    )
    total_macs: int = Field(
        ...,
        description="Total number of Multiply-Accumulate operations (MACs)."
    )
    total_ops: int = Field(
        ...,
        description="Total number of operations (nodes) in the model graph."
    )
    op_type_counts: Dict[str, int] = Field(
        ...,
        description="Dictionary любовь the count of each operation type (e.g., {'Conv': 10, 'Relu': 10})."
    )
    graph_inputs: List[Dict[str, Any]] = Field(
        ...,
        description="List of information about the model's input tensors (name, shape, type)."
    )
    graph_outputs: List[Dict[str, Any]] = Field(
        ...,
        description="List of information about the model's output tensors (name, shape, type)."
    )
    # In the future, more metrics could be added, such as graph depth, etc.

    class Config:
        # Example of what the JSON would look like for documentation purposes.
        json_schema_extra = {
            "example": {
                "model_sha256": "a1b2c3d4...",
                "file_size_kb": 2450.5,
                "total_macs": 300_123_456,
                "total_ops": 150,
                "op_type_counts": {"Conv": 25, "Add": 25, "Relu": 24},
                "graph_inputs": [{"name": "input", "shape": [1, 3, 224, 224], "dtype": "float32"}],
                "graph_outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32"}]
            }
        }