"""
Defines the data structure for a target hardware profile.
"""

from pydantic import BaseModel, Field
from typing import List, Literal

class HardwareProfile(BaseModel):
    """
    Represents the specifications of a target hardware device.
    """
    schema_version: str = Field("1.0", description="Version of the profile schema.")
    identifier: str = Field(..., description="Unique identifier for the hardware (e.g., 'esp32-s3').")
    vendor: str = Field(..., description="Hardware manufacturer (e.g., 'Espressif').")
    arch: str = Field(..., description="CPU architecture (e.g., 'Xtensa LX7', 'ARM Cortex-M4F').")
    cpu_freq_mhz: List[int] = Field(..., description="List of supported CPU frequencies in MHz (e.g., [160, 240]).")
    ram_total_kb: int = Field(..., description="Total available RAM in Kilobytes (KB).")
    accelerators: List[str] = Field(
        default_factory=list, 
        description="List of available hardware accelerators (e.g., 'vector_instructions', 'npu_kpu', 'gpu_xyz')."
    )
    supported_frameworks: List[Literal["tflite_micro", "onnx_runtime"]] = Field(
        ..., 
        description="Inference frameworks supported on the device (must be 'tflite_micro' or 'onnx_runtime')."
    )

    class Config:
        # Provides an example of how the JSON schema would look for documentation.
        json_schema_extra = {
            "example": {
                "schema_version": "1.0",
                "identifier": "esp32-s3",
                "vendor": "Espressif",
                "arch": "Xtensa LX7",
                "cpu_freq_mhz": [160, 240],
                "ram_total_kb": 512,
                "accelerators": ["vector_instructions", "fpu"],
                "supported_frameworks": ["tflite_micro"]
            }
        }