"""
Core logic for analyzing an ONNX model file using ONNX Runtime.
"""

import onnx
import hashlib
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import Counter
from typing import Dict, Any
import os
import warnings

from oak.analysis.model_profile import ModelProfile

class ModelAnalysisError(Exception):
    """Base exception for errors during model analysis."""
    pass

def _calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def _calculate_macs_from_profile(prof_file: str) -> int:
    """
    Calculates the total number of MACs (Multiply-Accumulate operations) 
    from an ONNX Runtime JSON profile file by analyzing tensor shapes for each operation.
    """
    total_macs = 0
    try: # More granular try-except for file processing
        with open(prof_file, 'r') as f:
            profile_data = json.load(f)
        
        for event in profile_data:
            if event.get('cat') == 'Node' and 'args' in event:
                args = event.get('args', {})
                op_name = args.get('op_name')

                try:
                    if op_name == 'Conv':
                        # MACs formula for Conv: C_in * K_h * K_w * C_out * H_out * W_out
                        # Weights format (weights): [C_out, C_in/groups, K_h, K_w]
                        weight_shape = args['input_type_shape'][1]['float']
                        c_out, _, k_h, k_w = weight_shape
                        # Output format (output): [N, C_out, H_out, W_out]
                        output_shape = args['output_type_shape'][0]['float']
                        _, _, h_out, w_out = output_shape
                        # Actual C_in comes from the input tensor
                        c_in = args['input_type_shape'][0]['float'][1]
                        
                        # Ignoring 'groups' for a standard approximation
                        op_macs = c_in * k_h * k_w * c_out * h_out * w_out
                        total_macs += op_macs

                    elif op_name == 'Gemm':
                        # MACs formula for Gemm (MatMul): N * K * M for matrices (N,K)x(K,M)
                        # In ONNX, input is (N, K) and weights are (M, K), so it's transposed.
                        input_shape = args['input_type_shape'][0]['float']
                        n_gemm, k_gemm = input_shape
                        weight_shape = args['input_type_shape'][1]['float']
                        m_gemm = weight_shape[0] # M is the first dimension of weights (output features)

                        op_macs = n_gemm * k_gemm * m_gemm
                        total_macs += op_macs
                except (IndexError, KeyError, TypeError):
                    # Skip MAC calculation for this node if shapes are not as expected.
                    # A warning could be added here if detailed logging per-node is needed.
                    # warnings.warn(f"Could not calculate MACs for a '{op_name}' node. Required shape info missing in profile.")
                    continue
    except FileNotFoundError:
        warnings.warn(f"Profile file {prof_file} not found. MACs will be reported as 0.")
        return 0
    except json.JSONDecodeError:
        warnings.warn(f"Error decoding JSON from profile file {prof_file}. MACs will be reported as 0.")
        return 0
    except Exception as e:
        warnings.warn(f"An unexpected error occurred while processing profile file {prof_file}: {e}. MACs will be reported as 0.")
        return 0
                
    return int(total_macs)


def analyze_model(model_path: Path) -> ModelProfile:
    """
    Analyzes an ONNX model file and extracts its characteristics.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        A ModelProfile object containing the extracted features.

    Raises:
        FileNotFoundError: If the model file is not found.
        ModelAnalysisError: For other errors during model analysis.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    prof_file_to_remove = None # Tracks the profile file for cleanup

    try:
        model_proto = onnx.load(str(model_path))
        op_types = [node.op_type for node in model_proto.graph.node]
        op_type_counts = dict(Counter(op_types))
        total_ops = len(op_types)

        inputs_info = [{"name": i.name, "shape": [dim.dim_value if dim.dim_value > 0 else -1 for dim in i.type.tensor_type.shape.dim], "dtype": onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type).lower()} for i in model_proto.graph.input]
        outputs_info = [{"name": o.name, "shape": [dim.dim_value if dim.dim_value > 0 else -1 for dim in o.type.tensor_type.shape.dim], "dtype": onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type).lower()} for o in model_proto.graph.output]

        total_macs = 0
        try:
            opts = ort.SessionOptions()
            opts.enable_profiling = True
            # Setting a profile_file_prefix can offer more control over the naming,
            # but end_profiling() already returns the full path of the generated file.
            # opts.profile_file_prefix = "oak_profile" 
            
            sess = ort.InferenceSession(str(model_path), opts, providers=['CPUExecutionProvider'])
            
            input_feeds: Dict[str, Any] = {}
            # Renamed loop variable to i_info to avoid conflict with 'i' from inputs_info list comprehension
            for i_info in sess.get_inputs(): 
                shape = [1 if dim is None or not isinstance(dim, int) or dim <= 0 else dim for dim in i_info.shape]
                # Added 'bool' to dtype_map
                dtype_map = {'float32': np.float32, 'float64': np.float64, 'int32': np.int32, 'int64': np.int64, 'bool': np.bool_} 
                
                # Improved parsing of tensor type string for robustness
                onnx_type_str = i_info.type 
                if onnx_type_str and onnx_type_str.startswith("tensor(") and onnx_type_str.endswith(")"):
                    dtype_str = onnx_type_str[len("tensor("):-1]
                else:
                    dtype_str = 'float32' # Fallback data type
                
                np_dtype = dtype_map.get(dtype_str, np.float32)
                input_feeds[i_info.name] = np.zeros(shape, dtype=np_dtype)

            sess.run(None, input_feeds)
            prof_file = sess.end_profiling() # prof_file is the path to the generated profile file
            prof_file_to_remove = prof_file # Mark for removal

            if prof_file and Path(prof_file).exists():
                total_macs = _calculate_macs_from_profile(prof_file)
            else:
                # This else block might not be reached if end_profiling() fails and raises an exception,
                # or if it returns None/empty string.
                warnings.warn(
                    "ONNX Runtime Profiler did not generate a profile file or the file path was invalid. "
                    "MACs will be reported as 0."
                )
                total_macs = 0 # Ensure total_macs is 0 if the profile file isn't generated

        except Exception as profile_error:
            warnings.warn(
                f"Failed to run ONNX Runtime Profiler: {profile_error}. MACs will be reported as 0."
            )
            total_macs = 0 # Ensure total_macs is 0 in case of a profiling error

        profile_data = ModelProfile(
            model_sha256=_calculate_sha256(model_path),
            file_size_kb=model_path.stat().st_size / 1024,
            total_macs=total_macs,
            total_ops=total_ops,
            op_type_counts=op_type_counts,
            graph_inputs=inputs_info,
            graph_outputs=outputs_info,
        )
        return profile_data

    except Exception as e:
        # Catch-all for other analysis errors (e.g., ONNX model loading issues)
        raise ModelAnalysisError(f"Failed to analyze model {model_path}: {e}") from e
    finally: # Finally block to ensure cleanup of the temporary profile file
        if prof_file_to_remove and Path(prof_file_to_remove).exists():
            try:
                os.remove(prof_file_to_remove)
                # For debugging: print(f"DEBUG: Temporary profile file {prof_file_to_remove} removed.")
            except OSError as e:
                warnings.warn(
                    f"Could not remove temporary profile file: {prof_file_to_remove}. Error: {e}"
                )