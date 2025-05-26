"""
Heuristic-based decision engine for OAK Phase 1.
"""

from oak.analysis.model_profile import ModelProfile
from oak.knowledge_base.hardware_profile import HardwareProfile
from .advisor_profile import AdvisorReport, OptimizationRecommendation

class HeuristicAdvisor:
    """
    Generates optimization recommendations based on a set of heuristic rules.
    """
    def advise(
        self,
        model_profile: ModelProfile,
        hw_profile: HardwareProfile,
        user_priority: str = "latency" # User priorities: 'latency', 'energy', 'size'
    ) -> AdvisorReport:
        """
        Analyzes the model and hardware profiles to suggest optimization strategies.

        Args:
            model_profile: Profile of the machine learning model.
            hw_profile: Profile of the target hardware.
            user_priority: The user's main optimization priority 
                           ('latency', 'energy', or 'size').

        Returns:
            An AdvisorReport containing a list of optimization recommendations.
        """
        recommendations = []

        # Rule 1: Baseline (No optimization)
        baseline_rom_kb = model_profile.file_size_kb
        baseline_ram_kb = baseline_rom_kb * 2
        is_feasible_baseline = baseline_ram_kb < hw_profile.ram_total_kb

        recommendations.append(OptimizationRecommendation(
            strategy_name="Baseline (FP32)",
            estimated_rom_kb=baseline_rom_kb,
            estimated_ram_kb=baseline_ram_kb,
            priority_score=0.5 if is_feasible_baseline else 0.1,
            summary=(
                f"Run the model as is (float32). "
                f"{'Feasible' if is_feasible_baseline else 'Unfeasible'}, as the estimated RAM usage "
                f"({baseline_ram_kb:.0f} KB) {'fits' if is_feasible_baseline else 'does not fit'} "
                f"within the available RAM ({hw_profile.ram_total_kb} KB)."
            )
        ))

        # Rule 2: INT8 Quantization
        # Applicable if the hardware supports TFLite Micro or ONNX Runtime
        if "tflite_micro" in hw_profile.supported_frameworks or "onnx_runtime" in hw_profile.supported_frameworks:
            int8_rom_kb = model_profile.file_size_kb / 4
            int8_ram_kb = int8_rom_kb * 2.5 
            is_feasible_int8 = int8_ram_kb < hw_profile.ram_total_kb
            
            summary_int8 = (
                "Quantize the model to INT8. Reduces size and can speed up inference on compatible hardware."
            )
            summary_int8 += (
                f" {'Feasible' if is_feasible_int8 else 'Unfeasible'}, as the estimated RAM usage "
                f"({int8_ram_kb:.0f} KB) {'fits' if is_feasible_int8 else 'does not fit'} "
                f"within the available RAM ({hw_profile.ram_total_kb} KB)."
            )

            score_int8 = 0.8 if is_feasible_int8 else 0.2
            if "vector_instructions" in hw_profile.accelerators and is_feasible_int8:
                score_int8 = min(score_int8 + 0.15, 1.0) 
                summary_int8 += (
                    " The hardware has vector instructions, which should significantly "
                    "accelerate INT8 performance."
                )
            # Bonus if a more capable framework is supported (e.g., ONNX Runtime might have better INT8 support than generic TFLM)
            elif "onnx_runtime" in hw_profile.supported_frameworks and is_feasible_int8:
                 score_int8 = min(score_int8 + 0.1, 1.0) # Small bonus for ONNX RT

            recommendations.append(OptimizationRecommendation(
                strategy_name="TFLite/ONNX-RT Full INT8 Quantization", # More generic name
                estimated_rom_kb=int8_rom_kb,
                estimated_ram_kb=int8_ram_kb,
                priority_score=round(score_int8,2),
                summary=summary_int8
            ))

        # >>> START OF NEW RULE: FP16 Quantization <<<
        if "tflite_micro" in hw_profile.supported_frameworks or "onnx_runtime" in hw_profile.supported_frameworks:
            fp16_rom_kb = model_profile.file_size_kb / 2
            fp16_ram_kb = fp16_rom_kb * 1.8 # Initial assumption, can be adjusted
            is_feasible_fp16 = fp16_ram_kb < hw_profile.ram_total_kb
            
            summary_fp16 = (
                "Quantize the model to FP16 (half-precision). "
                "Reduces model size (~50% vs FP32) with potentially less accuracy loss than INT8."
            )
            summary_fp16 += (
                f" {'Feasible' if is_feasible_fp16 else 'Unfeasible'}, as the estimated RAM usage "
                f"({fp16_ram_kb:.0f} KB) {'fits' if is_feasible_fp16 else 'does not fit'} "
                f"within the available RAM ({hw_profile.ram_total_kb} KB)."
            )

            score_fp16 = 0.65 if is_feasible_fp16 else 0.15 # Base score for FP16
            
            # Example bonus if hardware has explicit support (requires adding to HardwareProfile)
            # if "fp16_support" in hw_profile.accelerators and is_feasible_fp16:
            #     score_fp16 = min(score_fp16 + 0.1, 1.0)
            #     summary_fp16 += " The hardware may have native support for FP16 arithmetic."
            if "gpu_maxwell_128_cuda" in hw_profile.accelerators and is_feasible_fp16: # Jetson Nano has GPU
                score_fp16 = min(score_fp16 + 0.15, 1.0) # Bonus for GPU
                summary_fp16 += " The present GPU should offer good performance with FP16."

            recommendations.append(OptimizationRecommendation(
                strategy_name="FP16 Quantization", # Generic name
                estimated_rom_kb=fp16_rom_kb,
                estimated_ram_kb=fp16_ram_kb,
                priority_score=round(score_fp16, 2),
                summary=summary_fp16
            ))
        # >>> END OF NEW RULE: FP16 Quantization <<<

        # USER PRIORITY ADJUSTMENT LOGIC (EXISTING)
        # Only adjust if there's at least one feasible recommendation
        if any("Feasible" in rec.summary for rec in recommendations):
            for rec in recommendations:
                # Do not adjust priority for unfeasible strategies
                if "Feasible" not in rec.summary:
                    continue

                adjustment_factor = 0.0 

                if user_priority == "latency":
                    if "INT8" in rec.strategy_name:
                        adjustment_factor = 0.1
                    elif "FP16" in rec.strategy_name: # FP16 can be faster than FP32
                        adjustment_factor = 0.05
                    elif rec.strategy_name == "Baseline (FP32)":
                        # Penalize FP32 if a viable INT8 option exists
                        if any("INT8" in r.strategy_name and "Feasible" in r.summary for r in recommendations):
                             adjustment_factor = -0.05
                
                elif user_priority == "energy":
                    # Assuming INT8 is better for energy (rough estimate for v0.1)
                    if "INT8" in rec.strategy_name:
                        adjustment_factor = 0.15
                    elif "FP16" in rec.strategy_name: # FP16 generally better for energy than FP32
                        adjustment_factor = 0.05
                    elif rec.strategy_name == "Baseline (FP32)":
                         # Penalize FP32 if a viable INT8 option exists
                         if any("INT8" in r.strategy_name and "Feasible" in r.summary for r in recommendations):
                            adjustment_factor = -0.05
                
                elif user_priority == "size":
                    if "INT8" in rec.strategy_name: 
                        adjustment_factor = 0.15
                    elif "FP16" in rec.strategy_name: # FP16 is half the size of FP32
                        adjustment_factor = 0.10 # Good bonus for size
                    elif rec.strategy_name == "Baseline (FP32)":
                        # Penalize FP32 if a viable INT8 or FP16 option exists
                        if any(("INT8" in r.strategy_name or "FP16" in r.strategy_name) and "Feasible" in r.summary for r in recommendations):
                            adjustment_factor = -0.1 
                
                # Apply adjustment, ensuring score stays within 0.0 and 1.0
                rec.priority_score = min(max(rec.priority_score + adjustment_factor, 0.0), 1.0)
                rec.priority_score = round(rec.priority_score, 2)
        
        # Sort recommendations by priority score in descending order
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return AdvisorReport(
            model_sha256=model_profile.model_sha256,
            target_hardware=hw_profile.identifier,
            recommendations=recommendations
        )