# OAK: Optimization Advisor Kit 🌳

OAK (Optimization Advisor Kit) is a command-line interface (CLI) tool designed to help developers optimize Machine Learning models for efficient deployment on edge hardware (Edge AI / TinyML).

**Project Status:** Version 0.1.0 (Alpha) - Heuristic advisory with multi-hardware support and optimization strategies.

## Vision

To become the standard CLI tool for developers seeking to deploy Machine Learning models on edge hardware efficiently, sustainably, and energy-consciously. OAK aims to democratize Green AI at the edge by removing the complexity from the TinyML/Edge AI process.

## Features (v0.1.0)

- Analyzes Machine Learning models in ONNX format
- Extracts key model metrics including size, operation count, and MACs (Multiply-Accumulate operations)
- Provides optimization recommendations based on heuristic rules for specific target hardware
  - Included strategies: Baseline (FP32), Full INT8 Quantization, FP16 Quantization
- Considers available RAM and ROM memory on target hardware
- Allows users to specify optimization priority (latency, energy, size)
- Initial support for a variety of hardware platforms (see "Hardware Support" section below for a complete list).


**Note:** These hardware profiles are production-ready configurations based on official specifications and real-world deployment scenarios. They are not intended for testing purposes but for actual production deployment planning and optimization decisions.

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/zeluslab/oak.git
   cd OAK
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv  # Or python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   # .venv\Scripts\activate   # On Windows PowerShell
   # call .venv\Scripts\activate.bat # On Windows CMD
   ```

3. **Install dependencies from requirements.txt:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To get advice on an ONNX model for specific hardware:

```bash
python -m src.oak.cli.main advise PATH_TO_YOUR_MODEL.onnx --hardware HARDWARE_ID --user-priority PRIORITY
```

**Example:**
```bash
python -m src.oak.cli.main advise models/mobilenetv2-7.onnx --hardware jetson-nano-b01 --user-priority size
```

### `advise` Command Options:

- `MODEL_PATH`: Path to the ONNX model file (required)
- `--hardware TEXT`: Target hardware identifier (e.g., 'esp32-s3', 'raspberrypi-pico'). Default: esp32-s3
- `--user-priority TEXT`: User priority for optimization: 'latency', 'energy', or 'size'. Default: latency
- `--help`: Show help message and exit

### Example Output (for Jetson Nano with 'size' priority)

```
Loading Knowledge Base from '.../OAK/data'...
Target hardware: NVIDIA jetson-nano-b01
Analyzing model 'mobilenetv2-7.onnx'...
Model SHA256: 0e7c0aa4bc74...
Total operations: 105, Total MACs: 5562.39M
Generating recommendations...

OAK Optimization Report
Model: mobilenetv2-7.onnx | Hardware: jetson-nano-b01 | Priority: size

                                     Strategy Recommendations                                     
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Priority   ┃ Strategy                              ┃ ROM (KB) ┃ RAM (KB) ┃ Summary                                    ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       1.00 │ TFLite/ONNX-RT Full INT8 Quantization │   3409.0 │   8522.4 │ Quantize model to INT8. Reduces size and   │
│            │                                       │          │          │ can accelerate inference on compatible     │
│            │                                       │          │          │ hardware. Viable as estimated RAM usage    │
│            │                                       │          │          │ (8522 KB) fits in available RAM (4194304   │
│            │                                       │          │          │ KB).                                       │
│       0.90 │ FP16 Quantization                     │   6817.9 │  12272.3 │ Quantize model to FP16 (half-precision).   │
│            │                                       │          │          │ Reduces model size (~50% vs FP32) with     │
│            │                                       │          │          │ potentially lower accuracy loss than INT8.  │
│            │                                       │          │          │ Viable as estimated RAM usage (12272 KB)   │
│            │                                       │          │          │ fits in available RAM (4194304 KB). The    │
│            │                                       │          │          │ present GPU should offer good FP16          │
│            │                                       │          │          │ performance.                               │
│       0.40 │ Baseline (FP32)                       │  13635.9 │  27271.7 │ Run model as-is (float32). Viable as       │
│            │                                       │          │          │ estimated RAM usage (27272 KB) fits in     │
│            │                                       │          │          │ available RAM (4194304 KB).                │
└────────────┴───────────────────────────────────────┴──────────┴──────────┴────────────────────────────────────────────┘
```

## Hardware Support

OAK currently supports a diverse range of edge hardware platforms. Each profile contains key specifications like RAM, CPU architecture, and known accelerators, enabling tailored optimization advice. The following 20 platforms are currently profiled:

### Microcontrollers (MCUs)

* **Espressif ESP32-S3 (`esp32s3`)**: Dual-core Xtensa LX7 MCU with AI vector instructions and FPU.
* **Seeed Studio XIAO ESP32S3 Sense (`seeed-xiao-esp32s3-sense`)**: Compact ESP32-S3 board with integrated sensors.
* **Arduino Nano ESP32 (`arduino-nano-esp32`)**: Arduino Nano form-factor board featuring an ESP32-S3.
* **Espressif ESP32-WROOM-32 (`esp32-wroom-32`)**: Popular dual-core Xtensa LX6 MCU module with Wi-Fi/Bluetooth.
* **Raspberry Pi Pico (RP2040) (`raspberrypi-pico`)**: Low-cost, dual-core ARM Cortex-M0+ microcontroller.
* **Arduino Nano 33 BLE Sense (nRF52840) (`arduino-nano-33-ble`)**: ARM Cortex-M4F MCU with Bluetooth LE and onboard sensors.
* **STMicroelectronics STM32F407VG (`stm32f407vg`)**: High-performance ARM Cortex-M4F MCU with FPU and DSP instructions.
* **STMicroelectronics STM32H743ZI (`stm32h743zi`)**: High-performance ARM Cortex-M7 MCU with FPU_DP and DSP instructions.
* **Arduino Portenta H7 (STM32H747XI) (`arduino-portenta-h7`)**: Dual-core (ARM Cortex-M7 + M4F) high-performance MCU board.
* **NXP i.MX RT1060 (e.g., Teensy 4.0/4.1) (`nxp-imxrt1060`)**: High-speed ARM Cortex-M7 crossover MCU.
* **NXP i.MX RT1170 EVK (`nxp-imxrt1170-evk`)**: Dual-core (ARM Cortex-M7 + M4F) crossover MCU with VGLite NPU option.
* **Sony Spresense Main Board (CXD5602GG) (`sony-spresense`)**: Hexa-core ARM Cortex-M4F MCU with GPS and audio capabilities.
* **Kendryte K210 (`kendryte-k210`)**: Dual-core RISC-V 64-bit MCU with KPU AI accelerator.
* **Sipeed Maix Amigo (`sipeed-maix-amigo`)**: Portable Kendryte K210-based AI development device with screen and camera.
* **Renesas RA6M4 (`renesas-ra6m4`)**: ARM Cortex-M33 MCU with TrustZone and crypto acceleration.

### Edge Computing Platforms & Single Board Computers (SBCs)

* **NVIDIA Jetson Nano B01 (`jetson-nano-b01`)**: Quad-core ARM Cortex-A57 AI computing platform with 128-core Maxwell GPU.
* **Raspberry Pi Zero 2 W (`raspberrypi-zero-2w`)**: Compact quad-core ARM Cortex-A53 SBC.
* **Raspberry Pi 3 Model B+ (`raspberrypi-3bplus`)**: Quad-core ARM Cortex-A53 SBC.
* **Raspberry Pi 4 Model B (2GB) (`raspberrypi-4b`)**: Quad-core ARM Cortex-A72 SBC.
* **Coral Dev Board Micro (`coral-dev-board-micro`)**: Dual-core ARM Cortex-M7+M4F MCU board with Google Edge TPU AI accelerator.

Additional hardware profiles can be easily added by creating new JSON files in the `data/hardware/` directory.

## How to Contribute (Future)

Instructions on how to contribute, add new hardware profiles, submit benchmarks (Phase 2 of roadmap), and run tests will be added soon. For now, feel free to open Issues on GitHub to report bugs or suggest improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Roadmap

### Current Version (0.1.0 - Alpha)
- ✅ Basic ONNX model analysis
- ✅ Heuristic recommendation system  
- ✅ Multi-hardware support via JSON profiles
- ✅ Functional CLI with user-friendly interface

### Upcoming Versions
- **v0.2.0**: Improved estimation accuracy and support for more model formats
- **v0.3.0**: Integration with real benchmarking tools
- **v1.0.0**: Recommendation system based on empirical data and ML

---

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Project Structure

```
OAK/
├── data/
├── models/
├── src/
│   └── oak/
│       ├── advisor/
│       ├── analysis/
│       ├── cli/
│       └── knowledge_base/
├── LICENSE
├── pyproject.toml
├── poetry.lock
├── requirements.txt
└── README.md
```