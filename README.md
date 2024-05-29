![GitHub commit activity](https://img.shields.io/github/commit-activity/m/RyanTomich/photonic_compiler)


# Automated Compiler Software for Emerging Photonic Computing Hardware

## Overview
This project aims to make a compiler with optimization and code generation that can take neural network inference requests and divide the load between classical hardware (CPU/ GPU) and novel photonic hardware. We take advantage of the TVM compiler to translate Tensorflow and Pytorch models into their internal Relay IR, at which point this compiler conducts the next layer of scheduling and translation.

<img src="UROP_system_design.drawio.png" alt="drawing" style="width:500px;"/>

*Contributions highlighted in green


## File sections and definitions
- **json_parse:** <span style="color:red">IN PROGRESS</span>.
Everything related to code generation and parsing of TVM Relay IR .json files
    - parser.py: script for instruction generation and file structure
    - simple_LeNet_parsed.txt: generated instructions

- **ONNX-AlexNet:**
 model files/parameters for AlexNet ML model in the ONNX format

- **ONNX-ResNet:**
 model files/parameters for ResNet ML model in the ONNX format

- **Pytorch-LeNet:**
 model files/parameters and code for loading LeNet from pytorch

- **inference_pratice:**
recreating popular modles from scrach using numpy

## File sections and definitions
To recreate the Relay IR, run the followig for each model
- [LeNet_simple.py](Pytorch-LeNet/LeNet_simple.py)

To run optimization analysis and create RISC-Photonic instructions
- [parser.py](json_parser/parser.py)
