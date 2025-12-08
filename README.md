# GEMPA

GEMPA is an early-stage Earthquake Early Warning (EEW) system designed to detect seismic activity using real-time sensor data (received from SeedLink) and provide rapid alerts before strong shaking arrives. This project aims to explore low-cost, scalable detection pipelines suitable for research, prototyping, and educational use.

this pipeline is designed for running with STEAD dataset

## GPU Support

This pipeline supports:
- **CUDA** - for NVIDIA GPU acceleration
- **Multiple CUDA** (coming soon) - multi-GPU support for distributed training
- **DirectML** - for GPU acceleration on Windows (non cuda GPU)


## List Models
- **TCNSegmentation** - TCN segmentation for rapid earthquake detection by segmenting the area between P and S waves

## Commands

Available commands from `main.py`:
- **train** - Train the model with specified dataset and parameters
- **test** - Test the trained model on test dataset
- **ls** - List all available models
- **debug** - Debug configuration file
- **split** - Split dataset using the DataSplitter utility
- **info** - Display model information and architecture summary


```sh
python main.py train --model TCNSegmentation --csv {csv_path} --hdf5 {hdf5 path} --out ..\checkpoints --logs ..\logs --weight {last weight (optional)}
```