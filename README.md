# GEMPA

GEMPA is an early-stage Earthquake Early Warning (EEW) system designed to detect seismic activity using real-time sensor data and provide rapid alerts before strong shaking arrives. This project aims to explore low-cost, scalable detection pipelines suitable for research, prototyping, and educational use.

this pipeline is designed for running with STEAD dataset

## GPU Support

This pipeline supports:
- **CUDA** - for NVIDIA GPU acceleration
- **Multiple CUDA** (coming soon) - multi-GPU support for distributed training
- **DirectML** - for GPU acceleration on Windows (non cuda GPU)


```sh
python main.py train --model TCNSegmentation --csv {csv_path} --hdf5 {hdf5 path} --out ..\checkpoints --logs ..\logs --weight {last weight (optional)}
```