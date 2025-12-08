# GEMPA

GEMPA is an early-stage Earthquake Early Warning (EEW) system designed to detect seismic activity using real-time sensor data and provide rapid alerts before strong shaking arrives. This project aims to explore low-cost, scalable detection pipelines suitable for research, prototyping, and educational use.

this pipeline is designed for running with STEAD dataset


```
python main.py train --model TCNSegmentation --x_train ..\dataset\x_train.npy --x_test ..\dataset\x_test.npy --y_train ..\dataset\y_train.npy --y_test ..\dataset\y_test.npy --out ..\checkpoints --count 1000 --test_count 50 --train_pos 900 --test_pos 300
```