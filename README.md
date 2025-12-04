# GEMPA
this pipeline is designed for running with STEAD dataset


```
python main.py train --model TCNSegmentation --x_train ..\dataset\x_train.npy --x_test ..\dataset\x_test.npy --y_train ..\dataset\y_train.npy --y_test ..\dataset\y_test.npy --weight "..\gempa\checkpoints\best-experiments-tcn-seg-step-3_checkpoint_val_loss=-0.1010.pt"
```