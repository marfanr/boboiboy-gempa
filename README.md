# GEMPA
this pipeline is designed for running with STEAD dataset


```
python main.py train --model TCNSegmentation --x_train ..\dataset\x_train.npy --x_test ..\dataset\x_test.npy --y_train ..\dataset\y_train.npy --y_test ..\dataset\y_test.npy --out ..\checkpoints --count 1000 --test_count 50 --train_pos 900 --test_pos 300
```