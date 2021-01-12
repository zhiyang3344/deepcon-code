# deepcon-code

The beta implement of paper "DeepCon: Contribution Coverage Testing for Deep Learning Systems"

1. conda install tensorflow-gpu=1.14.0 keras=2.2.4

2. python nn_contribution_util.py --model_name lenet5 --start_index 0 --end_index 10000 --batch_size 100

3. python dfg_util.py --model_name lenet5 --start_index 0 --end_index 10000 --batch_size 100 --threshold 0
