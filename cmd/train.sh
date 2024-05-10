#!/bin/bash
python3 main.py --config exps/ease_cifar.json \
                --logger="wandb" \
                --data_dir="/dataset"
