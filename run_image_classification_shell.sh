#!/bin/sh

DIR=$0
METHOD=$1 # mobilenetv2

PYTHONPATH="$(dirname $DIR)/":$PYTHONPATH \
python ./image_classification/train_val.py \
          --root ./ --seed 0 --devices 0 \
          --dataset STL10 --im_size 128 --batch_size 8 \
          --architecture $METHOD --pretrained '' \
          --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD \
          --log

# # command
# python ./image_classification/train_val.py --root ./ --seed 0 --devices 0 --dataset STL10 --im_size 128 --batch_size 8 --architecture mobilenetv2 --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --log
