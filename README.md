# FanCy-Learning

Fang Chen's Deep Learning Toolbox

## Image Classification

Support:

- Backbones
  - MobileNetV2
  - MobileNetV3 (small, large)
  - ResNet & ResNeXt (with depth: 18, 34, 50, 101, 152)
  - SwinMLP
  - SwinTransformerV2
- Datasets
  - Cifar10 / Cifar100
  - STL10
  - ImageNet

Running Command:

```shell
# train
> python ./image_classification/train_val.py --root ./ --seed 0 --devices 0 --dataset STL10 --im_size 128 --batch_size 8 --architecture mobilenetv2 --pretrained '' --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --log
# test only
> python ./image_classification/train_val.py --root ./ --seed 0 --devices 0 --dataset STL10 --im_size 128 --batch_size 8 --architecture mobilenetv2 --pretrained '' --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --log --test_only
```

## Object Detection

coming soon

