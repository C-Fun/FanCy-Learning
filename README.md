# FanCy-Learning

Fang Chen's Deep Learning Toolbox

## Image Classification

Support:

- Backbones
  - [MobileNetV2](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) -- [code link](https://github.com/tonylins/pytorch-mobilenet-v2)
  - [MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) (small, large) -- [code link](https://github.com/d-li14/mobilenetv3.pytorch)
  - [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) & [ResNeXt](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html) (with depth: 18, 34, 50, 101, 152) -- [code link](https://github.com/miraclewkf/ResNeXt-PyTorch)
  - [Vision Transformer](https://arxiv.org/abs/2010.11929) -- [code link](https://github.com/lucidrains/vit-pytorch)
  - SwinMLP & [SwinTransformerV2](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html) -- [code link](https://github.com/microsoft/Swin-Transformer)
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

