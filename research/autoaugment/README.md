<font size=4><b>Train Wide-ResNet, Shake-Shake and ShakeDrop models on CIFAR-10
and CIFAR-100 dataset with AutoAugment.</b></font>

The CIFAR-10/CIFAR-100 data can be downloaded from:
https://www.cs.toronto.edu/~kriz/cifar.html.

The code replicates the results from Tables 1 and 2 on CIFAR-10/100 with the
following models: Wide-ResNet-28-10, Shake-Shake (26 2x32d), Shake-Shake (26
2x96d) and PyramidNet+ShakeDrop.

<b>Related papers:</b>

AutoAugment: Learning Augmentation Policies from Data

https://arxiv.org/abs/1805.09501



<b>Prerequisite:</b>

1.  Install TensorFlow. Be sure to run the code using python2 and not python3.

2.  Download CIFAR-10/CIFAR-100 dataset.

```shell
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

<b>How to run:</b>

```shell
# cd to the your workspace.
# Specify the directory where dataset is located using the data_path flag.
# Note: User can split samples from training set into the eval set by changing train_size and validation_size.

# For example, to train the Wide-ResNet-28-10 model on a GPU.
python train_cifar.py --model_name=wrn \
                      --checkpoint_dir=/tmp/training \
                      --data_path=/tmp/data \
                      --dataset='cifar10' \
                      --use_cpu=0
```

## Contact for Issues



## 文件标注
整个auto的使用是针对预处理部分，所以对于网络的结构并没有改变，只有简单的调用
主要的文件在于政策的选择和在第一个augmentation_transforms.py文件内置的apply方法的调用
policies.py 包含了适合cifar10,cifar100数据集的变化方法，如若想用其他数据集，则可在论文的附表中查看并自己写列表。
augmentation_transforms.py中包含了所有的变化方法，以及对图片的使用函数，使用函数的调用是在data_utils.py文件中
custom_ops.py中定义了网络的常用函数，包括卷积层，BN层
data_utils.py构造数据集文件，
helper_utils.py辅助文件，和auto无关
