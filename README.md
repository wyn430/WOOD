# WOOD
Implementation of our recent paper, [WOOD: Wasserstein-based Out-of-Distribution Detection](http://arxiv.org/abs/2112.06384).

## Abstract
The training and test data for deep-neural-network-based classifiers are usually assumed to be sampled from the same distribution. When part of the test samples are drawn from a distribution that is sufficiently far away from that of the training samples (a.k.a. out-of-distribution (OOD) samples), the trained neural network has a tendency to make high confidence predictions for these OOD samples. Detection of the OOD samples is critical when training a neural network used for image classification, object detection, etc. It can enhance the classifier's robustness to irrelevant inputs, and improve the system resilience and security under different forms of attacks. Detection of OOD samples has three main challenges: (i) the proposed OOD detection method should be compatible with various architectures of classifiers (e.g., DenseNet, ResNet), without significantly increasing the model complexity and requirements on computational resources; (ii) the OOD samples may come from multiple distributions, whose class labels are commonly unavailable; (iii) a score function needs to be defined to effectively separate OOD samples from in-distribution (InD) samples. To overcome these challenges, we propose a Wasserstein-based out-of-distribution detection (WOOD) method. The basic idea is to define a Wasserstein-distance-based score that evaluates the dissimilarity between a test sample and the distribution of InD samples. An optimization problem is then formulated and solved based on the proposed score function. The statistical learning bound of the proposed method is investigated to guarantee that the loss value achieved by the empirical optimizer approximates the global optimum. The comparison study results demonstrate that the proposed WOOD consistently outperforms other existing OOD detection methods.

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{wang2021wood,
      title={WOOD: Wasserstein-based Out-of-Distribution Detection}, 
      author={Yinan Wang and Wenbo Sun and Jionghua "Judy" Jin and Zhenyu "James" Kong and Xiaowei Yue},
      year={2021},
      eprint={2112.06384},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Installation

The code has been tested on following environment

```
Ubuntu 18.04
python 3.6
CUDA 11.0
torch 1.4.0
scikit-learn 0.21.3
geomloss 0.2.3
```

## Dataset
The experiments are conducted on MNIST, FashionMNIST, Cifar10, SVHN, and Tiny-ImageNet-200. The first four datasets can be automatically downloaded via PyTorch, the Tiny-ImageNet-200 needs to be manually downloaded and put the data files in the [folder](https://github.com/wyn430/WOOD/data) 

## Usage

### WOOD

The performance of the proposed WOOD framework is tested using [DenseNet](https://github.com/andreasveit/densenet-pytorch) as the backbone classifier.

```
CUDA_VISIBLE_DEVICES = ID  python main_OOD_binary.py [beta value] [number of epochs] [batch size] [InD batch size] [InD dataset] [OOD dataset] [Image channels]
CUDA_VISIBLE_DEVICES = ID  python main_OOD_dynamic.py [beta value] [number of epochs] [batch size] [InD batch size] [InD dataset] [OOD dataset] [Image channels]

e.g. CUDA_VISIBLE_DEVICES=0 python main_OOD_binary.py 0.1 60 60 50 Cifar10 Imagenet_c 3
     CUDA_VISIBLE_DEVICES=0 python main_OOD_dynamic.py 0.1 60 60 50 Cifar10 Imagenet_c 3
```
Note that the difference between [main_OOD_binary.py](https://github.com/wyn430/WOOD/blob/master/main_OOD_binary.py) and [main_OOD_dynamic.py](https://github.com/wyn430/WOOD/blob/master/main_OOD_dynamic.py) is the distance matrix used in the Wasserstein distance, which is discussed in our paper. The trained model is saved in [directory](https://github.com/wyn430/WOOD/tree/master/runs). The model performance will be routinely tested during training.

### Baseline Methods
The implementation of baseline methods is mainly based on the [repo](https://github.com/megvii-research/FSSD_OoD_Detection).

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/wyn430/WOOD/blob/master/LICENSE) file for details.

## Acknowledgments

The implementation of DenseNet is base on the [repo](https://github.com/andreasveit/densenet-pytorch).

The implementation of Wasserstein distance is mainly base on [geomloss](https://www.kernel-operations.io/geomloss/).


