# EECS-6322-Project
EECS 6322 WI 2023 Reproducibility challenge
This project is inspired1 by the Machine Learning Reproducibility Challenge
(https://paperswithcode.com/rc2020/task). The aim of this project is to
replicate the central claim in a selected deep learning focused paper. This
will include re-implementing the core methods and experiments in the paper

## Abstract
In this project, we present a reproduction study of the paper "TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification". We implemented the TransMatcher using PyTorch (on top of QAConv-GS as mentioned in the paper) and reproduced the experiments on three different person re-identification benchmarks. We evaluated the model's performance using the mean average precision (mAP), and Rank-1 metric and compared our results to those reported in the original paper.

## Implementation Details

We have implemented the TransMatcher decoder on top of the official PyTorch project of QAConv-GS \cite{TransMatcher}. We have used Resnet50-idn-b \cite{pan2018two} as the backbone network. We loaded their model from their official github repository \cite{XingangPan/IBN-Net}. This backbone network is pre-trained on ImageNet with states of Batch Normalization layers being fixed. We have used layer3 feature map and also appended a 3x3 neck convolution layer to produce the final feature map as explained in the paper. The input image is resized to 384 × 128. The batch size is set to 64, with K=4 for the GS sampler. The network is trained with the SGD optimizer, with a learning rate of 0.0005 for the backbone network, and 0.005 for newly added layers. They are decayed by 0.1 after 10 epochs, and 15 epochs are trained in total. All these settings are exactly the same as expressed in the paper.

## Results

## Summary
