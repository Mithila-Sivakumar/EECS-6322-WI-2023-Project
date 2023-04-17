# EECS-6322-Project
EECS 6322 WI 2023 Reproducibility challenge
This project is inspired1 by the Machine Learning Reproducibility Challenge
(https://paperswithcode.com/rc2020/task). The aim of this project is to
replicate the central claim in a selected deep learning focused paper. This
will include re-implementing the core methods and experiments in the paper

Project Report - EECS_6322_Project_Report.pdf
Presentation - EECS_6322_Presentation

## Abstract
In this project, we present a reproduction study of the paper "TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification". We implemented the TransMatcher using PyTorch (on top of QAConv-GS as mentioned in the paper). We implemented the TransMatcher model and reproduced the scenario where the model is trained on Market-1501 dataset and tested on CUHK03-np dataset. We evaluated the model's performance using the mean average precision (mAP), and Rank-1 metric and compared our results to those reported in the original paper

## Implementation Details

We have tried to reproduce the implemention the TransMatcher decoder on top of the official PyTorch project of QAConv-GS. We have used Resnet50-idn-b as the backbone network. We loaded their model from their official github repository. This backbone network is pre-trained on ImageNet with states of Batch Normalization layers being fixed. We have used layer3 feature map and also appended a 3x3 neck convolution layer to produce the final feature map as explained in the paper. The input image is resized to 384 × 128. The batch size is set to 64, with K=4 for the GS sampler. The network is trained with the SGD optimizer, with a learning rate of 0.0005 for the backbone network, and 0.005 for newly added layers. They are decayed by 0.1 after 10 epochs, and 15 epochs are trained in total. All these settings are exactly the same as expressed in the paper. We resued the dataloaders from the authors code and part of training and test loop and modified it according to our experiments.

Note - The datasets are not uploaded here in github, instead a link to them is provided below
 1. CUHK03-NP is [here](https://github.com/zhunzhong07/person-re-ranking/blob/master/CUHK03-NP/README.md). We need only the detected subset and not the labelled subset as mentioned in the paper
 2. Market-1501 is [here](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

To train the model using these datasets, create a folder named 'data' inside QAConv folder and upload the datasets there (folder name 'market' for market-1501 datset and folder name 'cuhk03_np_detected' for CUHK03 dataset)

The checkpoint file to do evaluation is [here](https://yuoffice-my.sharepoint.com/:f:/g/personal/msivakum_yorku_ca/Eu00buljiBVGgimzZwCkbnUBtb79SSrsb8gk_YcagmnygQ?e=NW9GZf). Access is within york university

## Results

While we were able to successfully train the model on Market-1501 dataset. Our test results on the CUHK03-np-detected, however, were not the same as reported in the paper. Our rank-1 and mAP were 16.5% and 17% respectively, which is 6% less than what is reported in the paper. In summary, while were sucessfully able to create and train the TransMatcher model, we were not able to successfully reproduce the exact results.

<img width="358" alt="Screen Shot 2023-04-16 at 7 15 24 PM" src="https://user-images.githubusercontent.com/127549357/232348788-0e1871d2-6117-4a48-8104-b77d0d151a44.png">


The results of training and testing are avilable in "results.json" and "eval.json" respectively.

