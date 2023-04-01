# FedPR: Learning Federated Visual Prompt in Null Space for MRI Reconstruction

This repo is the official implementation of "FedPR". It is based on [Visual Prompt Tuning](https://github.com/KMnP/vpt)

## Abatract

Federated Magnetic Resonance Imaging (MRI) reconstruction enables multiple hospitals to collaborate distributedly without aggregating local data, thereby protecting patient privacy. However, the data heterogeneity caused by different MRI protocols, insufficient local training data, and limited communication bandwidth inevitably impair global model convergence and updating. In this paper, we propose a new algorithm, FedPR, to learn federated visual prompts in the null space of global prompt for MRI reconstruction. FedPR is a new federated paradigm that adopts a powerful pre-trained model while only learning and communicating the prompts with few learnable parameters, thereby significantly reducing communication costs and achieving competitive performance on limited local data. Moreover, to deal with catastrophic forgetting caused by data heterogeneity, FedPR also updates efficient federated visual prompts that project the local prompts into an approximate null space of the global prompt, thereby suppressing the interference of gradients on the server performance. Extensive experiments on federated MRI show that FedPR significantly outperforms state-of-the-art FL algorithms with < 6% of communication costs when given the limited amount of local data.

![fig1](figs/fig1.png#pic_center)

![fig2](figs/fig2.png#pic_center)

## Clone this repository:
```
git clone https://github.com/chunmeifeng/FedPR.git
cd FedPR
```

## Dependencies
```
* Python 3.8
* Torch  1.12.1
* numpy  1.20.3
* h5py   3.1.0
* mmcv   1.7.1
* scipy  1.9.3
* timm   0.6.12
```


## Data Prepare
[fastMRI Dataset](https://fastmri.org/) 

[FeTS 2022 Challenge](https://www.synapse.org/#!Synapse:syn28546456/wiki/) 
 
[IXI Dataset](https://brain-development.org/)

We have transformed all MR volume pixels slice-by-slice into .mat format. To train FedPR on your own dataset please create a directory in your pytorch data root with the following structure,
The data organization structure is roughly as follows, with minor differences between different datasets. 

```
dataset_name
|── imgs
    ├── train
    |   |── Sequence001-T1-Slice0001.mat
    |   |── Sequence001-T2-Slice0001.mat
    |   |── ...
    |   |── Sequence00n-T1-Slice000k.mat
    |   └── Sequence00n-T2-Slice000k.mat
    └── validation
    |    |── valSequence001-T1-Slice0001.mat
    |    |── valSequence001-T2-Slice0001.mat
    |    |── ...
    |    |── valSequence00n-T1-Slice000k.mat
    |    └── valSequence00n-T2-Slice000k.mat
    ├── dataset-name_train.txt
    └── dataset-name_val.txt

```



## Inference

```
python test.py
```

## Training

To train a federated MRI reconstruction model with pre-trained model trained on FastMRI, run:

```
python train.py
```

The specific configuration is in file 'config/different_dataset.py'.



## Citation

```  
  @misc{feng2023learning,
  title={Learning Federated Visual Prompt in Null Space for MRI Reconstruction},       
  author={Feng, Chun-Mei and Li, Bangjun and Xu, Xinxing and Liu, Yong and Fu, Huazhu and Zuo, Wangmeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```


