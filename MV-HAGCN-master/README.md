# MV-HAGCN
This repository provides a reference implementation of MV-HAGCN as described in the paper:
> MV-HAGCN: Multi-View Hybrid Attention Graph Convolutional Network for predicting miRNA-disease associations
>
> KongLin Xing
> 

Available at 

## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.19.2
* torch==1.9.1+cu111
* scipy==1.5.4
* pandas==0.25.0

## Datasets：data
* disease: the four similarity matrices of disease
* mirna: the six similarity matrices of miRNA
* train: the train and test sets which contains 1:1, 1:5 and 1:10 data.
* Y: the initial feature obtained from RWR

## Usage
Step 1: Data Preprocessing
Before running the main model, you must first preprocess the data to integrate the similarity matrices. Run the data preprocessing script:
python `data_preprocessing.py`
Step 2: Run Link Prediction
After successful data preprocessing, run the main model for miRNA-disease association prediction:
python `Link_Prediction.py`


## About `trained_model.pth`
* It is a trained model, you can do prediction directly by load this file through `torch.load('trained_model.pth')`.
* Then, please add those parameters to the model
*    'feature': the initial feature of nodes, size 878
*    'A': the fused similarity matrix of miRNA
*    'B': the fused similarity matrix of disease
*    'o_ass': miRNA-disease adjacent matrix
*    'layer': convolutional layer, you can set the value with 2
* Then, you can obtain the finial feature matrix of miRNAs and diseases, and you can do prediction tasks.
* The experimental parameter setting about this model: epoch-2000, learning rate-0.0005, embedding size-256, convolution layer-2.



