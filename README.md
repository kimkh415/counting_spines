# Counting Spines
This was originally conceived as a project for the 10707:Deep Learning course at CMU. It is a deep learning solution to the problem of quantifying neuronal "dendritic spines" from microscopy images. Broadly speaking, this pipeline allows researchers to provide a set of hand-labelled microscopy images to train a lightweight convolutional neural network. This model is then used to create prediction maps of where it predicts dendritic spines are likely to exist. Subsequent clustering algorithms are used to quantify the number of predicted spines as well as their relative size. The full description can be found at: [MANUSCRIPT URL]


# Steps

## Deploy environment

Included in this repository are some system-specific .yml files. These can be used to easily create python environments with all the required packages. First make sure to have a system install of Anaconda3(https://www.anaconda.com/). Then just run:
'''conda env create -f (appropriate path).yml'''
Followed by activating the environment with:
'''activate counting_spines'''
This must be done before each session.

## Create Training Images and Labels

## Modify Configuration File
All modifiable directory paths and parameters for the whole pipeline can be found and modified in the configuration file (src/config.ini). The default values for the model and clustering parameters should be sufficient for most use cases. Some expected exceptions include the "training_epochs" parameter, which should be increased if it seems like the training curves have not reached inflection. If your hardware is insufficient(e.g. not enough RAM), then it may be helpful to reduce the number of filters and/or nodes in the neural network. More details on individual parameters can be found in the config.ini file itself.

## Run Training Pipeline
This is a multistep pipeline with 4 major steps and many intermediate files. Outlined here are the steps, description of purpose, and usage expectations. More detail specific non-config inputs for each script can be found using the "--help" flag. 

### 1_gen_data.py
This step is designed to take in the training images and labels in native format and do the following:

1. Convert data into numpy array format
1. Split training images into patches of size specified in config file
1. Assign binary label to each training patch based on whether or not it contains a dendritic spine
1. Filter out training patches which consist of empty space in order to remove negative training bias from the dataset
1. Store this data in binary format for subsequent steps

This step only needs to be run a single time per dataset.

### 2_cnn.py
This step uses the training patches to train the convolutional neural network model and then stores the trained model for subsequent use. Fortunately the current framework is simple enough such that a GPU is not required for training, though one can be used (SHOULD IMPLEMENT IN CONFIG FILE).

The trained model will be stored in a time-stamped directory along with relevant training data. For maximum accuracy, the training curves should have reached convergance. If this is not the case, then likely more training epochs should be used and the model retrained.

### 3_scanner.py
