# Neural Models for Anesthesia Stage Transition Detection and Classification

This repository contains code for the detection and classification of anesthesia-induced brain state transitions, including wakefulness, slow oscillations, and microarousals. We leverage a dual-model Convolutional Neural Network (CNN) and a self-supervised autoencoder-based multimodal clustering algorithm to achieve accurate brain state classification and transition detection based on in vivo LFP recordings from rats.

## Overview
The pipeline processes the data through a series of steps, including preprocessing, state classification, and transition detection, using a combination of supervised and self-supervised learning techniques. It achieves accuracy rates of up to 96% for specific states and averages over 85% across all states, with 74% accuracy for detecting transitions. The methodology employs a leave-one-out strategy for model training, ensuring broad applicability across subjects.

![Pipeline Overview](images/pipeline-1.png)

## Usage
For classification, check the example notebook located at `example_notebook/general_notebook.ipynb`. For transition detection, refer to the notebook at `example_notebook/transition_notebook.ipynb`.

Make sure to follow the instructions in the notebooks to properly preprocess your data, train the models, and perform the classification and transition detection tasks.

## Citations
If you find this repository useful for your research, please consider citing our work:


## Issues
For any questions or issues, feel free to raise an issue on this GitHub repository, and we will do our best! 
