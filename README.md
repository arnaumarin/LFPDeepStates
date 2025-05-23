[![DOI](https://zenodo.org/badge/694391403.svg)](https://doi.org/10.5281/zenodo.15066751)

# Neural models for anesthesia stage transition detection and classification

This repository contains code for the detection and classification of anesthesia-induced brain state transitions, including wakefulness, slow oscillations, and microarousals. We leverage a dual-model Convolutional Neural Network (CNN) and a self-supervised autoencoder-based multimodal clustering algorithm to achieve accurate brain state classification and transition detection based on in vivo LFP recordings from rats.

## Overview
The pipeline processes the data through a series of steps, including preprocessing, state classification, and transition detection, using a combination of supervised and self-supervised learning techniques. It achieves accuracy rates of up to 96% for specific states and averages over 85% across all states, with 74% accuracy for detecting transitions. The methodology employs a leave-one-out strategy for model training, ensuring broad applicability across subjects.

![Pipeline Overview](images/pipeline-1.png)

The demonstration dataset is publicly available at https://doi.org/10.5281/zenodo.14990181 Other data are available from the corresponding author on reasonable request.

## Usage
For classification, check the example notebook located at `example_notebook/general_notebook.ipynb`. For transition detection, refer to the notebook at `example_notebook/transition_notebook.ipynb`. For further understanding and visualization of the process during the Autoencoders phase, please check and follow the comments in the example notebooks located at `example_notebook/clusters_psd_notebook.ipynb` and `example_notebook/autoencoders_synthetic.ipynb`.

Make sure to follow the instructions in the notebooks to properly preprocess your data, train the models, and perform the classification and transition detection tasks.

## Citation

If you find this repository useful for your research, please consider citing the following works:

**Arnau Marin-Llobet, Arnau Manasanch, Leonardo Dalla Porta, Melody Torao-Angosto, and Maria V. Sanchez-Vives**  
*Neural models for detection and classification of brain states and transitions*  
**Communications Biology**, 8, 599 (2025).  
[https://doi.org/10.1038/s42003-025-07991-3](https://doi.org/10.1038/s42003-025-07991-3)

**Arnau Marin-Llobet, Arnau Manasanch, Leonardo Dalla Porta, and Maria V. Sanchez-Vives**  
*Deep neural networks for detection and classification of brain states*  
**Journal of Sleep Research**, Vol. 33, Wiley (2024)


## Issues
For any questions or issues, feel free to raise an issue on this GitHub repository, and we will do our best! 
