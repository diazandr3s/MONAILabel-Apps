#  Brain parcelation using highresnet 3D - WORK IN PROGRESS

- How to enumerate the labels. There are around 141 different label but numbersa are from 0 to 207.
- Make sure the hyperparamter combination doesn't occupy all the GPU memory

### Model Overview

MONAI Label App using highresnet 3D. 

    HighRes3DNet by Li et al. 2017 for T1-MRI brain parcellation

### Data

The dataset includes 35 brain MRI scans obtained from the OASIS project. The manual brain segmentations of these images were produced by Neuromorphometrics, Inc. (http://Neuromorphometrics.com/) using the brainCOLOR labeling protocol. The data were applied in the 2012 MICCAI Multi-Atlas Labeling Challenge and can be downloaded at (https://masi.vuse.vanderbilt.edu/workshop2012/index.php/Main_Page). In the challenge, 15 subjects were used as atlases and the remaining 20 images were used for testing.


Paper: Multi-atlas segmentation with joint label fusion and corrective learning—an open source implementation


https://www.frontiersin.org/articles/10.3389/fninf.2013.00027/full

- Target: Brain parcelation
- Task: Segmentation 
- Modality: MR - T1W

### Input

- 1 channel MR

### Output

- 208 channels representing brain parcelation
