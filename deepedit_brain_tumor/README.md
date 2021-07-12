# DeepEdit for Brain Tumor

### Model Overview

Interactive MONAI Label App using DeepEdit to label brain tumour over single modality 3D MRI Images

### Data

The training data is from the Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018 (https://www.med.upenn.edu/cbica/brats2020/data.html). Single label and single modality (T1gd) dataset can be downloaded from this [link](https://emckclac-my.sharepoint.com/:u:/g/personal/k2039747_kcl_ac_uk/ERTrN7bY-tZAsitESjVNIu8BI7LxirdsO0p80_tK1kz2-A?e=GWUQFX)

The script **convert_to_single_label_single_modality.py** could be used to convert images from multiple label into single label and multimodality into single modality (T1gd)

- Target: Tumor
- Task: Segmentation 
- Modality: MRI

### Inputs

- 1 channel MRI (T1gd)
- 3 channels (T1gd + foreground points + background points)

### Output

- 1 channel representing brain tumor


![DeepEdit for Brain Tumor](../docs/images/sample-apps/deepedit_brain_tumor.png)
