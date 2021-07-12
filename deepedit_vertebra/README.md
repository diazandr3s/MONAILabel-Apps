# DeepEdit for spine/vertebra segmentation

### Model Overview

Interactive MONAI Label App using DeepEdit to label the spine over CT Images

### Data

The training data is from VerSe2020 dataset (https://osf.io/t98fz/). Single label dataset is available at https://tinyurl.com/SingleLabelVerSe2020

The script **convert_to_single_label_VerSe2020.py** could be used to convert images from multiple label into single label


- Target: Spine/Vertebra
- Task: Segmentation 
- Modality: CT

### Inputs

- 1 channel CT
- 3 channels (CT + foreground points + background points)

### Output

- 1 channel representing the Spine


![DeepEdit for spine/vertebra](../docs/images/sample-apps/deepedit_vertebra.png)
