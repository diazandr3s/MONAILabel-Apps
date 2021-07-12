#  Heart ventricles segmentation using DynUNet

### Model Overview

Standard (non-interactive) MONAI Label App using [DynUNet](https://docs.monai.io/en/latest/_modules/monai/networks/nets/dynunet.html) to label left and right ventricle over MR (SAX) Images

### Data

We used 15 cardiac MR images (both ED and ES) to train this model. In order to check the performance of this system, researchers can use images from the [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html) or [M&M](https://www.ub.edu/mnms/) dataset.

- Target: Left and right ventricles of the heart
- Task: Segmentation 
- Modality: MR (SAX)

### Input

- 1 channel MR

### Output

- 4 channel representing background, right ventricle (RV), left ventricle (LV), and LV wall

![Heart chambers segmentation](../../docs/images/sample-apps/segmentation_heart_ventricles.png)
