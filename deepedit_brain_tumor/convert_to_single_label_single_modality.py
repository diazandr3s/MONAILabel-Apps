import os
import glob
from typing import Dict
import numpy as np
import torch
import monai
from monai.transforms.transform import MapTransform, Transform
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
from monai.data.dataset import PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
)
from monai.data.nifti_writer import write_nifti
from monai.utils import ImageMetaKey as Key
from monai.utils import GridSampleMode, GridSamplePadMode

# Define a new transform to convert brain tumor labels
class ConvertToSingleLabelBrainClassd(MapTransform):
    """
    Convert labels to one channel
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge labels 1, 2 and 3
            # result.append(np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # merge labels 2 and 3. Not 1 for Edema
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # label 0 is background
            result.append(d[key] == 0)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        # Output is only one channel
        d[key] = d[key][0,0,...]
        return d

# Define a new transform to get single modality
class ConvertToSingleModalityd(MapTransform):
    """
    Gets one modality
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Output is only one channel
            # Get T1 Gadolinium. Better to describe brain tumour. FLAIR is better for edema (swelling in the brain)
            d[key] = d[key][2, ...]
            # d[key] = d[key][None]
        return d


def save_nifti(data, meta_data, path):

    filename = meta_data['filename_or_obj'][0].split('/')[-1]
    spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None
    original_affine = meta_data.get("original_affine", None)[0,...]
    affine = meta_data.get("affine", None)[0,...]

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # change data shape to be (channel, h, w, d)
    while len(data.shape) < 4:
        data = np.expand_dims(data, -1)

    # change data to "channel last" format and write to nifti format file
    data = np.moveaxis(np.asarray(data), 0, -1)

    resample = True
    mode = GridSampleMode.BILINEAR
    padding_mode = GridSamplePadMode.BORDER
    align_corners = False
    dtype = np.float64
    output_dtype = np.float32

    write_nifti(
        data,
        file_name=path + filename,
        affine=affine,
        target_affine=original_affine,
        resample=resample,
        output_spatial_shape=spatial_shape,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        dtype=dtype,
        output_dtype=output_dtype,
    )

set_transforms = Compose(
                        [
                            LoadImaged(keys=["image", "label"]),
                            EnsureChannelFirstd(keys=["image", "label"]),
                            ConvertToSingleLabelBrainClassd(keys=["label"]),
                            ConvertToSingleModalityd(keys=["image"]),
                        ]
                        )

data_dir = "INPUT_PATH/MSDecathlon/Task01_BrainTumour"

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

train_d = [{"image": image_name, "label": label_name} for image_name, label_name in
            zip(train_images, train_labels)]

print(len(train_d))

train_ds = monai.data.Dataset(data=train_d, transform=set_transforms)
trainLoader = DataLoader(train_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

output_folder = 'OUTPUT_PATH/MSDecathlon/Task01_BrainTumour_single/'
for idx, img in enumerate(trainLoader):

    print('Processing image: ', idx + 1 )

    save_nifti(img['image'],
               img['image_meta_dict'],
               output_folder + 'imagesTr/')

    save_nifti(img['label'],
               img['label_meta_dict'],
               output_folder + 'labelsTr/')