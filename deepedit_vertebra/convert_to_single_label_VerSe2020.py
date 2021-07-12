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
            result.append( d[key] > 0)
            # label 0 is background
            result.append(d[key] == 0)
            d[key] = np.stack(result, axis=0).astype(np.float64)
        # Output is only one channel
        d[key] = d[key][0, ...]
        print(d[key].shape)
        return d


def save_nifti(data, meta_data, path):

    filename = meta_data['filename_or_obj'][0].split('/')[-1].split('_seg')[0] + '.nii.gz'
    spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None
    original_affine = meta_data.get("original_affine", None)[0,0:3,0:3]
    affine = meta_data.get("affine", None)[0,0:3,0:3]

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # change data shape to be (channel, h, w, d)
    while len(data.shape) < 4:
        data = np.expand_dims(data, -1)

    # change data to "channel last" format and write to nifti format file
    data = np.moveaxis(np.asarray(data), 0, -1)

    dtype = np.float64
    output_dtype = np.float32

    write_nifti(
        data,
        file_name=path + filename,
        affine=affine,
        target_affine=original_affine,
        output_spatial_shape=spatial_shape,
        dtype=dtype,
        output_dtype=output_dtype,
    )

set_transforms = Compose(
                        [
                            LoadImaged(keys="label"),
                            ConvertToSingleLabelBrainClassd(keys=["label"]),
                        ]
                        )

data_dir = "INPUT_PATH/VerSe2020/all_imgs/"
train_labels = sorted(glob.glob(data_dir + "*/*_seg.nii.gz"))
train_d = [{"label": label_name} for label_name in train_labels]

print(len(train_d))

train_ds = monai.data.Dataset(data=train_d, transform=set_transforms)
trainLoader = DataLoader(train_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

output_folder = 'OUTPUT_PATH/VerSe2020/single_labels/'
for idx, img in enumerate(trainLoader):

    print('Processing label: ', idx + 1)
    print('Image name: ', img['label_meta_dict']['filename_or_obj'][0].split('/')[-1] )

    try:
        save_nifti(img['label'],
                   img['label_meta_dict'],
                   output_folder)
    except:
        pass