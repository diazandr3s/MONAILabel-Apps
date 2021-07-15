import logging
import numpy as np

from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandHistogramShiftd,
    NormalizeIntensityd,
    Orientationd,
    RandSpatialCropd,
    RandAffined,
    RandRotated,
    Spacingd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def train_pre_transforms(self):
        # Train using resized transform + simple inferrer OR randomcrop randaffine and use sliding window inferrer
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Orientationd(keys=("image", "label"), axcodes="RAS"),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image"),
                RandHistogramShiftd(keys="image"),
                RandSpatialCropd(keys=("image", "label"), roi_size=(96, 96, 96)),
                RandAffined(
                    keys=("image", "label"),
                    mode=("bilinear", "nearest"),
                    prob=1.0,
                    spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1),
                ),
                RandRotated(keys=("image", "label"), range_x=[-10, 10], range_y=[-10, 10], range_z=[-10, 10]),
                ToTensord(keys=("image", "label")),
            ]
        )

    def train_post_transforms(self):
        return Compose(
            [
                ToTensord(keys=("pred", "label")),
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=("pred", "label"),
                    argmax=(True, False),
                    to_onehot=True,
                    n_classes=208,
                ),
            ]
        )

    def val_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Orientationd(keys=("image", "label"), axcodes="RAS"),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image"),
                CenterSpatialCropd(keys=("image", "label"), roi_size=[128, 128, 128]),
                ToTensord(keys=("image", "label")),
            ]
        )

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(96, 96, 96))
