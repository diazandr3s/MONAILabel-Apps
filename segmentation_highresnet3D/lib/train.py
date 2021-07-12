import logging
import numpy as np

from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandShiftIntensityd,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    Resized,
    Spacingd,
    ToTensord,
)

from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def train_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                # AddChanneld(keys=("image", "label")),
                Orientationd(keys=("image", "label"), axcodes="RAS"),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image"),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=1.0,
                    spatial_size=(112, 112, 112),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1),
                ),
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
                # AddChanneld(keys=("image", "label")),
                Orientationd(keys=("image", "label"), axcodes="RAS"),
                Spacingd(
                    keys=("image", "label"),
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image"),
                Resized(keys=("image", "label"), spatial_size=(112, 112, 112), mode=("area", "nearest")),
                ToTensord(keys=("image", "label")),
            ]
        )

    def val_inferer(self):
        return SlidingWindowInferer(roi_size=(112, 112, 112))
