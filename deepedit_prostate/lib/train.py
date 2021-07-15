import logging
import numpy as np

from monai.apps.deepgrow.interaction import Interaction
from monai.apps.deepgrow.transforms import (
    AddGuidanceSignald,
    AddInitialSeedPointd,
    AddRandomGuidanced,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandFlipd,
    Resized,
    Spacingd,
    CastToTyped,
    ToNumpyd,
    ToTensord,
)

from monailabel.deepedit.transforms import DiscardAddGuidanced
from monailabel.utils.train.basic_train import BasicTrainTask

logger = logging.getLogger(__name__)


class MyTrain(BasicTrainTask):
    def __init__(
        self,
        output_dir,
        train_datalist,
        val_datalist,
        network,
        model_size=(192, 192, 32),
        max_train_interactions=20,
        max_val_interactions=4,
        **kwargs,
    ):
        super().__init__(output_dir, train_datalist, val_datalist, network, **kwargs)

        self.model_size = model_size
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions

    def train_click_transforms(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                ToNumpyd(keys=("image", "label", "pred", "probability", "guidance")),
                FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
                AddRandomGuidanced(
                    guidance="guidance", discrepancy="discrepancy", probability="probability"),
                AddGuidanceSignald(image="image", guidance="guidance", sigma=[2.0, 2.0, 2.0/3.0]),
                DiscardAddGuidanced(image="image", probability=0.5),
                ToTensord(keys=("image", "label")),
            ]
        )
            
    def val_click_transforms(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                ToNumpyd(keys=("image", "label", "pred", "probability", "guidance")),
                FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy"),
                AddRandomGuidanced(
                    guidance="guidance", discrepancy="discrepancy", probability="probability"),
                AddGuidanceSignald(image="image", guidance="guidance", sigma=[2.0, 2.0, 2.0/3.0]),
                DiscardAddGuidanced(image="image", probability=0.0),
                ToTensord(keys=("image", "label")),
            ]
        )

    def loss_function(self):
        return DiceLoss(sigmoid=True)

    def train_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=['image', 'label']),
                Orientationd(keys=["image", "label"], axcodes="LPS"),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys="image"),
                RandAffined(keys=('image', 'label'), prob=0.15, 
                        rotate_range=((0, 0), (0, 0), (-1, 1)), 
                        scale_range=((-0.3, 0.4), (-0.3, 0.4), (0.0, 0.0)), 
                        mode=('bilinear', 'nearest'), as_tensor_output=False),
                RandGaussianNoised(keys=['image'], std=0.05, prob=0.15),
                RandScaleIntensityd(keys=['image'], factors=0.3, prob=0.15),
                RandFlipd(['image', 'label'], spatial_axis=[0], prob=0.5),
                Resized(keys=("image", "label"), spatial_size=self.model_size, mode=("area", "nearest")),
                CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
                FindAllValidSlicesd(label="label", sids="sids"),
                AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
                AddGuidanceSignald(image="image", guidance="guidance", sigma=[2.0, 2.0, 2.0/3.0]), # sigma to reflect image anisotropy
                DiscardAddGuidanced(image="image", probability=0.5),
                ToTensord(keys=("image", "label")),
            ]
        )

    def train_post_transforms(self):
        return Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ]
        )

    def val_pre_transforms(self):
        return Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=['image', 'label']),
                Orientationd(keys=["image", "label"], axcodes="LPS"),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "nearest")),
                NormalizeIntensityd(keys="image"),
                Resized(keys=("image", "label"), spatial_size=self.model_size, mode=("area", "nearest")),
                CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
                FindAllValidSlicesd(label="label", sids="sids"),
                AddInitialSeedPointd(label="label", guidance="guidance", sids="sids"),
                AddGuidanceSignald(image="image", guidance="guidance", sigma=[2.0, 2.0, 2.0/3.0]), # sigma to reflect image anisotropy
                DiscardAddGuidanced(image="image", probability=0.0),
                ToTensord(keys=("image", "label")),
            ]
        )

    def val_inferer(self):
        return SimpleInferer()

    def train_iteration_update(self):
        return Interaction(
            transforms=self.train_click_transforms(),
            max_interactions=self.max_train_interactions,
            key_probability="probability",
            train=True,
        )

    def val_iteration_update(self):
        return Interaction(
            transforms=self.val_click_transforms(),
            max_interactions=self.max_val_interactions,
            key_probability="probability",
            train=False,
        )
