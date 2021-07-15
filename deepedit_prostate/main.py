import logging
import os
import glob

from lib import Deepgrow, MyStrategy, MyTrain, Segmentation, dynunet_strides_kernels
from monai.networks.nets.dynunet_v1 import DynUNetV1

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random

logger = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        strides, kernels = dynunet_strides_kernels.dynUNet_strides_kernels(patch_size=(192, 192, 32), target_spacing=(1.0, 1.0, 3.0))
        self.network = DynUNetV1(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1: ],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        self.model_dir = os.path.join(app_dir, "model")
        self.pretrained_model = os.path.join(self.model_dir, "pretrained.pt")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.download(
            [
                (
                    self.pretrained_model,
                    "https://github.com/Project-MONAI/MONAILabel/releases/download/data/deepedit_prostate.pt",
                ),
            ]
        )

        super().__init__(app_dir, studies, os.path.join(self.model_dir, "train_stats.json"))

    def init_infers(self):
        return {
            "deepedit": Deepgrow([self.pretrained_model, self.final_model], self.network),
            "prostate": Segmentation([self.pretrained_model, self.final_model], self.network),
        }

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }

    def train(self, request):
        logger.info(f"Training request: {request}")

        output_dir = os.path.join(self.model_dir, request.get("name", "model_01"))

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        load_path = os.path.join(output_dir, "model.pt")
        if not os.path.exists(load_path) and request.get("pretrained", True):
            load_path = self.pretrained_model

        # Datalist for train/validation

        # Training images
        train_d = self.datastore().datalist()
        # Validation images
        data_dir = r"" # directory containing validation images
        val_images = sorted(glob.glob(os.path.join(data_dir, "imagesVal", "*.nii.gz")))
        val_labels = sorted(glob.glob(os.path.join(data_dir, "labelsVal", "*.nii.gz")))
        val_d = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]
        
        task = MyTrain(
            output_dir=output_dir,
            train_datalist=train_d,
            val_datalist=val_d,
            network=self.network,
            load_path=load_path,
            publish_path=self.final_model,
            stats_path=self.train_stats_path,
            device=request.get("device", "cuda"),
            lr=request.get("lr", 0.0001),
            max_epochs=request.get("epochs", 1),
            amp=request.get("amp", True),
            train_batch_size=request.get("train_batch_size", 1),
            val_batch_size=request.get("val_batch_size", 1),
        )
        return task()