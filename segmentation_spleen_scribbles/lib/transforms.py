import numpy as np
from monailabel.scribbles.transforms import InteractiveSegmentationTransform

from .utils import make_mideepnd_unary


class MakeMIDeepEGDUnaryd(InteractiveSegmentationTransform):
    def __init__(
        self,
        image: str,
        logits: str,
        scribbles: str,
        meta_key_postfix: str = "meta_dict",
        unary: str = "unary",
        tau: float = 1.0,
        scribbles_bg_label: int = 2,
        scribbles_fg_label: int = 3,
    ) -> None:
        super(MakeMIDeepEGDUnaryd, self).__init__(meta_key_postfix)
        self.image = image
        self.logits = logits
        self.scribbles = scribbles
        self.unary = unary
        self.tau = tau
        self.scribbles_bg_label = scribbles_bg_label
        self.scribbles_fg_label = scribbles_fg_label

    def _get_spacing(self, d, key):
        spacing = None
        src_key = "_".join([key, self.meta_key_postfix])
        if src_key in d.keys() and "affine" in d[src_key]:
            spacing = (np.sqrt(np.sum(np.square(d[src_key]["affine"]), 0))[
                       :-1]).astype(np.float32)

        return spacing

    def __call__(self, data):
        d = dict(data)

        # copy affine meta data from image input
        d = self._copy_affine(d, src=self.image, dst=self.unary)

        # read relevant terms from data
        image = self._fetch_data(d, self.image)
        logits = self._fetch_data(d, self.logits)
        scribbles = self._fetch_data(d, self.scribbles)

        # check if input logits are compatible with MIDeepSeg opt
        if logits.shape[0] > 2:
            raise ValueError(
                "MIDeepSeg can only be applied to binary probabilities for now, received {}".format(
                    logits.shape[0])
            )

        # attempt to unfold probability term
        logits = self._normalise_logits(logits, axis=0)
        spacing = self._get_spacing(d, self.image)

        unary_term = make_mideepnd_unary(
            image=image,
            prob=logits,
            scribbles=scribbles,
            scribbles_fg_label=self.scribbles_fg_label,
            scribbles_bg_label=self.scribbles_bg_label,
            spacing=spacing,
            tau=self.tau,
        )
        d[self.unary] = unary_term

        return d
