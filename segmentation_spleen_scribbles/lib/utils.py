import logging

import GeodisTK
import numpy as np

logger = logging.getLogger(__name__)


def geodesic_distance_2d(I, S, lamda=1.0, iter=2):
    # lamda=0 : euclidean dist
    # lamda=1 : geodesic dist
    return GeodisTK.geodesic2d_raster_scan(I, S, lamda, iter)


def geodesic_distance_3d(I, S, spacing, lamda=1.0, iter=4):
    # lamda=0 : euclidean dist
    # lamda=1 : geodesic dist
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamda, iter)


def stable_softmax_foreground_return(x):
    # expects first dimension to have backgroundidx=0, foregroundidx=1
    # returns only worked out values for foreground, background can be reconstructed
    # as 1-Foreground in the calling function
    z = x - np.max(x, axis=0)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=0)
    # only return for foreground
    return numerator[1, ...] / denominator


def mideepegd(D_fg, D_bg, P_fg, tau=1):
    """
    Implementing Equation 3 to 7 from Section 2.4 page 6 of: https://arxiv.org/pdf/2104.12166.pdf
    Additionally adding temperature term (\tau) to exponential in Equation 3 and 4
    For replicating papers implementation use \tau==1
    """
    # do ExponentialGaussianDistance calculation
    D_g = np.array([D_bg, D_fg])
    E_fg = stable_softmax_foreground_return(-D_g / tau)

    # calculate alpha_i for each element to be updated
    # Equation 7
    D_g_min = np.min(D_g, axis=0)
    alpha_i = np.exp(-D_g_min)

    # update prob using alpha_i
    R_fg = (1 - alpha_i) * P_fg + alpha_i * E_fg
    return np.array([1 - R_fg, R_fg])


def make_mideepnd_unary(
    image,
    prob,
    scribbles,
    scribbles_fg_label,
    scribbles_bg_label,
    spacing,
    tau=1,
):
    # inputs are expected to be of format [1, X, Y, [Z]]
    # simple check to see if input shape is expected
    if image.shape[0] != 1 or scribbles.shape[0] != image.shape[0]:
        raise ValueError("unexpected input shape received")

    # only binary probabilities supported at the moment
    if prob.shape[0] != 2:
        raise ValueError("only binary probabilities are supported at the moment, received {}".format(prob.shape[0]))

    # extract spatial dims
    spatial_dims = image.ndim - 1
    run_3d = spatial_dims == 3

    P_fg = prob[1, ...]
    scribbles = np.squeeze(scribbles)
    image = np.squeeze(image)

    # get Geodesic Distances for foreground (D_fg) and background (D_bg) scribbles
    S_fg = (scribbles == scribbles_fg_label).astype(np.uint8)
    S_bg = (scribbles == scribbles_bg_label).astype(np.uint8)

    if run_3d:
        D_fg = geodesic_distance_3d(image, S_fg, spacing, lamda=1.0)
        D_bg = geodesic_distance_3d(image, S_bg, spacing, lamda=1.0)
    else:
        D_fg = geodesic_distance_2d(image, S_fg, lamda=1.0)
        D_bg = geodesic_distance_2d(image, S_bg, lamda=1.0)

    # run Equation 3 to 7
    unary = mideepegd(D_fg, D_bg, P_fg, tau=tau)  # \tau=1 for paper default

    return unary

