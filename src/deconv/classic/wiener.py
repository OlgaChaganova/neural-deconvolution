import logging

import numpy as np
from skimage import restoration


def wiener_gray(blurred_image: np.array, psf: np.array, **algo_params) -> np.array:
    """Apply Wiener deconvolution for one-channel image.

    Parameters
    ----------
    blurred_image : np.array
        Blurred one-channel image.
    psf : np.array
        PSF.

    Returns
    -------
    np.array
        Restored one-channel image.
    """
    if psf.sum() != 1:
        psf /= psf.sum()
        logging.warning('PSF has sum more than 1. Normed')
    restored = restoration.wiener(image=blurred_image, psf=psf, **algo_params)
    return restored / restored.max()


def wiener_rgb(blurred_image: np.array, psf: np.array, **algo_params) -> np.array:
    """Apply Wiener deconvolution for RGB image per channel.

    Parameters
    ----------
    blurred_image : np.array
        Blurred RGB image.
    psf : np.array
        PSF.

    Returns
    -------
    np.array
        Restored RGB image.
    """
    rgb_restored = []
    for i in range(blurred_image.shape[-1]):
        # restored = restoration.wiener(blurred_image[:, :, i], psf, **algo_params)
        rgb_restored.append(wiener_gray(blurred_image[:, :, i], psf, **algo_params))
    rgb_restored = np.stack(rgb_restored)
    return np.transpose(rgb_restored, (1, 2, 0))
