import numpy as np
from skimage import restoration


def richardson_lucy_gray(blurred_image: np.array, psf: np.array, **algo_params) -> np.array:
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
    return restoration.richardson_lucy(image=blurred_image, psf=psf, **algo_params)


def richardson_lucy_rgb(blurred_image: np.array, psf: np.array, **algo_params) -> np.array:
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
        rgb_restored.append(restoration.richardson_lucy(blurred_image[:, :, i], psf, **algo_params))
    rgb_restored = np.stack(rgb_restored)
    return np.transpose(rgb_restored, (1, 2, 0))