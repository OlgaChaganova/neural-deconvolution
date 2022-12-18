import numpy as np
from scipy import signal


def convolve(image: np.array, psf: np.array) -> np.array:
    """Convolve image and psf.

    Parameters
    ----------
    image : np.array
        Input image.
    psf : np.array
        Point spread function.

    Returns
    -------
    np.array
       Convolved image.
    """
    if image.ndim == 3 and psf.ndim != 3:  # gray image
        psf = psf[:, :, None]
    convolved_image = signal.convolve(image, psf, mode='same', method='fft')
    convolved_image = np.abs(convolved_image)
    return convolved_image / convolved_image.max()