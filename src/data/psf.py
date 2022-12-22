import sys

import numpy as np

from src.data.convolution import shift

# external code source
sys.path.append('/home/chaganovaob/edu/trunk')
from sweet.sweet import Sweet


def get_psf(
    s: float,
    a: float,
    c: float,
    pupil_diam: float,
    view_dist: float,
    canvas_size_h: float,
) -> np.array:
    """Get PSF for eye with specific parameters."""
    sweet = Sweet()
    sweet.set_eye_prescription(S=s, A=a, C=c)
    sweet.set_experiment_params(
        pupil_diam=pupil_diam,  # Pupil diameter in the experiment (in mm)
        view_dist=view_dist,  # Viewing distance, how far is the participant from the monitor (in cm)
        canvas_size_h=canvas_size_h,  # Canvas size,how big is image (with padding) on monitor (in cm)
    )
    return shift(sweet._psf())
