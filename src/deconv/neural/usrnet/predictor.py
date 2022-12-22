import typing as tp

import numpy as np
import torch
from torch import nn

from src.deconv.neural.usrnet.model.model import USRNet
from src.deconv.neural.usrnet.utils.utils_image import single2tensor4


def load_weights(model: nn.Module, model_path: str):
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    return model


class USRNetPredictor(object):
    def __init__(
        self,
        model_path: str,
        scale_factor: int = 1,
        noise_level_model: float = 0,
        device: tp.Literal['cpu', 'cuda', 'auto'] = 'auto',
    ):
        self._device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            if device == 'auto'
            else torch.device(device)
        )

        model = USRNet(
            n_iter=8,
            h_nc=64,
            in_nc=4,
            out_nc=3,
            nc=[64, 128, 256, 512],
            nb=2, 
            act_mode="R", 
            downsample_mode='strideconv',
            upsample_mode="convtranspose",
        )
        self._model = load_weights(model=model, model_path=model_path).to(self._device)

        self._scale_factor = scale_factor
        self._noise_level_model = noise_level_model
    
    def forward(self, blurred_image: torch.tensor, psf: torch.tensor) -> np.array:
        blurred_image, psf = self._preprocess(blurred_image, psf)
        return self._forward(blurred_image, psf).cpu().numpy()
    
    def _forward(self, blurred_image: torch.tensor, psf: torch.tensor) -> torch.tensor:
        blurred_image = blurred_image.to(self._device)
        psf = psf.to(self._device)
        sigma = torch.tensor(self._noise_level_model).float().view([1, 1, 1, 1]).to(self._device)
        return self._model(blurred_image, psf, self._scale_factor, sigma)
    
    def _preprocess(self, blurred_image: np.array, psf: np.array) -> tp.Tuple[torch.tensor, torch.tensor]:
        blurred_image = torch.from_numpy(np.ascontiguousarray(blurred_image)).permute(2, 0, 1).float().unsqueeze(0).to(self._device)
        psf = single2tensor4(psf[..., np.newaxis]).to(self._device)
        return blurred_image, psf
