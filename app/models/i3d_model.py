import torch
from assets.I3D_modelfiles.pytorch_i3d import InceptionI3d

class I3DModel:
    def __init__(self, weights_path, device):
        self.device = device

        self.model = InceptionI3d(400, in_channels=3)
        self.model.replace_logits(300)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def __call__(self, x):
        return self.model(x)