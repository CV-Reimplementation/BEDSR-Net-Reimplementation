import argparse
import os

import torch.optim as optim
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper_bedsrnet import infer
from libs.loss_fn import get_criterion
from libs.models import get_model
from libs.seed import set_seed

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    result_path = os.path.dirname('results')
    
    set_seed()
    device = get_device(allow_only_gpu=False)

    val_transform = Compose(
        [
            Resize(512, 512),
            Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )

    val_loader = get_dataloader(
        'Adobe',
        'bedsrnet',
        "test",
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        transform=val_transform,
    )

    lambda_dict = {"lambda1": 1.0, "lambda2": 0.01}
    criterion = get_criterion('GAN', device)

    benet = get_model("benet", in_channels=3, pretrained=True)
    srnet = get_model("srnet", pretrained=True)
    generator, discriminator = srnet[0], srnet[1]

    benet.eval()
    benet.to(device)
    generator.to(device)
    discriminator.to(device)

    infer(
        val_loader, generator, discriminator, benet, criterion, lambda_dict, device
    )

