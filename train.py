import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer, LightningModule
from data.text_image_dm import TextImageDataModule, TextImageDataset
from models import CLIPWrapper
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.cli import LightningCLI


def main(hparams):
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    model = CLIPWrapper(hparams.model_name, config, hparams.minibatch_size)
    del hparams.model_name
    dm = TextImageDataModule(folder=hparams.folder, batch_size=hparams.batch_size, image_size=hparams.image_size, resize_ratio=hparams.resize_ratio, shuffle=hparams.shuffle)
    trainer = Trainer(precision=16, max_epochs=1, accelerator="gpu", devices=8, default_root_dir="checkpoints")
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)