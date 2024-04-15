import os
import torch
import torch.utils.data as du
from torch.utils.data import random_split
import lightning as L
import yaml
from model import UNet
from dataset import BreastCancerDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import warnings


def main(yaml_file_dir="configs/config.yaml"):

    torch.manual_seed(42)

    warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.*")
    #load config file
    cfg = yaml.safe_load(open(yaml_file_dir))

    # load dataset 70/10/20 split
    print("Loading dataset...")
    train_dataset = BreastCancerDataset(cfg["train_dataset"])
    valid_dataset = BreastCancerDataset(cfg["valid_dataset"])
    print("Finished loading dataset.")

    # split the dataset into a validation/test set 
    print("Splitting dataset...")
    valid_dataset_size =  len(valid_dataset)
    test_dataset_size = int(valid_dataset_size * 0.5)
    valid_dataset_size = valid_dataset_size - test_dataset_size
    valid_dataset, test_dataset = random_split(valid_dataset, [valid_dataset_size, test_dataset_size])
    print("Finished splitting dataset.")

    print("Creating dataloaders...")
    train_loader = du.DataLoader(train_dataset, **cfg["train_dataloader"])
    valid_loader = du.DataLoader(valid_dataset, **cfg["valid_dataloader"])
    test_loader = du.DataLoader(test_dataset, **cfg["test_dataloader"])
    print("Finished creating dataloaders.")

    # load model
    print("Loading model...")
    model = UNet(cfg["unet"], cfg["unet_optimizer"])
    print("Finished loading model.")

    checkpoint_callback = ModelCheckpoint(**cfg["callbacks"]["model_checkpoint"])

    # train model
    if cfg["logger"] is not None:
        logger = TensorBoardLogger(**cfg["logger"])
        trainer = L.Trainer(**cfg["trainer"], logger=logger, callbacks=[checkpoint_callback])
    else:
        trainer = L.Trainer(**cfg["trainer"], callbacks=[checkpoint_callback])


    print("Training model...")

    trainer.fit(model, train_loader, valid_loader, ckpt_path=cfg["last_checkpoint"])
    print("Finished training model.")

    print("Testing model...")
    trainer.test(model, test_loader)
    print("Finished testing model.")

def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--config', dest='config', default='configs/config.yaml')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
    






