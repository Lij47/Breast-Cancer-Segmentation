import os
import torch
import torch.utils.data as du
from torch.utils.data import random_split
import lightning as L
import yaml
from model import model
from dataset import BreastCancerDataset
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import multiprocessing
import argparse


def main(yaml_file_dir="configs/config.yaml"):

    torch.manual_seed(42)

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

    # loading data into dataloaders
    print("Creating dataloaders...")
    num_cpus = multiprocessing.cpu_count()
    train_loader = du.DataLoader(train_dataset, **cfg["train_dataloader"], num_workers=num_cpus)
    valid_loader = du.DataLoader(valid_dataset, **cfg["valid_dataloader"], num_workers=num_cpus)
    test_loader = du.DataLoader(test_dataset, **cfg["test_dataloader"], num_workers=num_cpus)
    print("Finished creating dataloaders.")

    # load model
    print("Loading model...")
    net = model(cfg["model"], cfg["optimizer"], cfg["lr_scheduler"])
    print(net)
    print("Finished loading model.")

    checkpoint_callback = ModelCheckpoint(**cfg["callbacks"]["model_checkpoint"])
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    earlystopping_callback = EarlyStopping(**cfg["callbacks"]["early_stopping"])
    callbacks = [checkpoint_callback, lr_callback, earlystopping_callback]


    # train model
    if cfg["logger"] is not None:
        logger = TensorBoardLogger(**cfg["logger"])
        trainer = L.Trainer(**cfg["trainer"], logger=logger, callbacks=callbacks)
    else:
        trainer = L.Trainer(**cfg["trainer"], callbacks=callbacks)


    print("Training model...")

    trainer.fit(net, train_loader, valid_loader, ckpt_path=cfg["last_checkpoint"])
    print("Finished training model.")

    print("Testing model...")
    trainer.test(net, test_loader)
    print("Finished testing model.")

def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--config', dest='config', default='configs/config.yaml')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
    






