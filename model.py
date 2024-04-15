import torch
import torch.nn.functional as F
import lightning as L
import torch.nn as nn
import torchmetrics as tm
from torchvision import transforms
from torchvision.utils import make_grid, draw_segmentation_masks
from matplotlib import pyplot as plt


# model taken from here https://arxiv.org/abs/1505.04597

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", maxpool_kernel_size=2):
        super(DownSampling, self).__init__()
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size)
        self.downconv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.downconv(x)
        return x

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpSampling, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size, stride)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, left, up):
        up = self.upconv(up)
        transform = transforms.CenterCrop(up.shape[2])
        left = transform(left)
        x = torch.cat([left, up], dim=1)
        x = self.conv(x)
        return x
    
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice


class UNet(L.LightningModule):
    def __init__(self, UNet_configs, optimizer_configs):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = ConvBlock(UNet_configs["in_channels"], 64)
        self.down1 = DownSampling(64, 128)
        self.down2 = DownSampling(128, 256)
        self.down3 = DownSampling(256, 512)
        self.down4 = DownSampling(512, 1024)
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=UNet_configs["out_channels"], kernel_size=1, padding="same")

        self.optimizer_configs = optimizer_configs
        self.valid_accuracy = tm.Accuracy(task="multiclass", num_classes=UNet_configs["out_channels"])
        self.valid_precision = tm.Precision(task="multiclass", num_classes=UNet_configs["out_channels"])
        self.valid_recall = tm.Recall(task="multiclass", num_classes=UNet_configs["out_channels"])
        self.valid_f1 = tm.F1Score(task="multiclass", num_classes=UNet_configs["out_channels"])
        self.test_accuracy = tm.Accuracy(task="multiclass", num_classes=UNet_configs["out_channels"])
        self.test_precision = tm.Precision(task="multiclass", num_classes=UNet_configs["out_channels"])
        self.test_recall = tm.Recall(task="multiclass", num_classes=UNet_configs["out_channels"])
        self.test_f1 = tm.F1Score(task="multiclass", num_classes=UNet_configs["out_channels"])

        self.color_map = []
        for _ in range(UNet_configs["out_channels"]):
            self.color_map.append(torch.randint(0, 256, (3,)).tolist())
        self.color_map = torch.tensor(self.color_map)
        self.color_map[0] = torch.tensor([0, 0, 0])

    def forward(self, input):
        left1 = self.conv1(input)
        left2 = self.down1(left1)
        left3 = self.down2(left2)
        left4 = self.down3(left3)
        left5 = self.down4(left4)
        up1 = self.up1(left4, left5)
        up2 = self.up2(left3, up1)
        up3 = self.up3(left2, up2)
        up4 = self.up4(left1, up3)
        output = self.final_conv(up4)
        return output

    def training_step(self, batch, batch_idx):
        inputs, target, _ = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target)
        loss += dice_loss(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target, image = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target)
        loss += dice_loss(output, target)
        target = target.type(torch.int64)
        output = torch.argmax(output, dim=1)
        metrics = {
            "valid_acc": self.valid_accuracy(output, target), 
            "valid_prec": self.valid_precision(output, target),
            "valid_recall": self.valid_recall(output, target), 
            "valid_f1": self.valid_f1(output, target),
        }
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self.color_map = self.color_map.to(self.device)
            rgb_target = self.color_map[target].permute(0, 3, 1, 2).type(torch.uint8)
            rgb_output = self.color_map[output].permute(0, 3, 1, 2).type(torch.uint8)
            rgb_target = rgb_target.unsqueeze(dim=0)
            rgb_output = rgb_output.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
            concatenated_images = torch.cat((image, rgb_target, rgb_output), dim=0)
            concatenated_images = concatenated_images.transpose(0,1)
            concatenated_images = concatenated_images.reshape(-1, *concatenated_images.shape[2:])
            grid_image = make_grid(concatenated_images, nrow=6, normalize=True, padding=2, pad_value=1)
            grid_image = grid_image.unsqueeze(dim=0)
            self.logger.experiment.add_images("Validation", grid_image, self.current_epoch)
        
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target, image = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target)
        loss += dice_loss(output, target)
        target = target.type(torch.int64)
        output = torch.argmax(output, dim=1)
        metrics = {
            "test_acc": self.test_accuracy(output, target), 
            "test_prec": self.test_precision(output, target),
            "test_recall": self.test_recall(output, target), 
            "test_f1": self.test_f1(output, target),
        }
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self.color_map = self.color_map.to(self.device)
            rgb_target = self.color_map[target].permute(0, 3, 1, 2).type(torch.uint8)
            rgb_output = self.color_map[output].permute(0, 3, 1, 2).type(torch.uint8)
            rgb_target = rgb_target.unsqueeze(dim=0)
            rgb_output = rgb_output.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
            concatenated_images = torch.cat((image, rgb_target, rgb_output), dim=0)
            concatenated_images = concatenated_images.transpose(0,1)
            concatenated_images = concatenated_images.reshape(-1, *concatenated_images.shape[2:])
            grid_image = make_grid(concatenated_images, nrow=3, normalize=True, padding=2, pad_value=1)
            grid_image = grid_image.unsqueeze(dim=0)
            self.logger.experiment.add_images("Testing", grid_image, self.current_epoch)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizer_configs)