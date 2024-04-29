import torch
import torch.nn.functional as F
import lightning as L
import torch.nn as nn
import torchmetrics as tm
from torchvision import transforms
from torchvision.utils import make_grid, draw_segmentation_masks
from matplotlib import pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101


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
    
def dice_loss(pred, target, smooth=1e-6, num_classes=22):
    pred = torch.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(pred * target, dim=(2,3))
    union = torch.sum(pred, dim=(2,3)) + torch.sum(target, dim=(2,3))
    dice_per_class = (2 * intersection + smooth) / (union + smooth)
    mean_dice = torch.mean(dice_per_class)
    return 1 - mean_dice

def mIoU(pred, target, smooth=1e-6):
    intersection = torch.logical_and(pred, target).sum(dim=(1,2)).float()
    union = torch.logical_or(pred, target).sum(dim=(1,2)).float()
    iou = (intersection + smooth) / (union + smooth)
    miou = torch.mean(iou)
    return miou

class model(L.LightningModule):
    def __init__(self, model_configs, optimizer_configs, lr_configs):
        super().__init__()
        self.save_hyperparameters()

        if model_configs["model_name"] == "deeplabv3_resnet50":
            print("Using DeepLabV3 with ResNet50 backbone")
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=weights)
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            self.model.classifier[4] = nn.Conv2d(256, model_configs["out_channels"], kernel_size=1)

        elif model_configs["model_name"] == "deeplabv3_resnet101":
            print("Using DeepLabV3 with ResNet50 backbone")
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
            self.model = deeplabv3_resnet101(weights=weights)
            self.model.classifier[4] = nn.Conv2d(256, model_configs["out_channels"], kernel_size=1)
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        elif model_configs["model_name"] == "fcn_resnet50":
            print("Using FCN with ResNet50 backbone")
            self.model = fcn_resnet50(pretrained=True, num_classes=model_configs["out_channels"])
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        elif model_configs["model_name"] == "fcn_resnet101":
            print("Using FCN with ResNet101 backbone")
            self.model = fcn_resnet101(pretrained=True, num_classes=model_configs["out_channels"])
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        elif model_configs["model_name"] == "UNet":
            print("Using UNet")
            self.conv1 = ConvBlock(model_configs["in_channels"], 64)
            self.down1 = DownSampling(64, 128)
            self.down2 = DownSampling(128, 256)
            self.down3 = DownSampling(256, 512)
            self.down4 = DownSampling(512, 1024)
            self.up1 = UpSampling(1024, 512)
            self.up2 = UpSampling(512, 256)
            self.up3 = UpSampling(256, 128)
            self.up4 = UpSampling(128, 64)
            self.final_conv = nn.Conv2d(in_channels=64, out_channels=model_configs["out_channels"], kernel_size=1, padding="same")
        else:
            raise ValueError("Model name not recognized. Using default model UNet")

        self.model_name = model_configs["model_name"]

        self.optimizer_configs = optimizer_configs
        self.lr_configs = lr_configs

        self.accuracy = tm.Accuracy(task="multiclass", num_classes=model_configs["out_channels"])
        self.precision = tm.Precision(task="multiclass", num_classes=model_configs["out_channels"])
        self.recall = tm.Recall(task="multiclass", num_classes=model_configs["out_channels"])
        self.f1 = tm.F1Score(task="multiclass", num_classes=model_configs["out_channels"])

        self.color_map = []
        for _ in range(model_configs["out_channels"]):
            self.color_map.append(torch.randint(0, 256, (3,)).tolist())
        self.color_map = torch.tensor(self.color_map)
        self.color_map[0] = torch.tensor([0, 0, 0])
        self.num_classes = model_configs["out_channels"]

    def forward(self, x):
        if self.model_name == "deeplabv3_resnet50" or self.model_name == "deeplabv3_resnet101" or self.model_name == "fcn_resnet50" or self.model_name == "fcn_resnet101":
            output = self.model(x)["out"]

        elif self.model_name == "smp_unet":
            output = self.model(x)

        elif self.model_name == "UNet":
            left1 = self.conv1(x)
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
        inputs, target, image = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target, label_smoothing=0.1)
        loss += dice_loss(output, target, num_classes=self.num_classes)
        target = target.type(torch.int64)
        output = torch.argmax(output, dim=1).type(torch.int64)
        metrics = {
            "train_acc": self.accuracy(output, target), 
            "train_prec": self.precision(output, target),
            "train_recall": self.recall(output, target), 
            "train_f1": self.f1(output, target),
            "train_iou": mIoU(output, target)
        }
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
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
            grid_image = make_grid(concatenated_images, nrow=6, padding=2, pad_value=1)
            grid_image = grid_image.unsqueeze(dim=0)
            self.logger.experiment.add_images("Validation", grid_image, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target, image = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target, label_smoothing=0.1)
        loss += dice_loss(output, target, num_classes=self.num_classes)
        target = target.type(torch.int64)
        output = torch.argmax(output, dim=1).type(torch.int64)
        metrics = {
            "valid_acc": self.accuracy(output, target), 
            "valid_prec": self.precision(output, target),
            "valid_recall": self.recall(output, target), 
            "valid_f1": self.f1(output, target),
            "valid_iou": mIoU(output, target)
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
            grid_image = make_grid(concatenated_images, nrow=6, padding=2, pad_value=1)
            grid_image = grid_image.unsqueeze(dim=0)
            self.logger.experiment.add_images("Validation", grid_image, self.current_epoch)
        
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target, image = batch
        output = self(inputs)
        # output = torch.randint(high=self.num_classes, size=(target.shape[0], self.num_classes, 224, 224)).to(self.device).type(torch.float32)
        loss = F.cross_entropy(output, target, label_smoothing=0.1)
        loss += dice_loss(output, target, num_classes=self.num_classes)
        target = target.type(torch.int64)
        output = torch.argmax(output, dim=1).type(torch.int64)
        metrics = {
            "test_acc": self.accuracy(output, target), 
            "test_prec": self.precision(output, target),
            "test_recall": self.recall(output, target), 
            "test_f1": self.f1(output, target),
            "test_iou": mIoU(output, target)
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
            grid_image = make_grid(concatenated_images, nrow=6, padding=2, pad_value=1)
            grid_image = grid_image.unsqueeze(dim=0)
            self.logger.experiment.add_images("Testing", grid_image, self.current_epoch)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_configs)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.lr_configs)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #         "frequency": 1,
        #     },
        # }
        return optimizer