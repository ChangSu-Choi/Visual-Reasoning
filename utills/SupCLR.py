import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utills.SupCLR_Dataset import SimCLR_Dataset
from pytorch_metric_learning import losses
import torch.nn.functional as F
import pandas as pd
import numpy as np

class SupCLR_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature


    def forward(self, feature_vectors, labels):

        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        feature_vectors_normalized = feature_vectors_normalized.view(feature_vectors_normalized.size(0), -1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )

        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))

class SupCLR(pl.LightningModule):
    def __init__(self, model, criterion, args):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.args = args

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)

        images_feature = self.model(images)
        loss = self.criterion(images_feature, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer


    def train_dataloader(self):
        # 데이터셋 및 DataLoader 설정
        train_df = pd.DataFrame(pd.read_csv(self.args.train_link))
        train_dataset = SimCLR_Dataset(train_df, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        return train_loader

    
    def validation_step(self, batch, batch_idx):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)

        images_feature = self.model(images)
        loss = self.criterion(images_feature, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def val_dataloader(self):
        valid_df = pd.DataFrame(pd.read_csv(self.args.valid_link))  # 검증 데이터 경로를 사용하여 데이터프레임 생성
        valid_dataset = SimCLR_Dataset(valid_df, mode='valid')  # mode를 'valid'로 설정
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True)
        return valid_loader