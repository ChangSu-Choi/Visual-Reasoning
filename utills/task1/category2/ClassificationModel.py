import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utills.task1.category2.ClassficationDataset import ClassficationDataset
from utills.task1.category2.get_data import get_data
from pytorch_metric_learning import losses
import torchvision
import timm
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

class ClassificationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.model_name == 'resnet50':
            self.model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
            self.model.fc = nn.Linear(self.model.fc.in_features, args.input_dim)
        elif self.args.model_name == 'vit':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.model.head = nn.Linear(768, args.input_dim)
        else:
            print('모델명을 잘못 입력하였음.')

        self.fc = nn.Sequential(
            nn.Linear(self.args.input_dim*4, self.args.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.args.mlp_hidden, 1),
        )

        self.criterion = nn.BCELoss()
               

    def forward(self, x):
        q1_imgs = self.model(x["q1_imgs"])
        q2_imgs = self.model(x["q2_imgs"])
        q3_imgs = self.model(x["q3_imgs"])
        a_img = self.model(x["a_img"])
        combined_imgs = torch.cat((q1_imgs, q2_imgs, q3_imgs, a_img), dim=1)
        return torch.sigmoid(self.fc(combined_imgs)).squeeze()

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        target_a1 = batch["target"].float()
        loss = self.criterion(logits, target_a1)


        predicted_a1 = torch.round(logits).detach().cpu().numpy()
        target_a1 = target_a1.detach().cpu().numpy()
        f1 = f1_score(target_a1, predicted_a1)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_f1", f1, prog_bar=True, on_step=True, sync_dist=True)

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer


    def train_dataloader(self):
        # 데이터셋 및 DataLoader 설정
        train_data, _ = get_data(self.args)
        train_df = pd.DataFrame(train_data)
        train_dataset = ClassficationDataset(train_df, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        return train_loader

    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        target_a1 = batch["target"].float()

        loss = self.criterion(logits, target_a1)

        predicted_a1 = torch.round(logits).detach().cpu().numpy()
        target_a1 = target_a1.detach().cpu().numpy()
        f1 = f1_score(target_a1, predicted_a1)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True)
        return {'loss': loss}


    def val_dataloader(self):
        _, valid_data = get_data(self.args)
        valid_df = pd.DataFrame(valid_data)
        valid_dataset = ClassficationDataset(valid_df, mode='valid')
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True)
        return valid_loader