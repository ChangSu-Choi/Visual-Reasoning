import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utills.SimCLR_Dataset import SimCLR_Dataset
import pandas as pd
import numpy as np

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        self.tot_neg = 0

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
            
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

class SimCLR(pl.LightningModule):
    def __init__(self, model, criterion, args):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.args = args

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch["img_feature_1"] = batch["img_feature_1"].to(self.device)
        batch["img_feature_2"] = batch["img_feature_2"].to(self.device)

        x_vec = self.model(batch["img_feature_1"])
        y_vec = self.model(batch["img_feature_2"])

        loss = self.criterion(x_vec, y_vec)
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
        batch["img_feature_1"] = batch["img_feature_1"].to(self.device)
        batch["img_feature_2"] = batch["img_feature_2"].to(self.device)

        x_vec = self.model(batch["img_feature_1"])
        y_vec = self.model(batch["img_feature_2"])

        loss = self.criterion(x_vec, y_vec)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def val_dataloader(self):
        valid_df = pd.DataFrame(pd.read_csv(self.args.valid_link))  # 검증 데이터 경로를 사용하여 데이터프레임 생성
        valid_dataset = SimCLR_Dataset(valid_df, mode='valid')  # mode를 'valid'로 설정
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True)
        return valid_loader