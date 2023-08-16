import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utills.task1.category3.ClassficationDataset import ClassficationDataset
from utills.task1.category3.get_data import get_data
from torch.utils.data.distributed import DistributedSampler

import torchvision
import timm
import pandas as pd
import torchmetrics

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
            nn.Linear(self.args.input_dim*6, self.args.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.args.mlp_hidden, 1),
        ) 
        self.criterion = nn.BCELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

        self.train_f1 = torchmetrics.F1Score(num_classes=1, task='binary', threshold=0.5)  # for training
        self.val_f1 = torchmetrics.F1Score(num_classes=1, task='binary', threshold=0.5)  # for validation
               

    def forward(self, x):

        keys = ['q1_imgs', 'q2_imgs', 'a1_imgs', 'a2_imgs', 'a3_imgs']

        img_features = {}
        for key in keys:
            img_features[key] = [self.model(img) for img in x[key]]

        
        q_a1 = torch.cat(img_features['q1_imgs'] + img_features['a1_imgs'], axis=1)
        q_a2 = torch.cat(img_features['q1_imgs'] + img_features['a2_imgs'], axis=1)
        q_a3 = torch.cat(img_features['q1_imgs'] + img_features['a3_imgs'], axis=1)
        # q12 = torch.cat(img_features['q1_imgs'] + img_features['q2_imgs'], axis=1)

        
        q_a1_logit = self.fc(q_a1).squeeze()
        q_a2_logit = self.fc(q_a2).squeeze()
        q_a3_logit = self.fc(q_a3).squeeze()

        return {
            "q_a1_logit": torch.sigmoid(q_a1_logit),
            "q_a2_logit": torch.sigmoid(q_a2_logit),
            "q_a3_logit": torch.sigmoid(q_a3_logit),
            "q1_feature": img_features['q1_imgs'],
            "q2_feature": img_features['q2_imgs'],
        }

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch)

        target_a1, target_a2, target_a3 = (batch["target"] == 1).float(), (batch["target"] == 2).float(), (batch["target"] == 3).float()
        target_neg = (batch["target"] == 4).float() ## 모두 0
        loss_a1 = self.criterion(logits["q_a1_logit"], target_a1)
        loss_a2 = self.criterion(logits["q_a2_logit"], target_a2)
        loss_a3 = self.criterion(logits["q_a3_logit"], target_a3)
        cosine_loss_q12 = self.cosine_loss(logits['q1_feature'], logits['q2_feature'], torch.tensor(-1).to(self.device))


        loss = (loss_a1 + loss_a2 + loss_a3 + cosine_loss_q12*0.2) / 4


        predicted_a1 = torch.round(logits["q_a1_logit"])
        predicted_a2 = torch.round(logits["q_a2_logit"])
        predicted_a3 = torch.round(logits["q_a3_logit"])



        # Update F1 score
        self.train_f1.update(predicted_a1, target_a1)
        self.train_f1.update(predicted_a2, target_a2)
        self.train_f1.update(predicted_a3, target_a3)


        # Compute F1 score at the end of the epoch
        f1 = self.train_f1.compute()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["target"].shape[0])
        self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch["target"].shape[0])


        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer


    def train_dataloader(self):
        # 데이터셋 및 DataLoader 설정
        train_data, _ = get_data(self.args)
        train_df = pd.DataFrame(train_data)
        train_dataset = ClassficationDataset(train_df, mode='train')
        
        
        # # Add the DistributedSampler
        # train_sampler = DistributedSampler(train_dataset)
        # train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=self.args.num_workers, drop_last=True)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        return train_loader

    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)

        target_a1, target_a2, target_a3 = (batch["target"] == 1).float(), (batch["target"] == 2).float(), (batch["target"] == 3).float()
        target_neg = (batch["target"] == 4).float() ## 모두 0
        loss_a1 = self.criterion(logits["q_a1_logit"], target_a1)
        loss_a2 = self.criterion(logits["q_a2_logit"], target_a2)
        loss_a3 = self.criterion(logits["q_a3_logit"], target_a3)
        cosine_loss_q12 = self.cosine_loss(logits['q1_feature'], logits['q2_feature'], torch.tensor(-1))


        loss = (loss_a1 + loss_a2 + loss_a3 + cosine_loss_q12*0.2) / 4

        predicted_a1 = torch.round(logits["q_a1_logit"])
        predicted_a2 = torch.round(logits["q_a2_logit"])
        predicted_a3 = torch.round(logits["q_a3_logit"])


        # Update F1 score
        self.val_f1.update(predicted_a1, target_a1)
        self.val_f1.update(predicted_a2, target_a2)
        self.val_f1.update(predicted_a3, target_a3)

        # Compute F1 score at the end of the epoch
        f1 = self.val_f1.compute()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch["target"].shape[0])
        self.log("val_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch["target"].shape[0])

        return {'loss': loss}


    def val_dataloader(self):
        _, valid_data = get_data(self.args)
        valid_df = pd.DataFrame(valid_data)
        valid_dataset = ClassficationDataset(valid_df, mode='valid')

        #  # Add the DistributedSampler
        # val_sampler = DistributedSampler(valid_dataset, shuffle=False)
        # valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=val_sampler, num_workers=self.args.num_workers, drop_last=True)
        
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True)
        return valid_loader