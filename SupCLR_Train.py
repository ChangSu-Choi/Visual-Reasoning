import argparse
import numpy as np
import os
import pytorch_lightning as pl
import subprocess
import timm
import torch
import torchvision
import yaml

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from utills.SupCLR import SupCLR, SupCLR_Loss

def set_up(args):
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

def get_model(args):
    if args.model_name == 'resnet50':
        model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, args.input_dim)
    elif args.model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.head = nn.Linear(768, args.input_dim)
    return model

def create_checkpoint():
    return ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        save_last=True,
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        dirpath='checkpoint/SupCLR',
    )

def create_logger(args):
    return TensorBoardLogger("tb_logs/SupCLR", name=f"SupCLR-{args.model_name}-bs{args.batch_size}-epochs{args.epochs}")

def create_trainer(args, checkpoint_callback, logger):
    trainer_args = {
        'max_epochs': args.epochs, 
        'log_every_n_steps': args.log_every_n_steps, 
        'logger': logger, 
        'callbacks': [checkpoint_callback], 
        'devices': args.devices, 
        'strategy': "ddp"
    }

    if args.resume_from_checkpoint is not None:
        trainer_args['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    return pl.Trainer(**trainer_args)

def lower_cpu_priority():
    pid = os.getpid()
    result = subprocess.run(['renice', '15', '-p', f'{pid}'],capture_output=True,text=True)
    if result.returncode == 0:
        print("Successfully lowered CPU priority.")
    else:
        print("Failed to lower CPU priority.")

def main(args):
    set_up(args)
    model = get_model(args)
    criterion = SupCLR_Loss(args.clr_temperature)
    supclr = SupCLR(model, criterion, args)
    checkpoint_callback = create_checkpoint()
    logger = create_logger(args)
    trainer = create_trainer(args, checkpoint_callback, logger)
    trainer.fit(supclr)

def get_config():
    with open("utills/config.yaml", "r") as f:
        return yaml.safe_load(f)

def parse_args(config):
    parser = argparse.ArgumentParser(description="SupCLR python code")
    parser.add_argument("--input_dim", type=int, default=config['input_dim'])
    parser.add_argument("--mlp_hidden", type=int, default=config['mlp_hidden'])
    parser.add_argument("--batch_size", type=int, default=config['batch_size'])
    parser.add_argument("--model_name", type=str, default=config['model_name'])
    parser.add_argument("--mode", type=str, default=config['mode'])
    parser.add_argument("--num_workers", type=str, default=config['num_workers'])
    parser.add_argument("--devices", default=config['devices'])
    parser.add_argument("--epochs", type=int, default=config['epochs'])
    parser.add_argument("--clr_temperature", type=int, default=config['clr_temperature'])
    parser.add_argument("--learning_rate", type=int, default=config['learning_rate'])
    parser.add_argument("--log_every_n_steps", type=int, default=config['log_every_n_steps'])
    parser.add_argument("--model_save_path", type=str, default=config['model_save_path'])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--train_link", type=str, default=config['task1_category2_train'])
    parser.add_argument("--valid_link", type=str, default=config['task1_category2_valid'])
    return parser.parse_args()

if __name__ == "__main__":
    config = get_config()
    args = parse_args(config)
    if args.devices != -1:
        lower_cpu_priority()
    main(args)
