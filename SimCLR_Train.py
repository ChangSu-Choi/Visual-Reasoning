import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision
import torch
import torch.nn as nn
from utills.SimCLR import SimCLR_Loss
from utills.SimCLR import SimCLR
import numpy as np
import timm
import argparse
import yaml
import os
import subprocess


def main(args):
    # 정확도 < 속도
    torch.set_float32_matmul_precision('medium')
    # 정확도 < 속도
    torch.set_float32_matmul_precision('medium')
    #파이토치의 랜덤시드 고정
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # 넘파이 랜덤시드 고정
    np.random.seed(0)
    # Config 및 기타 설정
    if args.model_name == 'resnet50':
        model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, args.input_dim)
    elif args.model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.head = nn.Linear(768, args.input_dim)

    criterion = SimCLR_Loss(args.batch_size, args.clr_temperature)
    simclr = SimCLR(model, criterion, args)

    # ModelCheckpoint 생성 및 설정
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 추적할 지표 설정
        mode='min',  # 지표를 최소화하도록 설정
        save_top_k=2,  # 베스트 모델 k개를 저장, 1로 설정하면 베스트 모델 1개만 저장
        save_last=True,  # 마지막 모델도 저장
        filename='best-model-{epoch:02d}-{val_loss:.2f}',  # 저장될 파일명 형식
        dirpath='checkpoint/SimCLR',  # 체크포인트 저장 경로
    )
    
    # 텐서 보드 
    logger = TensorBoardLogger("tb_logs/SimCLR", name=f"SimCLR-{args.model_name}-bs{args.batch_size}-epochs{args.epochs}")
    # 체크포인트부터 학슬 할 건지 아닌지
    if args.resume_from_checkpoint is None:
        trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=args.log_every_n_steps, logger=logger, callbacks=[checkpoint_callback], devices=args.devices, strategy="ddp")
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=args.log_every_n_steps, logger=logger, callbacks=[checkpoint_callback], devices=args.devices, strategy="ddp", resume_from_checkpoint=checkpoint_path)
    trainer.fit(supclr)

if __name__ == "__main__":
    with open("utills/config.yaml", "r") as f:
        config = yaml.safe_load(f)


    parser = argparse.ArgumentParser(description="simCLR python code")
    parser.add_argument("--input_dim", type=int, default=config['input_dim'], help="Path to the input file.")
    parser.add_argument("--mlp_hidden", type=int, default=config['mlp_hidden'], help="Path to the output file.")
    parser.add_argument("--batch_size", type=int, default=config['batch_size'], help="Number of training epochs.")
    parser.add_argument("--model_name", type=str, default=config['model_name'], help="resnet50 or vit")
    parser.add_argument("--mode", type=str, default=config['mode'], help="resnet50 or vit")
    parser.add_argument("--num_workers", type=str, default=config['num_workers'], help="num_workers for the trainer")
    parser.add_argument("--devices", default=config['devices'], help="gpus for the trainer")
    parser.add_argument("--epochs", type=int, default=config['epochs'], help="epochs for the trainer")
    parser.add_argument("--clr_temperature", type=int, default=config['clr_temperature'], help="epochs for the trainer")
    parser.add_argument("--learning_rate", type=int, default=config['learning_rate'], help="learning-rate for the trainer")
    parser.add_argument("--log_every_n_steps", type=int, default=config['log_every_n_steps'], help="log_every_n_steps for the trainer")
    parser.add_argument("--model_save_path", type=str, default=config['model_save_path'], help="Path to save the trained model weights.")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume training from.")
    parser.add_argument("--train_link", type=str, default=config['task1_category2_train'], help="train_link for the trainer")
    parser.add_argument("--valid_link", type=str, default=config['task1_category2_valid'], help="valid_link for the trainer")

    args = parser.parse_args()

    if args.devices == -1:
        pass
    else:
        ## cpu stuck이 나버려서 우선순위를 내려 보려고 함
        # 현재 프로세스의 PID 확인
        pid = os.getpid()
        result = subprocess.run(['renice', '15', '-p', f'{pid}'],capture_output=True,text=True)
        if result.returncode == 0:
            print("스케쥴러 우선순위 내림:")
            print(result.stdout)
        else:
            print("스케쥴러 우선순위 내림 실패:")
            print(result.stderr)
        
        with open("utills/config.yaml", "r") as f:
            config = yaml.safe_load(f)

    main(args)