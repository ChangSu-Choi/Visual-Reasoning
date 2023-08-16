import pytorch_lightning as pl
from importlib import import_module
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
import argparse
import yaml
import os
import subprocess
import warnings

def get_model(args):
    # 모듈 경로를 생성합니다.
    if args.category:
        module_path = f"utills.task{args.task}.category{args.category}.ClassificationModel"
    else:
        module_path = f"utills.task{args.task}.ClassificationModel"
    # 모듈을 동적으로 로드합니다.
    module = import_module(module_path)
    # 모듈에서 ClassificationModel 클래스를 가져옵니다.
    ClassificationModel = getattr(module, 'ClassificationModel')
    # ClassificationModel를 인스턴스화하고 반환합니다.
    return ClassificationModel(args)


def main(args):
    # 정확도 < 속도
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    model = get_model(args)

    if args.freeze is True:
        # 모델 전체 프리즈 시키고
        for param in model.parameters():
            param.requires_grad = False
        # model.named_parameters() 로 name을 확인하고 아래와 같이 required_grad = True 로 변경
        for name, param in model.named_parameters():
            if name.count("fc"):
                param.requires_grad = True
    else:
        pass


    # ModelCheckpoint 생성 및 설정
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 추적할 지표 설정
        mode='min',  # 지표를 최소화하도록 설정
        save_top_k=2,  # 베스트 모델 k개를 저장, 1로 설정하면 베스트 모델 1개만 저장
        save_last=True,  # 마지막 모델도 저장
        filename='best-model-{epoch:02d}-{val_loss:.2f}',  # 저장될 파일명 형식
        dirpath=f'checkpoint/Classification/task{args.task}/category{args.category}'\
            if args.category else f'checkpoint/Classification/task{args.task}',  # 체크포인트 저장 경로
    )
    
    # 텐서 보드 
    log_dir = f"tb_logs/Classification/task{args.task}/{f'category{args.category}' if args.category else ''}"
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(log_dir, name=f"{args.task}-{args.category}-{args.model_name}-bs{args.batch_size}-epochs{args.epochs}-{'Freeze' if args.freeze else 'NonFreeze'}")

    # 체크포인트부터 학슬 할 건지 아닌지
    if args.resume_from_checkpoint is None:
        trainer = pl.Trainer(accelerator="gpu", max_epochs=args.epochs, log_every_n_steps=args.log_every_n_steps, logger=logger, callbacks=[checkpoint_callback], devices=args.devices)
#        trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=args.log_every_n_steps, logger=logger, callbacks=[checkpoint_callback], devices=args.devices, strategy="ddp", gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=args.log_every_n_steps, logger=logger, callbacks=[checkpoint_callback], devices=args.devices, strategy="ddp", resume_from_checkpoint=checkpoint_path, gpus=1)
    
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    trainer.fit(model)

if __name__ == "__main__":
    with open("utills/config.yaml", "r") as f:
        config = yaml.safe_load(f)


    parser = argparse.ArgumentParser(description="Classification python code")
    parser.add_argument("--input_dim", type=int, default=config['input_dim'], help="Path to the input file.")
    parser.add_argument("--mlp_hidden", type=int, default=config['mlp_hidden'], help="Path to the output file.")
    parser.add_argument("--batch_size", type=int, default=config['batch_size'], help="Number of training epochs.")
    parser.add_argument("--model_name", type=str, default=None, help="resnet50 or vit")
    parser.add_argument("--mode", type=str, default=config['mode'], help="resnet50 or vit")
    parser.add_argument("--num_workers", type=str, default=config['num_workers'], help="num_workers for the trainer")
    parser.add_argument("--devices", default=config['devices'], help="gpus for the trainer")
    parser.add_argument("--epochs", type=int, default=config['epochs'], help="epochs for the trainer")
    parser.add_argument("--clr_temperature", type=int, default=config['clr_temperature'], help="epochs for the trainer")
    parser.add_argument("--learning_rate", type=int, default=config['learning_rate'], help="learning-rate for the trainer")
    parser.add_argument("--log_every_n_steps", type=int, default=config['log_every_n_steps'], help="log_every_n_steps for the trainer")
    parser.add_argument("--model_save_path", type=str, default=config['model_save_path'], help="Path to save the trained model weights.")

    parser.add_argument("--freeze", action="store_false", default=False, help="About using freezing model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume training from.")
    parser.add_argument("--train_link", type=str0, help="train_link for the trainer")
    parser.add_argument("--valid_link", type=str, default=config['classfication_task1_category2_valid'], help="valid_link for the trainer")
    parser.add_argument("--task", type=int, default=None, help="Witch task?")
    parser.add_argument("--category", type=int, default=None, help="Witch category?")

    args = parser.parse_args()

    # GPU 전부 사용 안할거면 남들 사용하게 cpu우선순위 낮춤
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
    
    warnings.filterwarnings('ignore', 'Invalid SOS parameters for sequential JPEG')


    main(args)