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
    module_path = f"utills.task{args.task}.{f'category{args.category}.' if args.category else ''}ClassificationModel"
    module = import_module(module_path)
    return getattr(module, 'ClassificationModel')(args)


def freeze_params(model, freeze):
    for param in model.parameters():
        param.requires_grad = not freeze
    if freeze:
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True


def main(args):
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    model = get_model(args)
    freeze_params(model, args.freeze)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        save_last=True,
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        dirpath=f'checkpoint/Classification/task{args.task}/{f"category{args.category}" if args.category else ""}'
    )
    log_dir = f"tb_logs/Classification/task{args.task}/{f'category{args.category}' if args.category else ''}"
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(log_dir, name=f"{args.task}-{args.category}-{args.model_name}-bs{args.batch_size}-epochs{args.epochs}-{'Freeze' if args.freeze else 'NonFreeze'}")

    trainer_args = {"max_epochs": args.epochs, 
                    "log_every_n_steps": args.log_every_n_steps, 
                    "logger": logger, 
                    "callbacks": [checkpoint_callback], 
                    "devices": args.devices, 
                    "strategy": "ddp"}

    if args.resume_from_checkpoint:
        trainer_args["resume_from_checkpoint"] = args.resume_from_checkpoint

    trainer = pl.Trainer(**trainer_args)

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
    parser.add_argument("--train_link", type=str, default=config['classfication_task1_category2_train'], help="train_link for the trainer")
    parser.add_argument("--valid_link", type=str, default=config['classfication_task1_category2_valid'], help="valid_link for the trainer")
    parser.add_argument("--task", type=int, default=None, help="Witch task?")
    parser.add_argument("--category", type=int, default=None, help="Witch category?")

    args = parser.parse_args()

    # GPU 전부 사용 안할거면 남들 사용하게 cpu우선순위 낮춤
    if args.devices != -1:
        pid = os.getpid()
        result = subprocess.run(['renice', '15', '-p', f'{pid}'], capture_output=True, text=True)
        print(f"스케쥴러 우선순위 {'내림' if result.returncode == 0 else '내림 실패'}:\n{result.stdout or result.stderr}")
        
    warnings.filterwarnings('ignore', 'Invalid SOS parameters for sequential JPEG')
    main(args)