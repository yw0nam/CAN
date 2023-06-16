import os
import pandas as pd
from models.pl_model_hf import PL_model
import pytorch_lightning as pl
from dataset_hf import *
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2Processor
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf as OC
from utils import str2bool
from pytorch_lightning.strategies import DDPStrategy

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("-t", '--train_config', default='./configs/IEMOCAP/train.yaml', type=str)
    p.add_argument('--exp_name', type=str, required=True)
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--using_model', required=True, type=str)
    p.add_argument('--using_contra', required=True, type=str2bool, nargs='?', const=True, default=False)
    p.add_argument('--using_cma', required=True, type=str2bool, nargs='?', const=True, default=False,)
    p.add_argument('--using_weight_decay', required=True, type=str2bool, nargs='?', const=True, default=False,)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--accumulate_grad', type=int, default=1)
    # p.add_argument('--lower_clip_length', type=float, default="0.0")
    p.add_argument('--upper_clip_length', type=float, default="15.0")
    config = p.parse_args()

    return config


def main(args):
    pl.seed_everything(1004)
    num_gpu = torch.cuda.device_count()
    train_config = OC.load(args.train_config)

    train_config['path']['log_path'] = os.path.join(args.save_path, "log")
    train_config['path']['ckpt_path'] = os.path.join(args.save_path, "ckpt")
    train_config['path']['exp_name'] = args.exp_name
    train_config['optimizer']['batch_size'] = args.batch_size
    train_config['trainer']['grad_acc'] = args.accumulate_grad
    train_config['exp_setting']['using_cma'] = args.using_cma
    train_config['exp_setting']['using_model'] = args.using_model
    train_config['exp_setting']['using_contra'] = args.using_contra
    train_config['exp_setting']['using_weight_decay'] = args.using_weight_decay
    # Load train and validation data
    train = pd.read_csv(train_config['path']['train_csv']).query(f"wav_length <= {args.upper_clip_length}")
    dev = pd.read_csv(train_config['path']['dev_csv']).query(f"wav_length <= {args.upper_clip_length}")
    
    text_tokenizer = AutoTokenizer.from_pretrained(train_config['model']['text_encoder'])
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config['model']['audio_encoder'])
    
    train_dataset = multimodal_dataset(train, train_config)
    val_dataset = multimodal_dataset(dev, train_config)

    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(val_dataset),
    )

    total_batch_size = train_config['optimizer']['batch_size'] * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / (total_batch_size * train_config['trainer']['grad_acc']) * train_config['step']['max_epochs'])
    n_warmup_steps = int(n_total_iterations * train_config['step']['warmup_ratio'])
    
    train_config['step']['total_step'] = n_total_iterations
    train_config['step']['warm_up_step'] = n_warmup_steps
    
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    train_loader = DataLoader(
        train_dataset, train_config['optimizer']['batch_size'], num_workers=6,
        collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
        shuffle=False, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, train_config['optimizer']['batch_size'], num_workers=6,
        collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True, 
        shuffle=False, drop_last=True
    )
        
        
    # Load model and configuration.
    model = PL_model(train_config)
    
    setattr(model, 'train_dataloader', lambda: train_loader)
    setattr(model, 'val_dataloader', lambda: val_loader)
    

    checkpoint_callback = ModelCheckpoint(
        monitor="val/emotion_loss",
        dirpath=os.path.join(train_config['path']['ckpt_path'], train_config['path']['exp_name']),
        filename="step={step:06d}-val_emotion_loss={val/emotion_loss:.5f}",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
        every_n_train_steps=train_config['step']['total_step'] // 10 
    )
    logger = TensorBoardLogger(
        train_config['path']['log_path'], name=train_config['path']['exp_name'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpu,
        strategy="deepspeed_stage_2",
        max_steps=train_config['step']['total_step'],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, lr_monitor],
        profiler="simple",
        accumulate_grad_batches=train_config['trainer']['grad_acc'],
        logger=logger,
        gradient_clip_val=train_config['trainer']['grad_clip_thresh'],
        precision=16,
    )
    
    trainer.fit(model)
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)
