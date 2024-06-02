import os
import argparse
import datetime
import json
import time
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch, val_one_epoch,test_one_epoch
from llama import Tokenizer, Tokenizer_llama3
from llama_vqa import LLaMA_VQA
from dataloader import load_data, load_data_instruct
from torch.utils.data import DataLoader, ConcatDataset

def save_arguments(args, filepath):
    with open(filepath, 'w') as file:
        json.dump(vars(args), file)

def load_arguments(filepath):
    with open(filepath, 'r') as file:
        args_dict = json.load(file)
    return args_dict

# Optionally, repopulate argparse.ArgumentParser with these arguments
def repopulate_arguments(args_dict):
    parser = argparse.ArgumentParser(description="Example script")
    for key, value in args_dict.items():
        parser.add_argument(f'--{key}', type=type(value),default=value)
    return parser.parse_args([])

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./pretrained/llama/', type=str, help='path of llama model')
    parser.add_argument('--model', default='llama7B_adapter', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--adapter_layer', type=int, default=32, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--max_feats', type=int, default=10, metavar='LENGTH', help='the maximum feature length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='nextqa', type=str, help='dataset')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--vaq', action='store_true', help='vaq loss')
    parser.add_argument('--qav', action='store_true', help='qav loss')
    parser.add_argument('--bias', type=float, default=3., help='attention bias')
    parser.add_argument('--tau', type=float, default=100., help='tau')
    parser.add_argument('--sub', action='store_true', help='subtitles for VLEP and TVQA')
    parser.add_argument('--eval', action='store_true', help='eval')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--memory', action='store_true', help='meomory')
    parser.add_argument('--finetune', action='store_true', help='finetune')
    parser.add_argument('--data_ratio', type=float, default=1., help='tau')
    parser.add_argument('--textvid', action='store_true', help='virtual video training')
    parser.add_argument('--variance', type=float, default=0., help='variance')
    parser.add_argument('--evalall', action='store_true', help='evalall')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--onlyqa', action='store_true', help='onlyqa')
    parser.add_argument('--llama2', action='store_true', help='llama2')
    parser.add_argument('--llama3', action='store_true', help='llama3')
    parser.add_argument('--answer_balance', action='store_true', help='balance_abcde')
    parser.add_argument('--video_caption', action='store_true', help='video captioning training')
    parser.add_argument('--instruct', action='store_true', help='instruct')
    parser.add_argument('--openvqa', action='store_true', help='openvqa')
    parser.add_argument('--weight_captioning', type=float, default=1.0, help='weight_captioning')
    parser.add_argument('--webvid', action='store_true', help='webvidfituning')
    parser.add_argument('--openvqa_eval', action='store_true', help='logits for MCQA')
    parser.add_argument('--single_frame', action='store_true', help='single_frame')


    

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if args.llama3:
        tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}./tokenizer.model')
    else:
        tokenizer = Tokenizer(model_path=f'{args.llama_model_path}./tokenizer.model')


    model = LLaMA_VQA(args)
    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    best_acc = 0.

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    if args.eval:

        epoch=0
        if args.dataset == 'egos':
            args.batch_size=1
            args.max_seq_len = 600
            
        model.module.re_init_freqs(600)
        if args.test:
            data_loader_val = load_data(args, tokenizer, split='test') 
            if args.distributed:
                data_loader_val.sampler.set_epoch(epoch)
            val_stats = test_one_epoch(model_without_ddp, data_loader_val, optimizer, epoch, args=args)
        else:
            data_loader_val = load_data(args, tokenizer, split='val') 
            if args.distributed:
                data_loader_val.sampler.set_epoch(epoch)
            val_stats = val_one_epoch(model_without_ddp, data_loader_val, optimizer, epoch, args=args)
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    elif args.textvid:
        data_loader_vals = {}
        batch_size = args.batch_size
        max_seq_len = args.max_seq_len
        data_loader_train = load_data(args, tokenizer, split='train')

        eval_datasets = ['egos']

        for dataset_name in eval_datasets:
        # for dataset_name in ['egos','tvqa']:
            args.dataset=dataset_name
            if dataset_name in ['egos']:
                args.batch_size=1
                args.max_seq_len = 600
            else:
                args.batch_size= batch_size
                args.max_seq_len = 200            
            data_loader_vals[dataset_name] = load_data(args, tokenizer, split='val')

        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            model.module.re_init_freqs(600)
            train_stats = train_one_epoch(model, data_loader_train, optimizer, epoch, loss_scaler, args=args)
            val_stats = {}
            for key,data_loader_val in data_loader_vals.items():
                if args.distributed:
                    data_loader_val.sampler.set_epoch(epoch) 
                val_stats[key] = val_one_epoch(model_without_ddp, data_loader_val, optimizer, epoch, args=args)
              
            if True:
                model_name = f"checkpoint_{epoch}"
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name=model_name)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                for key, val_stat in val_stats.items():
                    log_stat = {'dataset:':key, **{f'val_{k}': v for k, v in val_stat.items()}}
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stat) + "\n")          
                    
    else:
        model.module.re_init_freqs(600)
        data_loader_train = load_data(args, tokenizer, split='train')
        data_loader_val = load_data(args, tokenizer, split='val')       
        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
                data_loader_val.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(model, data_loader_train, optimizer, epoch, loss_scaler, args=args)
            val_stats = val_one_epoch(model_without_ddp, data_loader_val, optimizer, epoch, args=args)

            if args.output_dir and best_acc < val_stats['acc']:
                best_acc = val_stats['acc']
                model_name = 'checkpoint_best'
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name=model_name)
            if True:
                model_name = f"checkpoint_{epoch}"
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name=model_name)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, **{f'val_{k}': v for k, v in val_stats.items()}}

            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_arguments(args, args.output_dir+'/args.json')

    main(args)
