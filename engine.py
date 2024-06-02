import torch
import math
import sys
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
import json
import random
def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 50)
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        mode='vqa'
        if args.video_caption:
            if random.random()>0.5:
                mode='caption'
        if args.webvid:
            mode = 'caption'
        vqa_loss, vaq_loss, qav_loss = model(data,mode=mode)
        # print(vqa_loss, vaq_loss, qav_loss)
        loss = vqa_loss + vaq_loss*args.weight_captioning + qav_loss
        loss_value = loss.item()
        vqa_loss_value = vqa_loss.item()
        vaq_loss_value = vaq_loss.item()
        qav_loss_value = qav_loss.item()

   
        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(vaq_loss=vaq_loss_value)
        metric_logger.update(qav_loss=qav_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 10)
    results = {}
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        answer = data['answer'].cuda()
        bsz = answer.shape[0]
        if args.openvqa_eval:
            mode = 'caption'
        else:
            mode = 'vqa'
        with torch.no_grad():
            logits = model(data, inference=True,mode=mode)
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)

        eval = (answer == prediction)
        acc = eval.sum().item() / bsz
 
        misc.log_qtype(data, eval, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) /10)
    results = {}
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        answer = data['answer'].cuda()
        bsz = answer.shape[0]
        if args.openvqa_eval:
            mode = 'caption'
        else:
            mode = 'vqa'
        with torch.no_grad():
            logits = model(data, inference=True,mode=mode)

        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)
        
        results[data['qid'][0]]=prediction.item()

        eval = (answer == prediction)
        acc = eval.sum().item() / bsz
        
        misc.log_qtype(data, eval, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)

    # gather the stats from all processes
    with open(f'{args.output_dir}/egos.json','w') as f:
        json.dump(results,f)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
