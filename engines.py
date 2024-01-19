

from utils import *
from metrics import calculate_accuracies
import pandas as pd
from torch import nn
import torch.distributed as dist


def trainer(model, data_loader, optimizer, loss_function, epoch, device, scheduler, args, wandb):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))    
    header = f'Train Epoch: [{epoch}]'
    print_freq = args.print_freq
    
    for i, (images, questions, answers, answer_str, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    
        images, questions, answers= images.to(device), questions.to(device), answers.to(device)
        vqa_outputs = model(images, questions)

        vqa_loss = loss_function(vqa_outputs, answers)
        
        optimizer.zero_grad()
        vqa_loss.backward()
        optimizer.step()
        scheduler.step()
        if args.wandb:
            wandb.log({"train_vqa_loss_iter": vqa_loss.item()})
        metric_logger.update(vqa_loss=vqa_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

    

@torch.no_grad()
def validator(model, data_loader, device, loss_function, args, epoch):
    print()
    model.eval()
    print_freq = args.print_freq
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    results = []
    for i, (images, questions, answers, answer_str, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, 'Validation:')):
        with torch.no_grad():
            images, questions, answers, question_id= images.to(device), questions.to(device), answers.to(device), question_id.to(device)
            vqa_outputs = model(images, questions)
            vqa_loss = loss_function(vqa_outputs, answers)
            metric_logger.update(vqa_loss=vqa_loss.item())
            _, vqa_predicted  = torch.max(vqa_outputs, 1)
            
            question_id = question_id.cpu().tolist()
            predictions = vqa_predicted.cpu().tolist()
            answers = answers.cpu().tolist()
            for ques_id, pred, target in zip(question_id, predictions, answers):
                ques_id = int(ques_id)          
                results.append({"question_id":ques_id, 
                                "prediction":data_loader.dataset.ix_to_ans[str(pred)],
                                "target":data_loader.dataset.ix_to_ans[str(target)]})
                
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger.global_avg()}")
    stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stat, results


@torch.no_grad()
def evaluate(model, data_loader, device):
    print()
    metric_logger = MetricLogger(delimiter="  ")
    model.eval()
    results = []
    for i, (images, questions, answers, answer_str, question_id) in enumerate(metric_logger.log_every(data_loader, 20, 'Evaluating:')):
        with torch.no_grad():
            images, questions, answers, question_id= images.to(device), questions.to(device), answers.to(device), question_id.to(device)
            vqa_outputs = model(images, questions)
            _, vqa_predicted  = torch.max(vqa_outputs, 1)
            
            question_id = question_id.cpu().tolist()
            predictions = vqa_predicted.cpu().tolist()
            answers = answers.cpu().tolist()
            for ques_id, pred, target in zip(question_id, predictions, answers):
                ques_id = int(ques_id)          
                results.append({"question_id":ques_id, 
                                "prediction":data_loader.dataset.ix_to_ans[str(pred)],
                                "target":data_loader.dataset.ix_to_ans[str(target)]})
                
    return results
            
        
    
    
    

