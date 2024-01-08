

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
            wandb.log({"vqa_loss": vqa_loss.item()})
        metric_logger.update(vqa_loss=vqa_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

    

@torch.no_grad()
def validator(model, data_loader, device, loss_function, args):
    
    print()
    model.eval()
    print_freq = args.print_freq
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    question_ids = []
    predictions = []
    targets = []
    for i, (images, questions, answers, answer_str, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, 'Validation:')):
        with torch.no_grad():
            images, questions, answers, question_id= images.to(device), questions.to(device), answers.to(device), question_id.to(device)
            vqa_outputs = model(images, questions)
            vqa_loss = loss_function(vqa_outputs, answers)
            metric_logger.update(vqa_loss=vqa_loss.item())
            _, vqa_predicted  = torch.max(vqa_outputs, 1)
            
            question_ids.append(question_id)
            predictions.append(vqa_predicted)
            targets.append(answers)
            
    question_ids_tensor = torch.cat(question_ids, dim=0)
    predictions_tensor = torch.cat(predictions, dim=0)
    targets_tensor = torch.cat(targets, dim=0)
    
    tensors_to_gather = [question_ids_tensor, predictions_tensor, targets_tensor]
    
    if dist.get_rank() == 0:
        gathered_question_ids = [torch.empty_like(question_ids_tensor) for _ in range(dist.get_world_size())]
        gathered_predictions = [torch.empty_like(predictions_tensor) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.empty_like(targets_tensor) for _ in range(dist.get_world_size())]
    else:
        gathered_question_ids = gathered_predictions = gathered_targets = None

    # Gather the tensors
    for tensor, gathered in zip(tensors_to_gather, [gathered_question_ids, gathered_predictions, gathered_targets]):
        dist.gather(tensor, gather_list=gathered, dst=0)

    # Concatenate the results on the master process
    if dist.get_rank() == 0:
        final_question_ids = torch.cat(gathered_question_ids, dim=0).cpu().tolist()
        final_predictions = torch.cat(gathered_predictions, dim=0).cpu().tolist()
        final_targets = torch.cat(gathered_targets, dim=0).cpu().tolist()
    
        val_data = {
            "id": final_question_ids,
            "prediction": final_predictions,
            "target":final_targets
        }
        val_data = pd.DataFrame(val_data)
        val_accuracies = calculate_accuracies(val_data, data_loader.dataset)
        if args.wandb:
            wandb.log(val_accuracies)
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger.global_avg()}")
    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if dist.get_rank() == 0:
        print(f"Accuracies: {val_accuracies}")
        return result, val_accuracies
    return result, []
        
    
    
    

