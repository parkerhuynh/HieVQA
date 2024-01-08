

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
def validator(model, data_loader, device, loss_function, args, idx_to_answer_type):
    print()
    model.eval()
    print_freq = args.print_freq
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    question_ids = []
    predictions = []
    targets = []
    for i, (images, questions, answers, answer_str, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, 'Validation:')):
        # print("val", questions, answers, question_types)
        with torch.no_grad():
            images, questions, answers= images.to(device), questions.to(device), answers.to(device)
            vqa_outputs = model(images, questions)

            vqa_loss = loss_function(vqa_outputs, answers)
            
            metric_logger.update(vqa_loss=vqa_loss.item())

            _, vqa_predicted  = torch.max(vqa_outputs, 1)


            question_ids += question_id
            predictions += vqa_predicted
            targets += answer_str
    print(question_ids)
    print(predictions)
    print(targets)
    gather_predictions = {"question_id": [], "prediction": [], "target": []}
    local_size = vqa_predicted.size()
    local_size_tuple = tuple(local_size)

    if is_main_process():
        for key in gather_predictions.keys():
            # Correctly form the size tuple for the gathered_predictions tensor
            # Assuming you want a tensor of shape (world_size, *local_size_tuple)
            tensor_size = (dist.get_world_size(), *local_size_tuple)
            gathered_predictions = torch.empty(tensor_size, device=vqa_predicted.device, dtype=vqa_predicted.dtype)
            # You might want to gather predictions here or do other operations
            # For now, just appending the empty tensor
            gather_predictions[key].append(gathered_predictions)

    print(gather_predictions)
    val_data = pd.DataFrame(total_outputs)
    
        
    print(val_data)
    val_result = calculate_accuracies(val_data)
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger.global_avg()}: {val_result}")

    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result.update(val_result)
    return result
        
    
    
    

