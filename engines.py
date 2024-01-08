

from utils import *
from metrics import calculate_accuracies
import pandas as pd
from torch import nn


def trainer(model, data_loader, optimizer, loss_function, epoch, device, scheduler, args, wandb):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('weighted_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('question_type_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))    
    header = f'Train Epoch: [{epoch}]'
    print_freq = args.print_freq
    
    for i, (images, questions, question_types, answers, large_question_types_str) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    
        images, questions, question_types, answers= images.to(device), questions.to(device), question_types.to(device), answers.to(device)
        qt_output, vqa_outputs = model(images, questions)

        qt_loss, vqa_loss, weighted_loss = loss_function(qt_output, question_types, vqa_outputs, answers)
        
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        scheduler.step()
        if args.wandb:
            wandb.log({"iter_weighted_loss": weighted_loss.item(),
                       "iter_question_type":qt_loss.item(),
                       "iter_vqa":vqa_loss.item()
                       })
        metric_logger.update(weighted_loss=weighted_loss)
        metric_logger.update(question_type_loss=qt_loss)
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
    metric_logger.add_meter('weighted_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('question_type_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    total_outputs = {"question_type": [], "vqa": [], "question_type_label": [], "vqa_label": [], "large_question_type": []}
    for i, (images, questions, question_types, answers, large_question_types_str) in enumerate(metric_logger.log_every(data_loader, print_freq, 'Validation:')):
        # print("val", questions, answers, question_types)
        with torch.no_grad():
            images, questions, question_types, answers= images.to(device), questions.to(device), question_types.to(device), answers.to(device)
            qt_output, vqa_outputs = model(images, questions)

            qt_loss, vqa_loss, weighted_loss = loss_function(qt_output, question_types, vqa_outputs, answers)
            
            metric_logger.update(weighted_loss=weighted_loss.item())
            metric_logger.update(question_type_loss=qt_loss.item())
            metric_logger.update(vqa_loss=vqa_loss.item())

            _, qt_predicted_classes  = torch.max(qt_output, 1)
            qt_predicted_classes = qt_predicted_classes.cpu().tolist()
            vqa_predctions = []
            for i, question_type_id in enumerate(qt_predicted_classes):
                question_type_str = idx_to_answer_type[question_type_id]
                vqa_output_i = vqa_outputs[question_type_str]
                _, vqa_predicted_classes  = torch.max(vqa_output_i, 1)
                vqa_predicted_classes = vqa_predicted_classes.cpu().tolist()
                vqa_prediction = vqa_predicted_classes[i]
                vqa_predctions.append(vqa_prediction) 

            # _, vqa_predicted_classes  = torch.max(vqa_output, 1)
            # vqa_predicted_classes = vqa_predicted_classes.cpu().tolist()
            
            total_outputs["question_type"] += qt_predicted_classes
            total_outputs["vqa"] += vqa_predctions

            total_outputs["question_type_label"] += question_types.cpu().tolist()
            total_outputs["vqa_label"] += answers.cpu().tolist()
            
            total_outputs["large_question_type"] += large_question_types_str

    val_data = pd.DataFrame(total_outputs)
    val_result = calculate_accuracies(val_data)
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger.global_avg()}: {val_result}")

    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result.update(val_result)
    return result
        
    
    
    

