

from utils import *
from metrics import calculate_accuracies
import pandas as pd
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

def trainer(model, data_loader, optimizer, loss_function, epoch, device, args, wandb):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('qt_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Train Epoch: [{epoch}]'
    print_freq = args.print_freq
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        images = batch["image"].to(device)
        questions_rnn = batch["question_rnn"].to(device)
        question_type = batch["question_type"].to(device)
        answers = batch["answer"].to(device)        
        question_bert = batch["question_bert"].to(device)
        question_bert_att_mask = batch["question_bert_att_mask"].to(device)
        qt_output, vqa_outputs = model(images, questions_rnn, question_bert, question_bert_att_mask)
        qt_loss, vqa_losses = loss_function(qt_output, question_type, vqa_outputs, answers)
        optimizer.zero_grad()
        qt_loss.backward(retain_graph=True)
        for vqa_based_qt in vqa_losses:
            vqa_losses[vqa_based_qt].backward(retain_graph=True)
        optimizer.step()
        if args.wandb:
            wandb.log({"train_vqa_loss_iter": qt_loss.item()})
        vqa_loss = sum(vqa_losses.values())
        total_loss = qt_loss+vqa_loss
        metric_logger.update(vqa_loss=vqa_loss)
        metric_logger.update(qt_loss=qt_loss)
        metric_logger.update(total_loss=total_loss)
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
    metric_logger.add_meter('qt_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    
    
    results = []
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, 'Validation:')):
        with torch.no_grad():
            images = batch["image"].to(device)
            questions_rnn = batch["question_rnn"].to(device)
            question_type = batch["question_type"].to(device)
            answers = batch["answer"].to(device)
            
            question_bert = batch["question_bert"].to(device)
            question_bert_att_mask = batch["question_bert_att_mask"].to(device)
            
            question_id = batch["question_id"]
            question_type_str = batch["question_type_str"]
            vqa_answer_str = batch["answer_str"]
            
            
            qt_output, vqa_outputs = model(images, questions_rnn, question_bert, question_bert_att_mask)
            qt_loss, vqa_losses = loss_function(qt_output, question_type, vqa_outputs, answers)

            vqa_loss = sum(vqa_losses.values())
            total_loss = qt_loss+vqa_loss
            metric_logger.update(vqa_loss=vqa_loss.item())
            metric_logger.update(qt_loss=qt_loss)
            metric_logger.update(total_loss=total_loss)
            
            _, qt_predictions  = torch.max(qt_output, 1)
            qt_predictions = qt_predictions.cpu().tolist()
            
            for idx, (ques_id, qt_pred, qt_target, ans_str) in enumerate(zip(question_id, qt_predictions, question_type_str, vqa_answer_str)):
                ques_id = int(ques_id)
                pre_ques_type_str = data_loader.dataset.idx_to_ans_type[qt_pred]
                target_ques_type_str = qt_target
                
                
                
                vqa_result_qt = vqa_outputs[pre_ques_type_str]
                _, vqa_result_qt  = torch.max(vqa_result_qt, 1)
                pred_vqa = vqa_result_qt[idx].item()
                
                pred_idx_to_answer = data_loader.dataset.ix_to_ans[pre_ques_type_str]
                pred_vqa_str = pred_idx_to_answer[str(pred_vqa)]
                
                
                results.append({"question_id":ques_id, 
                                "small_answer_type_prediction": pre_ques_type_str,
                                "small_answer_type_target":target_ques_type_str,
                                "target class":ans_str,
                                "prediction class": pred_vqa_str})
                
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger.global_avg()}")
    stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stat, results

                
            

            
        
    
    

    
    

