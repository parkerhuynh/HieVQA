

from utils import *
from metrics import calculate_accuracies
import pandas as pd
from torch import nn
import torch.distributed as dist


def trainer(model, data_loader, optimizer, loss_function, epoch, device, scheduler, args, wandb):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('qt_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Train Epoch: [{epoch}]'
    print_freq = args.print_freq
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # "image": image,
        # "question_rnn": question,
        # "answer": ann["answer_idx"],
        # "answer_type": ann['answer_type_idx'],
        # "question_id": ann['id'],
        # "question_text": ques["question_str"],
        # "question_bert": encoding['input_ids'].flatten(),
        # "question_bert_att_mask": encoding['attention_mask'].flatten()
        
        images = batch["image"].to(device)
        questions_rnn = batch["question_rnn"].to(device)
        question_type = batch["question_type"].to(device)
        answers = batch["answer"].to(device)
        
        question_bert = batch["question_bert"].to(device)
        question_bert_att_mask = batch["question_bert_att_mask"].to(device)
        
        qt_output, vqa_outputs = model(images, questions_rnn, question_bert, question_bert_att_mask)        
        qt_loss, vqa_loss, total_loss = loss_function(qt_output, question_type, vqa_outputs, answers)
        
        # print(qt_loss, total_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        if args.wandb:
            wandb.log({"train_vqa_loss_iter": vqa_loss.item()})
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
            vqa_answer_str = batch["question_type_str"]
            
            
            qt_output, vqa_outputs = model(images, questions_rnn, question_bert, question_bert_att_mask)
            qt_loss, vqa_loss, total_loss = loss_function(qt_output, question_type, vqa_outputs, answers)
            
            
            metric_logger.update(vqa_loss=vqa_loss.item())
            metric_logger.update(qt_loss=qt_loss)
            metric_logger.update(total_loss=total_loss)
            _, qt_predictions  = torch.max(qt_output, 1)
            qt_predictions = qt_predictions.cpu().tolist()
            vqa_answers = answers.cpu().tolist()
            
            for idx, (ques_id, qt_pred, qt_target, ans_str) in enumerate(zip(question_id, qt_predictions, question_type_str, vqa_answer_str)):
                ques_id = int(ques_id)
                pre_ques_type_str = data_loader.dataset.idx_to_ans_type[qt_pred]
                target_ques_type_str = qt_target
                
                
                
                vqa_result_qt = vqa_outputs[pre_ques_type_str]
                _, vqa_result_qt  = torch.max(vqa_result_qt, 1)
                pred_vqa = vqa_result_qt[idx].item()
                
                pred_idx_to_answer = data_loader.dataset.ix_to_ans[pre_ques_type_str]
                pred_vqa_str = pred_idx_to_answer[pred_vqa]
                
                
                results.append({"question_id":ques_id, 
                                "qt_prediction": pre_ques_type_str,
                                "qt_target":target_ques_type_str,
                                "vqa_target":ans_str,
                                "vqa_prediction": pred_vqa_str})
                
            print(results)
            a
                
            

            
            # for ques_id, pred, target in zip(question_id, predictions, answers):
            # _, qt_predictions  = torch.max(qt_output, 1)
            # vqa_predictions = []
            # for i, qt_idx in enumerate(qt_predictions.cpu().tolist()):
            #     qt_str = data_loader.dataset.idx_to_ans_type[qt_idx]
            #     vqa_output_at = vqa_outputs[qt_str]
            #     _, vqa_output_at  = torch.max(vqa_output_at, 1)
            #     pred_i = vqa_output_at[i].item()
            #     vqa_predictions.append(pred_i)
            
            # vqa_predictions = torch.tensor(vqa_predictions)
            # total_vqa_predictions.append(vqa_predictions)
            
    return result, [], []
        
    
    

    
    

