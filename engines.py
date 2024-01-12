

from utils import *
from metrics import calculate_accuracies
import pandas as pd
from torch import nn
import torch.distributed as dist
import wandb


class VQA_Trainer:
    def __init__(self, model, loss_function, optimizer, scheduler, device, args):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args

        self.best_qt_loss = float('inf')
        self.stop_early_epoch = 0
        
    def trainer(self, data_loader, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('qt_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('total_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Train Epoch: [{epoch}]'
        print_freq = self.args.print_freq
        
        for i, (images, questions, answers, answer_type, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
            images, questions, answers, answer_type= images.to(self.device), questions.to(self.device), answers.to(self.device), answer_type.to(self.device)
            qt_output, vqa_outputs = self.model(images, questions)

            qt_loss, vqa_loss, total_loss, vqa_losses = self.loss_function(qt_output, answer_type, vqa_outputs, answers)
            
            self.optimizer.zero_grad()
            # total_loss.backward()
            qt_loss.backward(retain_graph=True)
            for vqa_loss_i in vqa_losses.values():
                vqa_loss_i.backward(retain_graph=True)
            self.optimizer.step()
            self.scheduler.step()
            if self.args.wandb:
                wandb.log({"train_vqa_loss_iter": vqa_loss.item()})
            metric_logger.update(vqa_loss=vqa_loss)
            metric_logger.update(qt_loss=qt_loss)
            metric_logger.update(total_loss=total_loss)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def validator(self, data_loader, epoch):
        print()
        self.model.eval()
        print_freq = self.args.print_freq
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('vqa_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('qt_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('total_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
        total_question_ids = []
        total_qt_predictions = []
        total_qt_targets = []
        
        total_vqa_predictions = []
        total_vqa_targets = []
        
        for i, (images, questions, answers, answer_type, question_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, 'Validation:')):
            with torch.no_grad():
                images, questions, answers, answer_type, question_ids = images.to(self.device), questions.to(self.device), answers.to(self.device), answer_type.to(self.device), question_ids.to(self.device)
                qt_output, vqa_outputs = self.model(images, questions)
                qt_loss, vqa_loss, total_loss, vqa_losses = self.loss_function(qt_output, answer_type, vqa_outputs, answers)
                metric_logger.update(vqa_loss=vqa_loss.item())
                metric_logger.update(qt_loss=qt_loss)
                metric_logger.update(total_loss=total_loss)
                
                
                total_question_ids.append(question_ids)
                total_qt_targets.append(answer_type)
                total_vqa_targets.append(answers)
                
                _, qt_predictions  = torch.max(qt_output, 1)
                total_qt_predictions.append(qt_predictions)
                vqa_predictions = []
                for i, qt_idx in enumerate(qt_predictions.cpu().tolist()):
                    qt_str = data_loader.dataset.idx_to_ans_type[qt_idx]
                    vqa_output_at = vqa_outputs[qt_str]
                    _, vqa_output_at  = torch.max(vqa_output_at, 1)
                    pred_i = vqa_output_at[i].item()
                    vqa_predictions.append(pred_i)
                
                vqa_predictions = torch.tensor(vqa_predictions)
                total_vqa_predictions.append(vqa_predictions)
                
        # total_vqa_predictions = torch.tensor(total_vqa_predictions)
        # print(total_vqa_predictions)
        question_ids_tensor = torch.cat(total_question_ids, dim=0)
        qt_predictions_tensor = torch.cat(total_qt_predictions, dim=0)
        vqa_predictions_tensor = torch.cat(total_vqa_predictions, dim=0).to(device)
        qt_targets_tensor = torch.cat(total_qt_targets, dim=0)
        vqa_targets_tensor = torch.cat(total_vqa_targets, dim=0)
        
        tensors_to_gather = [question_ids_tensor, qt_predictions_tensor, vqa_predictions_tensor, qt_targets_tensor, vqa_targets_tensor]
        if dist.get_rank() == 0:
            gathered_question_ids = [torch.empty_like(question_ids_tensor) for _ in range(dist.get_world_size())]
            gathered_qt_predictions = [torch.empty_like(qt_predictions_tensor) for _ in range(dist.get_world_size())]
            gathered_vqa_predictions = [torch.empty_like(vqa_predictions_tensor) for _ in range(dist.get_world_size())]
            gathered_qt_target = [torch.empty_like(qt_targets_tensor) for _ in range(dist.get_world_size())]
            gathered_vqa_target = [torch.empty_like(vqa_targets_tensor) for _ in range(dist.get_world_size())]
        else:
            gathered_question_ids = gathered_qt_predictions = gathered_vqa_predictions = gathered_qt_target = gathered_vqa_target= None
    
        # Gather the tensors
        for tensor, gathered in zip(tensors_to_gather, [gathered_question_ids, gathered_qt_predictions, gathered_vqa_predictions, gathered_qt_target, gathered_vqa_target]):
            dist.gather(tensor, gather_list=gathered, dst=0)
    
        # Concatenate the results on the master process
        if dist.get_rank() == 0:
            final_question_ids = torch.cat(gathered_question_ids, dim=0).cpu().tolist()
            final_qt_predictions = torch.cat(gathered_qt_predictions, dim=0).cpu().tolist()
            final_vqa_predictions = torch.cat(gathered_vqa_predictions, dim=0).cpu().tolist()
            final_qt_targets = torch.cat(gathered_qt_target, dim=0).cpu().tolist()
            final_vqa_targets = torch.cat(gathered_vqa_target, dim=0).cpu().tolist()
        
            val_prediction_csv = {
                "id": final_question_ids,
                "prediction": final_vqa_predictions,
                "target":final_vqa_targets,
                "answer_type": final_qt_targets,
                "answer_type_prediction": final_qt_predictions,
                
            }
            val_prediction_csv = pd.DataFrame(val_prediction_csv)
            
            val_accuracies, val_prediction_csv = calculate_accuracies(val_prediction_csv, data_loader.dataset)
            val_accuracies["epoch"] = epoch
            if args.wandb:
                wandb.log(val_accuracies)
        metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {metric_logger.global_avg()}")
        result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if dist.get_rank() == 0:
            print(f"Accuracies: {val_accuracies}")
            return result, val_accuracies, val_prediction_csv
        return result, [], []
    






