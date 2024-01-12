import torch
import torch.nn as nn
import torch.nn.functional as F


def kronecker_delta(A, B):
    return A.eq(B).int()

class HierarchicalLoss(nn.Module):
    def __init__(self, args, train_dataset):
        super(HierarchicalLoss, self).__init__()
        self.args = args
        self.answer_type_to_idx = train_dataset.ans_type_to_idx
        
    def forward(self, pq, tq, pas, ta):
        qt_loss = F.cross_entropy(pq, tq)
        vqa_losses = {key: 0 for key in pas.keys()}
        _, pq_class  = torch.max(pq, 1)
        
        for question_type_str in pas.keys():
            mask_ta = ta.clone()
            question_type_idx = self.answer_type_to_idx[question_type_str]
            pq_filter_values = (pq_class == question_type_idx).int()
            tq_filter_values = (tq == question_type_idx).int()
            kronecker_delta_values = pq_filter_values & tq_filter_values
            
            output_i = pas[question_type_str]
            number_category = output_i.size()[1] - 1
            mask = mask_ta > number_category
            mask_ta[mask] = 0
            vqa_loss_i = F.cross_entropy(output_i, mask_ta, reduction = "none")
            vqa_loss_i = vqa_loss_i*kronecker_delta_values
            vqa_loss_i = vqa_loss_i.mean()
            vqa_losses[question_type_str] = vqa_loss_i
        total_vqa_loss = sum(vqa_losses.values())
        return qt_loss, total_vqa_loss vqa_losses