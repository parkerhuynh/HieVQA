import torch
import torch.nn as nn
import torch.nn.functional as F


def kronecker_delta(A, B):
    return A.eq(B).int()

class HierarchicalLoss(nn.Module):
    def __init__(self, args, train_dataset):
        super(HierarchicalLoss, self).__init__()
        self.args = args
        self.answer_type_to_idx = train_dataset.answer_type_to_idx
        self.idx_to_answer_type = train_dataset.idx_to_answer_type
        
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
        total_loss = qt_loss + total_vqa_loss
        return qt_loss, total_vqa_loss, total_loss
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class HierarchicalLoss(nn.Module):
#     def __init__(self, num_categories):
#         super(HierarchicalLoss, self).__init__()
#         self.num_categories = num_categories
#         self.question_loss = nn.CrossEntropyLoss(reduction='none')  # Don't reduce in order to handle the second level loss individually

#     def forward(self, pq_logits, tq_indices, pa_logits, ta_indices):
#         # Level 1 Loss: Question Classification Loss
#         # pq_logits are the raw, unnormalized scores for each question category (output from a model, before softmax)
#         # tq_indices are the indices of the true question categories
#         LQ = self.question_loss(pq_logits, tq_indices)

#         # Level 2 Loss: Answer Prediction Loss
#         LAA = torch.zeros_like(LQ)
#         rho = torch.argmax(pq_logits, dim=1)  # Predicted question category

#         for i in range(len(tq_indices)):
#             # Get the range of indices for the answer logits based on the predicted category
#             start_index = sum(self.num_categories[:rho[i]])
#             end_index = start_index + self.num_categories[rho[i]]

#             # Compute the answer prediction loss for the correct subset
#             LAA[i] = F.cross_entropy(pa_logits[i, start_index:end_index].unsqueeze(0), 
#                                       ta_indices[i].unsqueeze(0), 
#                                       reduction='none')

#         # Combine Level 1 and Level 2 Loss
#         LCQVQA = LQ + LAA.squeeze()

#         return LCQVQA.mean()