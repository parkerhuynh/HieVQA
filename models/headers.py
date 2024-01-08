import torch
from torch import nn

class VQA_header(nn.Module):
    def __init__(self, args, ans_vocab_size):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(args.model_config["image_feature_output"], 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, ans_vocab_size))

    def forward(self, hidden_states):
        output = self.mlp(hidden_states)
        output = torch.sigmoid(output)
        return output
    
class QR_header(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(args.model_config["image_feature_output"], 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, 1))

    def forward(self, hidden_states):
        output = self.mlp(hidden_states)
        output = torch.sigmoid(output)
        return output

    
class AC_header(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(args.model_config["image_feature_output"], 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, 3))

    def forward(self, hidden_states):
        output = self.mlp(hidden_states)
        output = torch.sigmoid(output)
        return output
