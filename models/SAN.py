import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.headers import VQA_header, QR_header, AC_header

class ImageEncoder(nn.Module):
    def __init__(self, image_feature_output=1024):
        super(ImageEncoder, self).__init__()
        self.cnn = models.vgg16(pretrained=True).features
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(512, image_feature_output),
            nn.Tanh())

    def forward(self, image):
        image = self.cnn(image)
        image = image.view(-1, 512, 196).transpose(1, 2)
        image_embedding = self.fc(image)
        return image_embedding

class QuestionEncoder(nn.Module):

    def __init__(self, word_embedding=500, rnn_hidden_size=1024, num_layers=1, batch_first=True):
        super(QuestionEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=word_embedding,
                            hidden_size=rnn_hidden_size, batch_first=batch_first)

    def forward(self, ques):
        _, hx = self.lstm(ques)
        h, _ = hx
        ques_embedding = h[0]
        return ques_embedding
    
class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = torch.tanh(hi+hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = torch.softmax(ha, dim=1)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u

class AnswerEncoder(nn.Module):
    def __init__(self, answer_vocal_size, word_embedding_size, hidden_size):
        super(AnswerEncoder, self).__init__()
        self.answer_embeddings = nn.Embedding(answer_vocal_size, word_embedding_size)
        self.fc = nn.Linear(word_embedding_size, 1024)
        self.tanh = nn.Tanh()

    def forward(self, input_a):
        input_a = self.answer_embeddings(input_a)
        input_a = torch.mean(input_a, dim=1)  # Taking the mean along the sequence dimension
        input_a = self.fc(input_a)
        input_a = self.tanh(input_a)
        return input_a

class SAN(nn.Module):
    # num_attention_layer and num_mlp_layer not implemented yet
    def __init__(self, args,question_vocab_size, ans_vocab_size):          # embed_size, word_embed_size, num_layers, hidden_size
        super(SAN, self).__init__()

        self.img_encoder = ImageEncoder(args.model_config["image_feature_output"])
        self.word_embeddings = nn.Embedding(question_vocab_size, args.model_config["word_embedding"])
        self.ques_encoder = QuestionEncoder(
            word_embedding = args.model_config["word_embedding"], 
            rnn_hidden_size = args.model_config["rnn_hidden_size"],
            num_layers = args.model_config["rnn_layers"], 
            batch_first=True)
        
        self.san = nn.ModuleList([Attention(d=args.model_config["image_feature_output"], k=args.model_config["att_ff_size"])] * args.model_config["num_att_layers"])
        self.mlp = nn.Sequential(
                nn.Linear(args.model_config["image_feature_output"], 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, ans_vocab_size))

    def forward(self, images, questions):

        image_embeddings = self.img_encoder(images)

        embeds = self.word_embeddings(questions)

        ques_embeddings = self.ques_encoder(embeds)

        vi = image_embeddings
        u = ques_embeddings
        for attn_layer in self.san:
            u = attn_layer(vi, u)
            
        combined_feature = self.mlp(u)
        return combined_feature
    