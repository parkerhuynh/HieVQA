import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from models.headers import VQA_header, QR_header, AC_header

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        # Pdb().set_trace()
        x = x / x.norm(p=self.p, dim=1, keepdim=True)
        return x
    
class ImageEncoder(nn.Module):

    def __init__(self, output_size=1024):
        super(ImageEncoder, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        self.fflayer = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.Tanh())

    def forward(self, image):
        image = self.extractor(image)
        image_embedding = self.fflayer(image)
        return image_embedding

class QuestionEmbedding(nn.Module):
    def __init__(self, word_embedding_size, hidden_size):
        super(QuestionEmbedding, self).__init__()
        self.gru = nn.GRU(word_embedding_size, hidden_size, num_layers= 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1024)
        self.tanh = nn.Tanh()

    def forward(self, input_data):
        output, hidden = self.gru(input_data)
        last_hidden = hidden.squeeze(0)
        embedding = self.fc(last_hidden)
        embedding = self.tanh(embedding)

        return embedding
    
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
    
class VQA(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        super(VQA, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(output_size = args.model_config["image_feature_output"])

        self.word_embeddings = nn.Embedding(question_vocab_size, args.model_config["word_embedding"])
        self.question_encoder = QuestionEmbedding(
            word_embedding_size = args.model_config["word_embedding"],
            hidden_size = args.model_config["rnn_hidden_size"]
            )
        self.mlp = nn.Sequential(
                nn.Linear(args.model_config["image_feature_output"], 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, ans_vocab_size))

    def forward(self, image, question, answer = None):
        image = self.image_encoder(image)
        question = self.word_embeddings(question)
        question = self.question_encoder(question)


        if self.args.task in ['acvqa']:
            answer = self.answer_encoder(answer)
            combine  = torch.mul(image,question)
            combine  = torch.mul(combine,answer)

        else:
            combine  = torch.mul(image,question)
        output = self.mlp(combine)
        return output
