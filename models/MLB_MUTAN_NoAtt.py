import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from models import fusion
from models.headers import VQA_header, QR_header, AC_header

class ImageEncoder(nn.Module):
    def __init__(self,output_size=2048, att = False):
        super(ImageEncoder, self).__init__()
        self.att = att
        self.cnn = models.resnet152(pretrained=True)

        
        modules = list(self.cnn.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        # Freeze the parameters of the CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(14 * 14 * 2048, output_size),
            nn.Tanh()
        )

    def forward(self, image):
        features = self.cnn(image)
        if self.att:
            return features
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features
    
class QuestionEmbedding(nn.Module):
    def __init__(self, word_embedding_size = 500, hidden_size = 2048):
        super(QuestionEmbedding, self).__init__()
        self.gru = nn.GRU(word_embedding_size, hidden_size, batch_first = True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, question):
        _, hidden = self.gru(question)
        embedding = hidden.squeeze(0)
        embedding = self.fc(embedding)
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
    
class AbstractNoAtt(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size):
        super(AbstractNoAtt, self).__init__()
        print(args)

        self.opt = args
        self.image_encoder = ImageEncoder(output_size = self.opt.model_config['fusion']["dim_v"], att = False)
        self.word_embeddings = nn.Embedding(question_vocab_size,  self.opt.model_config['fusion']["word_embedding"])
        self.question_encoder = QuestionEmbedding(
            word_embedding_size = args.model_config['fusion']["word_embedding"],
            hidden_size =  args.model_config['fusion']["dim_q"]
            )
        

        # self.linear_classif = nn.Linear(self.opt['fusion']['dim_h'], ans_vocab_size)
        args.model_config["image_feature_output"]= self.opt.model_config['fusion']['dim_h']
        if args.task in ["cvqa", "mergevqa"]:
            self.mlp = VQA_header(args, ans_vocab_size)
        elif args.task in ["qrvqa"]:
            self.mlp = QR_header(args)
        elif args.task in ['acvqa']:
            self.answer_encoder = AnswerEncoder(
                answer_vocal_size = ans_vocab_size,
                word_embedding_size = args.model_config["word_embedding"],
                hidden_size = args.model_config["rnn_hidden_size"]
                )
            self.mlp = AC_header(args)

    def _fusion(self, input_v, input_q):
        raise NotImplementedError

    def forward(self, input_v, input_q):
        input_v = self.image_encoder(input_v)
        input_q = self.word_embeddings(input_q)
        input_q = self.question_encoder(input_q)

        x = self._fusion(input_v, input_q)
        x = self.mlp(x)
        return x

class MLBNoAtt(AbstractNoAtt):

    def __init__(self,  opt, question_vocab_size, ans_vocab_size):
        super(MLBNoAtt, self).__init__(opt, question_vocab_size, ans_vocab_size)
        self.fusion = fusion.MLBFusion(self.opt.model_config['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


class MutanNoAtt(AbstractNoAtt):

    def __init__(self, opt, question_vocab_size, ans_vocab_size):
        opt.model_config['fusion']['dim_h'] = opt.model_config['fusion']['dim_mm']
        super(MutanNoAtt, self).__init__(opt, question_vocab_size, ans_vocab_size)
        self.fusion = fusion.MutanFusion(self.opt.model_config['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x

    


if __name__ == '__main__':

    #################################################### QUESTION #######################################################
    batch_ques = [
    [1, 5, 3, 8, 4, 9],       # Sequence 1
    [2, 7, 6, 2, 1],          # Sequence 2
    [3, 4, 1, 0, 0],          # Sequence 3
    [9, 2, 5, 6, 8, 7, 1],    # Sequence 4
    [1, 3, 5, 2, 0, 0, 0],    # Sequence 5
    [4, 3, 2, 6, 5, 7, 8, 9], # Sequence 6
    [1, 4, 2, 8, 0, 0, 0, 0], # Sequence 7
    [5, 7, 3, 2, 1, 0, 0, 0]  # Sequence 8
    ]

    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in batch_ques], batch_first=True, padding_value=0)
    #print(f"question batch shape: {padded_batch.shape}")
    vocabulary_size = 10
    embedding_dim = 16
    #word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
    #embedded_batch = word_embeddings(padded_batch)
    #embedded_batch = embedded_batch
    #embedded_batch = torch.tensor(embedded_batch).to(torch.int64)
    #question_encoder = QuestionEmbedding(
    #        word_embedding_size = 16,
    #        hidden_size = 2400
    #        )
    #ques_feature = question_encoder(embedded_batch)
    #print(f"question features shape: {ques_feature.shape}")
    
    #################################################### IMAGE #######################################################
    batch_size = 8
    image_height = 448
    image_width = 448
    channels = 3  # Assuming RGB images
    random_images = np.random.randint(0, 256, size=(batch_size, image_height, image_width, channels), dtype=np.uint8)
    torch_images = torch.tensor(random_images, dtype=torch.float32)
    torch_images /= 255.0
    torch_images = torch_images.permute(0, 3, 1, 2)
    torch_images = torch_images

    #print(f"image batch shape: {torch_images.shape}")
    #image_encoder = ImageEncoder(output_size = 2048)
    #image_feature = image_encoder(torch_images)
    #print(f"image feature shape: {image_feature.shape}")

    print("Debug MUTAN")

    MutanAtt = MutanNoAtt(model_config["MutanNoAtt"], vocabulary_size, 3, )

    print("Debug MLB")

    MutanAtt = MLBNoAtt(model_config["MLBNoAtt"], vocabulary_size, 3, )

    output = MutanAtt(torch_images, padded_batch)


