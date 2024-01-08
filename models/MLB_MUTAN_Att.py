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
#import fusion
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
        image_embedding = self.fc(features)
        return image_embedding
    

class QuestionEmbedding(nn.Module):
    def __init__(self, word_embedding_size = 500, hidden_size = 2048):
        super(QuestionEmbedding, self).__init__()
        self.gru = nn.GRU(word_embedding_size, hidden_size, batch_first = True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, question):
        _, hidden = self.gru(question)
        embedding = hidden.squeeze(0)
        #embedding = self.fc(embedding)
        #embedding = self.tanh(embedding)
        return embedding
    
class AbstractAtt(nn.Module):

    def __init__(self,  opt, question_vocab_size, ans_vocab_size):
        super(AbstractAtt, self).__init__()
        self.opt = opt
        self.opt = opt
        self.image_encoder = ImageEncoder(output_size = self.opt["dim_v"], att = True)
        self.word_embeddings = nn.Embedding(question_vocab_size,  self.opt["word_embedding"])
        self.question_encoder = QuestionEmbedding(
            word_embedding_size = opt["word_embedding"],
            hidden_size =  opt["dim_q"]
            )
        # Modules for attention
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['attention']['dim_v'], 1, 1)
        self.linear_q_att = nn.Linear(self.opt['dim_q'],
                                      self.opt['attention']['dim_q'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'],
                                  self.opt['attention']['nb_glimpses'], 1, 1)
        # Modules for classification
        self.list_linear_v_fusion = None
        self.linear_q_fusion = None
        self.linear_classif = None

    def _fusion_att(self, x_v, x_q):
        raise NotImplementedError

    def _fusion_classif(self, x_v, x_q):
        raise NotImplementedError

    def _attention(self, input_v, x_q_vec):
        
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)

        # Process visual before fusion
        #x_v = input_v.view(batch_size*width*height, dim_features)
        x_v = input_v
        x_v = F.dropout(x_v,
                        p=self.opt['attention']['dropout_v'],
                        training=self.training)
        x_v = self.conv_v_att(x_v)
        if 'activation_v' in self.opt['attention']:
            x_v = torch.tanh(x_v)
        x_v = x_v.view(batch_size,
                       self.opt['attention']['dim_v'],
                       width * height)
        x_v = x_v.transpose(1,2)

        # Process question before fusion
        x_q = F.dropout(x_q_vec, p=self.opt['attention']['dropout_q'],
                           training=self.training)
        x_q = self.linear_q_att(x_q)
        if 'activation_q' in self.opt['attention']:
            x_q = torch.tanh(x_q)
        x_q = x_q.view(batch_size,
                       1,
                       self.opt['attention']['dim_q'])
        x_q = x_q.expand(batch_size,
                         width * height,
                         self.opt['attention']['dim_q'])

        # First multimodal fusion
        x_att = self._fusion_att(x_v, x_q)

        if 'activation_mm' in self.opt['attention']:
            x_att =torch.tanh(x_att)

        # Process attention vectors
        x_att = F.dropout(x_att,
                          p=self.opt['attention']['dropout_mm'],
                          training=self.training)
        # can be optim to avoid two views and transposes
        x_att = x_att.view(batch_size,
                           width,
                           height,
                           self.opt['attention']['dim_mm']) 
        x_att = x_att.transpose(2,3).transpose(1,2)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size,
                           self.opt['attention']['nb_glimpses'],
                           width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width*height)
            x_att = F.softmax(x_att,-1)
            list_att.append(x_att)

        self.list_att = [x_att.data for x_att in list_att]

        # Apply attention vectors to input_v
        x_v = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v = x_v.transpose(1,2)

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               width * height,
                               1)
            x_att = x_att.expand(batch_size,
                                 width * height,
                                 self.opt['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
            list_v_att.append(x_v_att)

        return list_v_att

    def _fusion_glimpses(self, list_v_att, x_q_vec):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att,
                            p=self.opt['fusion']['dropout_v'],
                            training=self.training)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            if 'activation_v' in self.opt['fusion']:
                x_v = torch.tanh(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        # Process question
        x_q = F.dropout(x_q_vec,
                        p=self.opt['fusion']['dropout_q'],
                        training=self.training)
        x_q = self.linear_q_fusion(x_q)
        if 'activation_q' in self.opt['fusion']:
            x_q = torch.tanh(x_q)

        # Second multimodal fusion
        x = self._fusion_classif(x_v, x_q)
        return x

    def _classif(self, x):

        if 'activation' in self.opt['classif']:
            x = torch.tanh(x)
        x = F.dropout(x,
                      p=self.opt['classif']['dropout'],
                      training=self.training)
        x = self.linear_classif(x)
        return x

    def forward(self, input_v, input_q):
        if input_v.dim() != 4 and input_q.dim() != 2:
            raise ValueError
        input_v = self.image_encoder(input_v)
        input_q = self.word_embeddings(input_q)
        input_q = self.question_encoder(input_q)
        x_q_vec = input_q
        list_v_att = self._attention(input_v, x_q_vec)
        x = self._fusion_glimpses(list_v_att, x_q_vec)
        x = self._classif(x)
        return x

class MLBAtt(AbstractAtt):

    def __init__(self, opt, question_vocab_size, ans_vocab_size):
        # TODO: deep copy ?
        opt['attention']['dim_v']  = opt['attention']['dim_h']
        opt['attention']['dim_q']  = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(MLBAtt, self).__init__(opt, question_vocab_size, ans_vocab_size)
        # Modules for classification
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'],
                      self.opt['fusion']['dim_h'])
            for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['fusion']['dim_h']
                                         * self.opt['attention']['nb_glimpses'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_h']
                                        * self.opt['attention']['nb_glimpses'],
                                        ans_vocab_size)

    def _fusion_att(self, x_v, x_q):
        x_att = torch.mul(x_v, x_q)
        return x_att

    def _fusion_classif(self, x_v, x_q):
        x_mm = torch.mul(x_v, x_q)
        return x_mm


class MutanAtt(AbstractAtt):

    def __init__(self, opt, question_vocab_size, ans_vocab_size):
        # TODO: deep copy ?
        opt['attention']['dim_v'] = opt['attention']['dim_hv']
        opt['attention']['dim_q'] = opt['attention']['dim_hq']
        super(MutanAtt, self).__init__(opt, question_vocab_size, ans_vocab_size)
        # Modules for classification
        self.fusion_att = fusion.MutanFusion2d(self.opt['attention'],
                                               visual_embedding=False,
                                               question_embedding=False)
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'],
                      int(self.opt['fusion']['dim_hv']
                          / opt['attention']['nb_glimpses']))
            for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['fusion']['dim_hq'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_mm'],
                                        ans_vocab_size)
        self.fusion_classif = fusion.MutanFusion(self.opt['fusion'],
                                                 visual_embedding=False,
                                                 question_embedding=False)

    def _fusion_att(self, x_v, x_q):
        return self.fusion_att(x_v, x_q)

    def _fusion_classif(self, x_v, x_q):
        return self.fusion_classif(x_v, x_q)
    


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

    MutanAtt = MutanAtt(model_config["MutanAtt"], vocabulary_size, 3 )
    output = MutanAtt(torch_images, padded_batch)

    print("Debug MLB")

    MLBAtt = MLBAtt(model_config["MLBAtt"], vocabulary_size, 3, )

    output = MLBAtt(torch_images, padded_batch)


