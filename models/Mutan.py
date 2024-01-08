import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class ImageEncoder(nn.Module):
    def __init__(self, args, output_size=2048, att = False):
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

    def forward(self, question):
        _, hidden = self.gru(question)
        return hidden
    
    
class MutanFusion(nn.Module):
    def __init__(self, model_config, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__()
        print(model_config)
        self.model_config = model_config
        self.visual_embedding = visual_embedding
        self.question_embedding =question_embedding

        if self.visual_embedding:
            self.linear_v = nn.Linear(model_config["dim_v"], model_config["dim_hv"])
        if self.question_embedding:
            self.linear_q = nn.Linear(model_config["dim_q"], model_config["dim_hq"])

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(model_config['dim_hv'], model_config['dim_mm'])
            for i in range(model_config['R'])])
        
        self.list_linear_hq = nn.ModuleList([
            nn.Linear(model_config['dim_hq'], model_config['dim_mm'])
            for i in range(model_config['R'])])


    def forward(self, input_v, input_q):
        batch_size = input_v.size(0)
        if self.visual_embedding:
            x_v = nn.Dropout(p=self.model_config['dropout_v'])(input_v)
            x_v = self.linear_v(x_v)
            x_v = torch.tanh(x_v)
        else:
            x_v = input_v
        if self.question_embedding:
            x_q = nn.Dropout(p=self.model_config['dropout_q'])(input_q)
            x_q = self.linear_q(x_q)
            x_q = torch.tanh(x_q)
        else:
            x_q = input_q

        x_mm = []

        for i in range(self.model_config['R']):
            x_hv = nn.Dropout(p=self.model_config['dropout_hv'])(x_v)
            x_hv = self.list_linear_hv[i](x_hv)
            x_hv = torch.tanh(x_hv)

            x_hq = nn.Dropout(p=self.model_config['dropout_hq'])(x_q)
            x_hq = self.list_linear_hq[i](x_hq)
            x_hq = torch.tanh(x_hq)
            
            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.model_config['dim_mm'])
        x_mm = torch.tanh(x_mm)
        
        return x_mm



class MUTANnoAttVQA(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size, model_config):
        super(MUTANnoAttVQA, self).__init__()

        self.image_encoder = ImageEncoder(args, output_size = model_config["dim_v"], att = False)
        self.word_embeddings = nn.Embedding(question_vocab_size, model_config["word_embedding"])
        self.question_encoder = QuestionEmbedding(
            word_embedding_size = model_config["word_embedding"],
            hidden_size = model_config["dim_q"]
            )
        
        self.mutan = MutanFusion(model_config)
        self.mlp = nn.Sequential(nn.Linear( model_config["dim_mm"], ans_vocab_size))

    def forward(self, image, question):
        image_embeddings = self.image_encoder(image)
        embeds = self.word_embeddings(question)
        ques_embeddings = self.question_encoder(embeds)

        combined = self.mutan(image_embeddings, ques_embeddings)
        output = self.mlp(combined)

        return output
    
class MutanFusion2d(MutanFusion):
    def __init__(self, model_config, visual_embedding=True, question_embedding=True):
        super(MutanFusion2d, self).__init__(model_config)
    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 3:
            raise ValueError
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        dim_hv = input_v.size(2)
        dim_hq = input_q.size(2)
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()
        x_v = input_v.view(batch_size * weight_height, self.model_config['dim_hv'])
        x_q = input_q.view(batch_size * weight_height, self.model_config['dim_hq'])
        x_mm = super().forward(x_v, x_q)
        x_mm = x_mm.view(batch_size, weight_height, self.model_config['dim_mm'])
        return x_mm

    
class MUTANAttVQA(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size, model_config):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        
        super(MUTANAttVQA, self).__init__()
        ######################################################################################
        model_config['attention']['dim_v'] = model_config['attention']['dim_hv']
        model_config['attention']['dim_q'] = model_config['attention']['dim_hq']
        self.model_config = model_config
        
        self.fusion_att = MutanFusion2d(self.model_config['attention'],
                                                    visual_embedding=False,
                                                    question_embedding=False)
        
        self.list_linear_v_fusion = nn.ModuleList([
                    nn.Linear(self.model_config['dim_v'],
                            int(self.model_config['fusion']['dim_hv']
                                / self.model_config['attention']['nb_glimpses']))
                    for i in range(self.model_config['attention']['nb_glimpses'])])
        
        self.linear_q_fusion = nn.Linear(self.model_config['dim_q'],
                                         self.model_config['fusion']['dim_hq'])
        
        self.mlp = nn.Linear(self.model_config['fusion']['dim_mm'], ans_vocab_size)
        self.fusion_classif = MutanFusion(self.model_config['fusion'],
                                                 visual_embedding=False,
                                                 question_embedding=False)
        #######################################################################################
        
        self.conv_v_att = nn.Conv2d(model_config['dim_v'],
                                    model_config['attention']['dim_v'], 1, 1)
    
        self.linear_q_att = nn.Linear(self.model_config['dim_q'],
                                      self.model_config['attention']['dim_q'])
        self.conv_att = nn.Conv2d(self.model_config['attention']['dim_mm'],
                                  self.model_config['attention']['nb_glimpses'], 1, 1)
        

        self.image_encoder = ImageEncoder(args, output_size = model_config["dim_v"], att = True)
        self.word_embeddings = nn.Embedding(question_vocab_size, model_config["word_embedding"])

        self.question_encoder = QuestionEmbedding(
            word_embedding_size = model_config["word_embedding"],
            hidden_size = model_config["dim_q"]
            )

        
       

    def _attention(self, input_v, input_q):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)
        x_v = input_v
        x_v = nn.Dropout(p =self.model_config['attention']['dropout_v'])(x_v)
        x_v = self.conv_v_att(x_v)
        x_v = torch.tanh(x_v)
        x_v = x_v.view(batch_size,
                       self.model_config['attention']['dim_v'],
                       width * height)
        x_v = x_v.transpose(1,2)

        x_q = nn.Dropout(p =self.model_config['attention']['dropout_q'])(input_q)
        x_q = self.linear_q_att(x_q)
        x_q = torch.tanh(x_q)
        x_q = x_q.view(batch_size, 1, self.model_config['attention']['dim_q'])
        x_q = x_q.expand(batch_size, width * height, self.model_config['attention']['dim_q'])
        
        x_att = self.fusion_att(x_v, x_q)
        x_att = torch.tanh(x_att)
        x_att = nn.Dropout(p =self.model_config['attention']['dropout_mm'])(x_v)
        x_att = x_att.view(batch_size, width, height, self.model_config['attention']['dim_mm'])
        x_att = x_att.transpose(2,3).transpose(1,2)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size,
                           self.model_config['attention']['nb_glimpses'],
                           width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []

        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width*height)
            x_att = nn.Softmax(dim=1)(x_att)
            list_att.append(x_att)

        self.list_att = [x_att.data for x_att in list_att]
        x_v = input_v.view(batch_size, self.model_config['dim_v'], width * height)
        x_v = x_v.transpose(1,2)
        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               width * height,
                               1)
            x_att = x_att.expand(batch_size,
                                 width * height,
                                 self.model_config['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, self.model_config['dim_v'])
            list_v_att.append(x_v_att)
        return list_v_att

    def _fusion_glimpses(self, list_v_att, x_q_vec):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = nn.Dropout(p =self.model_config['attention']['dropout_mm'])(x_v_att)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            x_v = torch.tanh(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        # Process question
        x_q = nn.Dropout(p=self.model_config['fusion']['dropout_q'])(x_q_vec)
        x_q = self.linear_q_fusion(x_q)
        x_q = torch.tanh(x_q)
        # Second multimodal fusion
        x = self.fusion_classif(x_v, x_q)
        return x

    def forward(self, image, question):
        
        input_v = self.image_encoder(image)
        embeds = self.word_embeddings(question)
        input_q = self.question_encoder(embeds)
        list_v_att = self._attention(input_v, input_q)
        x = self._fusion_glimpses(list_v_att, input_q)
        x = torch.tanh(x)
        x = nn.Dropout(p=self.model_config['fusion']['dropout_q'])(x)
        x = self.mlp(x)

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

    MutanAtt = MUTANAttVQA(vocabulary_size, 3, model_config["MutanAtt"])

    output = MutanAtt(torch_images, padded_batch)


