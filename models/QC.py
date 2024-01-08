import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        # Pdb().set_trace()
        x = x / x.norm(p=self.p, dim=1, keepdim=True)
        return x
class ImageEncoder(nn.Module):

    def __init__(self, args, output_size=1024, image_channel_type='I'):
        super(ImageEncoder, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        if args.freeze_image_encoder == "T":
            for param in self.extractor.parameters():
                param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        #if image_channel_type.lower() == 'normi':
        #    extactor_fc_layers.append(Normalize(p=2))
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

class QC(nn.Module):

    def __init__(self, args, question_vocab_size, ans_vocab_size, model_config):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        super(QC, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(args, output_size = model_config["image_feature_output"], image_channel_type = model_config["image_channel_type"])
        self.word_embeddings_1 = nn.Embedding(question_vocab_size, model_config["word_embedding"])
        self.word_embeddings_2 = nn.Embedding(question_vocab_size, model_config["word_embedding"])
        self.question_encoder = QuestionEmbedding(
            word_embedding_size = model_config["word_embedding"],
            hidden_size = model_config["rnn_hidden_size"]
            )
        self.cap_encoder = QuestionEmbedding(
            word_embedding_size = model_config["word_embedding"],
            hidden_size = model_config["rnn_hidden_size"]
            )
        self.mlp = nn.Sequential(
                nn.Linear(model_config["image_feature_output"], 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, ans_vocab_size))

    def forward(self, caption, question):
        cap_embeds = self.word_embeddings_1(caption)
        cap_embeddings = self.cap_encoder(cap_embeds)

        ques_embeds = self.word_embeddings_2(question)
        ques_embeddings = self.question_encoder(ques_embeds)
        print("#"*1000)
        print(cap_embeddings.size)
        print(ques_embeddings.size)

        combined =torch.concat(cap_embeddings, ques_embeddings, )
        print(combined.size)
        output = self.mlp(combined)
        if self.args.dataset not in ["simpsonsvqa"]:
            output = torch.sigmoid(output)
        return output
