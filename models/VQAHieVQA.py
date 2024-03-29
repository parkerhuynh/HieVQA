import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):

    def __init__(self, image_feature_output=1024):
        super(ImageEncoder, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        self.fflayer = nn.Sequential(
            nn.Linear(4096, image_feature_output),
            nn.Tanh())

    def forward(self, image):
        image = self.extractor(image)
        image_embedding = self.fflayer(image)
        return image_embedding


class QuestionEmbedding(nn.Module):
    """
    A question embedding module using GRU for text encoding.
    """
    def __init__(self, word_embedding_size, hidden_size):
        super(QuestionEmbedding, self).__init__()
        # Define a GRU layer
        self.gru = nn.GRU(word_embedding_size, hidden_size, num_layers=1, batch_first=True)
        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size, 1024)
        # Define a Tanh activation function
        # self.tanh = nn.Tanh()

    def forward(self, input_data):
        # Process input through the GRU layer
        output, hidden = self.gru(input_data)
        # Use the last hidden state
        last_hidden = hidden.squeeze(0)
        # Pass the last hidden state through the fully connected layer and Tanh activation
        embedding = self.fc(last_hidden)
        # embedding = self.tanh(embedding)
        return embedding


class QuestionType(nn.Module):
    def __init__(self, args, idx_to_answer_type):
        """
        Initializes the QuestionType model with a Multilayer Perceptron (MLP) architecture.
        """
        super().__init__()
        self.qt_header = nn.Sequential(
            nn.Linear(args.model_config["image_feature_output"], 1000),
            nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(1000, len(idx_to_answer_type))
        )
        self.qt_loss_fn = nn.CrossEntropyLoss()

    def forward(self, hidden_features, train=True):
        """
        INPUT:
        question_feature (Tensor): The input features for the questions.
        question_types (Tensor): The true labels for the question types.
        train (bool): Flag indicating whether the model is in training mode.

        OUTPUT
        outputs (Tensor): The predicted class probabilities for each question type.
        qt_loss : Cross-Entropy loss, returned only in training mode.
        """
        outputs = self.qt_header(hidden_features)
        return outputs

class VQA_header(nn.Module):
    """
    A Visual Question Answering (VQA) header module.
    """

    def __init__(self,args, ans_vocab_type_dict, idx_to_answer_type):
        super().__init__()

        # ModuleDict for VQA headers
        self.vqa_headers = nn.ModuleDict({
            answer_type: nn.Sequential(
                nn.Linear(1024, 1000),
                nn.Dropout(p=0.5),
                nn.Tanh(),
                nn.Linear(1000, len(ans_vocab_type_dict[answer_type]))
            ).cuda() for answer_type in ans_vocab_type_dict.keys()
        })
        number_classes = []
        for answer_type in ans_vocab_type_dict.keys():
            number_classes.append(len(ans_vocab_type_dict[answer_type]))
        self.max_number_answer = max(number_classes)
        self.vqa_loss_func = torch.nn.CrossEntropyLoss()

        # Mapping from question type index to string key
        self.question_type_dict = idx_to_answer_type

    def forward(self, hidden_states):
        results = {}
        for question_category in self.vqa_headers.keys():
            outputs = self.vqa_headers[question_category](hidden_states)
            results[question_category] = outputs
        return results
    

class VQAHieVQA(nn.Module):
    """
    A Visual Question Answering (VQA) model
    """
    def __init__(self, args, question_vocab_size, ans_vocab_type_dict, idx_to_answer_type):
        super(VQAHieVQA, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(image_feature_output=args.model_config["image_feature_output"])

        self.word_embeddings = nn.Embedding(question_vocab_size, args.model_config["word_embedding"])
        self.question_encoder = QuestionEmbedding(
            word_embedding_size=args.model_config["word_embedding"],
            hidden_size=args.model_config["rnn_hidden_size"])
        
        self.vqa_mlp = VQA_header(args, ans_vocab_type_dict, idx_to_answer_type)
    
        self.questiont_type_mlp = QuestionType(args, idx_to_answer_type)
        self.qt_loss_func = torch.nn.CrossEntropyLoss()
    def debug_print(self, message):
        """
        Prints debug information if debug mode is enabled.
        """
        if self.args.debug_print:
            print(message)

    def forward(self, image, question):
        image = self.image_encoder(image)
        question = self.word_embeddings(question)
        question = self.question_encoder(question)
        combine_features = image*question
        question_type_output = self.questiont_type_mlp(combine_features)
        vqa_outputs =  self.vqa_mlp(combine_features)
        return question_type_output, vqa_outputs