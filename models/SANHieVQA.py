import torch
import torch.nn as nn
from torchvision import models

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


        # Mapping from question type index to string key
        self.question_type_dict = idx_to_answer_type

    def forward(self, hidden_states, question_type_output):
        max_values, predicted_classes  = torch.max(question_type_output, 1)
        predicted_classes = predicted_classes.cpu().tolist()
        predicted_classes = [self.question_type_dict[predicted_class] for predicted_class in predicted_classes]

        holder_answers = torch.zeros(hidden_states.shape[0], self.max_number_answer).cuda()
        for i, q_type in enumerate(predicted_classes):
            outputs = self.vqa_headers[q_type](hidden_states[i])
            output_size = outputs.size()[0]
            holder_answers[i, :output_size] = outputs
        return holder_answers
    
class Ques_Ans_Attention(nn.Module):
    def __init__(self, args, d=1024, k=512, dropout=True):
        super(Ques_Ans_Attention, self).__init__()
        self.args = args
        self.dropout = dropout

        # Linear transformation layers
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        self.ff_attention = nn.Linear(k, 1)

        if self.dropout:
            self.dropout_ = nn.Dropout(p=0.5)

    def debug_print(self, message, tensor):
        if self.args.debug_print:
            print(f"{message}: {tensor.size()}")

    def forward(self, vi, vq, train=True):
        hi = self.ff_image(vi)
        self.debug_print("image transformed feature size", hi)

        hq = self.ff_ques(vq).unsqueeze(dim=1)
        self.debug_print("unsqueeze Question transformed feature size", hq)

        ha = torch.tanh(hi + hq)
        self.debug_print("Element-wise sum size", ha)

        if self.dropout and train:
            ha = self.dropout_(ha)

        ha = self.ff_attention(ha).squeeze(dim=2)
        self.debug_print("Reduced Attention weights size", ha)

        pi = torch.softmax(ha, dim=1)
        self.debug_print("Attention distribution size", pi)

        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        self.debug_print("Attended Visual Features", vi_attended)

        u = vi_attended + vq
        return u

class SANHieVQA(nn.Module):
    """
    A Visual Question Answering (VQA) model
    """
    def __init__(self, args, question_vocab_size, ans_vocab_type_dict, idx_to_answer_type):
        super(SANHieVQA, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(image_feature_output=args.model_config["image_feature_output"])

        self.word_embeddings = nn.Embedding(question_vocab_size, args.model_config["word_embedding"])
        self.question_encoder = QuestionEmbedding(
            word_embedding_size=args.model_config["word_embedding"],
            hidden_size=args.model_config["rnn_hidden_size"])
        
        self.vqa_mlp = VQA_header(args, ans_vocab_type_dict, idx_to_answer_type)
    
        self.questiont_type_mlp = QuestionType(args, idx_to_answer_type)
        self.san = nn.ModuleList([Ques_Ans_Attention(self.args, args.model_config["image_feature_output"], 512)] * args.model_config["num_att_layers"])
        

    def debug_print(self, message):
        """
        Prints debug information if debug mode is enabled.
        """
        if self.args.debug_print:
            print(message)

    def forward(self, image, question, train=True):
        # Extract Image Features
        self.debug_print("Extract Question and Image Features")
        image = self.image_encoder(image)

        # Extract Question Features
        self.debug_print(f"image feature: {image.size()}")
        question = self.word_embeddings(question)
        self.debug_print(f"question Word Embedding: {question.size()}")
        question = self.question_encoder(question)
        self.debug_print(f"question feature: {question.size()}")

        # Question Guided Attention
        self.debug_print("\nQuestion-guided Attention")

        vi = image
        u = question
        for attn_layer in self.san:
            u = attn_layer(vi, u)
        attended_features = u


        self.debug_print(f"Fusion feature: {attended_features.size()}")

        # Calculate losses if in training mode
        self.debug_print("\nQuestion Type Problem")
        question_type_output = self.questiont_type_mlp(attended_features)
        self.debug_print(f"question_type_output size: {question_type_output.size()}")

        vqa_output =  self.vqa_mlp(attended_features, question_type_output)
        return question_type_output, vqa_output
