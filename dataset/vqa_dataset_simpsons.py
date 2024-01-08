import os
import json
import random
from random import random as rand

import numpy as np
from torch.utils.data import Dataset
from dataset.utils import pre_question

from torchvision.transforms.functional import hflip
import torch
from transformers import BertTokenizer, RobertaTokenizer
from dataset.preprocessing_simpsons import *

import random

import os
import json
import torch
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, args, transform, split='train'):
        self.transform = transform
        self.split = split
        self.args = args
        self.token_to_ix, self.pretrained_emb, self.token_size = self.prepare_question_vocab()
        self.ans_to_ix, self.ix_to_ans = self.prepare_answer_vocab()
        self.ans_size = len(self.ans_to_ix)
        
        self.questions = self.load_questions()
        self.annotations, self.idx_to_ans  = self.load_annotations() if split != 'test' else ([], [])
        
        random.shuffle(self.annotations)

        
    def load_json_file(self, file_path):
        """
        Load a JSON file from the given file path.
        """
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except IOError as e:
            print(f"Error opening {file_path}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        return None
    
    def qid_to_ques(self, ques_list, img_path):
        """
        Process and map questions to their corresponding image paths.
        """
        qid_to_ques_dict = {}
        for ques in ques_list:
            qid = str(ques['id'])
            ques['img_path'] = os.path.join(img_path, ques['img_path'])
            ques['question'] = rnn_proc_ques(ques["question"], self.token_to_ix, self.args.max_ques_len)
            qid_to_ques_dict[qid] = ques
        return qid_to_ques_dict
    
    def load_questions(self):
        """ Load questions from the dataset. """
        questions = {}
        for ques_path, img_path in zip(self.args.data_config[f'{self.split}_que'], self.args.data_config[f'{self.split}_image']):
            full_path = os.path.join(self.args.data_path, ques_path)
            loaded_questions = self.qid_to_ques(self.load_json_file(full_path)['questions'], img_path)
            questions.update(loaded_questions)
        return questions
    
    def load_annotations(self):
        """ Load annotations from the dataset. """
        annotations = []
        for ann_path in self.args.data_config[f'{self.split}_ann']:
            full_path = os.path.join(self.args.data_path, ann_path)
            annotations.extend(self.load_json_file(full_path)['annotations'])
            
        annotations = annotation_preprocessing(annotations)
        idx_to_ans = {}
        processed_ann = []
        for ann in annotations:
            idx_to_ans[ann["id"]] = ann
            answer_str_qt = ann["answer"]
            ann["answer_idx"] = self.ans_to_ix[answer_str_qt]
            processed_ann.append(ann)
        return processed_ann, idx_to_ans

    def prepare_question_vocab(self):
        """
        Prepare the vocabulary for question processing.
        """
        stat_ques_list = []
        for file in self.args.data_config["stat_ques_list"]:
            ques_path  = os.path.join(self.args.data_path, file)
            stat_ques_list += self.load_json_file(ques_path)['questions']

        token_to_ix, pretrained_emb = question_vocal(stat_ques_list, self.args)
        token_size = len(token_to_ix)
        return token_to_ix, pretrained_emb, token_size
    
    def prepare_answer_vocab(self):
        """
        Prepare the vocabulary for answer processing in the dataset.
        """
        stat_ans_list = []
        for file in self.args.data_config["stat_answer_list"]:
            ans_path  = os.path.join(self.args.data_path, file)
            stat_ans_list += self.load_json_file(ans_path)['annotations']
        stat_ans_list = annotation_preprocessing(stat_ans_list)
        
        ans_to_ix, ix_to_ans = annotation_vocal(stat_ans_list, self.args)
        return ans_to_ix, ix_to_ans
        
    
    def __len__(self):
        if self.split =="train":
            return self.args.batch_size_train*2
        return len(self.annotations)
    
    
    def example_processing(self, image, ques, ann):
        """
        Prepare the vocabulary for answer processing in the dataset.
        """
        # question = rnn_proc_ques(ques["question"], self.token_to_ix, self.args.max_ques_len)
        question=  torch.from_numpy(ques["question"])
        if self.split != 'test':
            return image, question, ann["answer_idx"], ann['answer'], ann['id']
        else:
            return image, question, ques['id']
    
    def __getitem__(self, index):
        
        if self.split in ['train', 'val'] or self.args.task == "acvqa":
            ann = self.annotations[index]
            ques = self.questions[str(ann["id"])]
        else:
            ques = self.question_list[index]
            ann = {}

        image_path = ques["img_path"]
        image = image_preprocessing(image_path, self.transform, self.args)
        return self.example_processing(image, ques, ann)

            

def annotation_preprocessing( anns):
    proccesed_anns = []
    for ann in anns:
        ans_count = 0
        for judge in ann["judgements"].values():
            if judge["answer"] == 1:
                ans_count += 1
        if ans_count >= 2:
            ann["question_label"]= 1
            ann["answer"] = prep_ans(ann["answer"])
            proccesed_anns.append(ann)
    return proccesed_anns
