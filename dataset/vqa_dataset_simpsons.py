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
        self.super_types = self.load_json_file('/home/reda/scratch/ngoc/code/HieVQA/dataset/super_answer_type_simpsons.json')
        self.ans_to_ix, self.ix_to_ans = self.prepare_answer_vocab()
        self.ans_size = len(self.ans_to_ix)
        self.ans_type_to_idx = {}
        self.idx_to_ans_type = {}
        
        for i, ans_type in enumerate(self.ans_to_ix.keys()):
            self.ans_type_to_idx[ans_type] = i
            self.idx_to_ans_type[i] = ans_type
            
        self.questions = self.load_questions()
        self.annotations, self.idx_to_ann  = self.load_annotations() if split != 'test' else ([], [])
        
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
        idx_to_ann = {}
        processed_ann = []
        for ann in annotations:
            # print(ann)
            idx_to_ann[ann["id"]] = ann
            
            origine_ans_str = ann["original_answer"]
            ans_str = ann["answer"]
            at_str = self.super_types[origine_ans_str]
            
            ans_to_idx_dict = self.ans_to_ix[at_str]
            
            at_idx = self.ans_type_to_idx[at_str]
            ans_idx = ans_to_idx_dict[ans_str]
            
            ann["answer_type_idx"] = at_idx
            ann["answer_idx"] = ans_idx
            processed_ann.append(ann)
        return processed_ann, idx_to_ann

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
        
        ans_to_ix, ix_to_ans = annotation_vocal(stat_ans_list, self.args, self.super_types)
        return ans_to_ix, ix_to_ans
        
    
    def __len__(self):
        # if self.args.debug:
        #     return self.args.batch_size_train*4
        # if self.split == "train":
        #     return self.args.batch_size_train*4
        return self.args.batch_size_train*4 #len(self.annotations)
    
    
    def example_processing(self, image, ques, ann):
        """
        Prepare the vocabulary for answer processing in the dataset.
        """
        # question = rnn_proc_ques(ques["question"], self.token_to_ix, self.args.max_ques_len)
        question=  torch.from_numpy(ques["question"])
        # print(ques)
        # print(ann)
        # print(ann["answer_idx"], ann['answer_type_idx'], ann['id'])
        return image, question, ann["answer_idx"], ann['answer_type_idx'], ann['id']
    
    def __getitem__(self, index):
        
        if self.split in ['train', 'val']:
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
            ann["answer"] = prep_ans(ann["answer"])
            ann["original_answer"] = ann["answer"]
            ann["processed_answer_type"] = ann["answer_type"]
            proccesed_anns.append(ann)
        elif ann["overall_scores"]["question"] < 0.5:
            ann["original_answer"] = ann["answer"]
            ann["answer"] = "unanswerable"

            ann["processed_answer_type"] = "unanswerable"
            proccesed_anns.append(ann)
    return proccesed_anns
