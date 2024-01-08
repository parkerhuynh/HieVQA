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
from dataset.preprocessing_vqav2 import *

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
        self.annotations = []
        self.questions = {}
        self.question_list = []
        self.prepare_answer_vocab()
        self.prepare_question_vocab()
        
        self.load_data()
        
        # self.example_proc_ans = getattr(self, f"{self.args.dataset}_proc_ans")
        # self.example_processing = getattr(self, f"{self.args.task}_example_processing")

        # random.shuffle(self.annotations)

    
    
    def load_data(self):
        
        # Iterate over paths for questions, and images
        for ques_path, img_path in zip(self.args.data_config[f'{self.split}_que'], 
                                    self.args.data_config[f'{self.split}_image']):
            # Construct full question file path
            ques_path = os.path.join(self.args.data_path, ques_path)

            # Load and update questions
            loaded_questions = self.load_json_file(ques_path)['questions']
            self.questions.update(self.ques_load(loaded_questions, img_path))
            self.question_list += loaded_questions
        

        # Load annotations only if not in 'test' split
        if self.split != 'test':
            for ann_path in self.args.data_config[f'{self.split}_ann']:
                print(ann_path)
                # Construct full annotation file path
                ann_path = os.path.join(self.args.data_path, ann_path)
                # Load and append annotations
                loaded_annotations = self.load_json_file(ann_path)['annotations']
                self.annotations += loaded_annotations
            filtered_ann = []
            for ann in self.annotations:
                ann["multiple_choice_answer"] = prep_ans(ann["multiple_choice_answer"])
                if ann["multiple_choice_answer"] in self.top_answer_list:
                    filtered_ann.append(ann)
            print(f"The exaples reduced from {len(self.annotations)} to {len(filtered_ann)}!")
            self.annotations = filtered_ann
            
            self.answer_type_to_idx= {
                'object': 0, 
                'color': 1, 
                'yes/no': 2, 
                'action': 3, 
                'number': 4, 
                'other': 5,
                'location': 6,
                'unanswerable': 7,
                'human': 8
                }
            self.idx_to_answer_type= {
                0:'object', 
                1:'color', 
                2:'yes/no', 
                3:'action', 
                4:'number', 
                5:'other',
                6:'location',
                7:'unanswerable',
                8:'human'
            }
            

        
    def load_json_file(self, file_path):
        """
        Load a JSON file from the given file path.
        Args:
            file_path (str): The path to the JSON file.
        Returns:
            dict: The loaded JSON data.
        """
        try:
            with open(file_path, 'r') as file:
                print(file_path)
                return json.load(file)
        except IOError as e:
            print(f"Error opening {file_path}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        return None
    
    def __len__(self):
        if self.args.debug:
            return self.args.batch_size_train*10
        return len(self.question_list) if self.split == "test" else len(self.annotations)


    def prepare_answer_vocab(self):
        """
        Prepare the vocabulary for answer processing in the dataset.
        """
        stat_ans_list = []
        
        for file in self.args.data_config["stat_answer_list"]:
            ans_path  = os.path.join(self.args.data_path, file)
            stat_ans_list += self.load_json_file(ans_path)['annotations']
    
        self.ans_to_ix, self.ix_to_ans, self.top_answer_list = annotation_vocal_load(stat_ans_list, self.args)
        self.answer_super_types = json.load(open(self.args.data_config["super_type"], 'r'))
        self.ans_size = len(self.ans_to_ix)
        
    def prepare_question_vocab(self):
        """
        Prepare the vocabulary for question processing.
        """
        stat_ques_list = []
        for file in self.args.data_config["stat_ques_list"]:
            ques_path  = os.path.join(self.args.data_path, file)
            stat_ques_list += self.load_json_file(ques_path)['questions']

        # Create token-index mapping and load pretrained embeddings for questions
        self.token_to_ix, self.pretrained_emb = question_token_load(stat_ques_list, self.args)
        # Calculate the size of the token vocabulary
        self.token_size = len(self.token_to_ix)
        
    def ques_load(self, ques_list, img_path):
        """
        Process and map questions to their corresponding image paths.
        Args:
            ques_list (list): A list of question dictionaries.
            img_path (str): The base path to the images.
        Returns:
            dict: A dictionary mapping question IDs to their corresponding question data,
                including the updated image path.
        """
        qid_to_ques = {}
        for ques in ques_list:
            image_dict = {
                "train": "COCO_train2014_",
                "val": "COCO_val2014_"
            }
            # Extract the question ID and convert it to a string
            qid = str(ques['question_id'])
            img_name = image_dict[self.split] + str(ques["image_id"]).zfill(12) + ".jpg"
            # Update the image path in the question dictionary
            ques['img_path'] = os.path.join(img_path, img_name)

            # Map the question ID to the question dictionary
            qid_to_ques[qid] = ques
        return qid_to_ques
    
            
    def example_processing(self, image, ques, ann):
        """
        Prepare the vocabulary for answer processing in the dataset.
        """
        question = rnn_proc_ques(ques["question"], self.token_to_ix, self.args.max_ques_len)
        question=  torch.from_numpy(question)

        answer_str = ann["multiple_choice_answer"]

        answer_type = self.answer_super_types[answer_str]
        question_dict = self.ans_to_ix[answer_type]

        answer_type_idx = self.answer_type_to_idx[answer_type]
        answer_idx = question_dict[answer_str]
        
        
        # print(answer_type_idx, answer_idx)

        if self.split != 'test':
            return image, question, answer_type_idx, answer_idx
        else:
            return image, question, ques['id']
    
    def __getitem__(self, index):
        
        ann = self.annotations[index]
        ques = self.questions[str(ann["question_id"])]

        image_path = ques["img_path"]
        image = image_preprocessing(image_path, self.transform, self.args)
        return self.example_processing(image, ques, ann)