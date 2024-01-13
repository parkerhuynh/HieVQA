import os
from PIL import Image
import sys
import json
import re
import en_core_web_lg, random, re, json
import numpy as np
import pickle
import torch
contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}
manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', #'-',
                '>', '<', '@', '`', ',', '?', '!']

def image_preprocessing(image_path, transform, args):
    image_path = os.path.join(args.data_path, image_path)
    image = Image.open(image_path).convert('RGB')
    image = transform(image)

    return image

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def tokenize(stat_ques_list, args):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if args.model_config["use_glove"]:
        spacy_tool = en_core_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if args.model_config["use_glove"]:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)
    return token_to_ix, pretrained_emb

def question_vocal(question_list, args):
    question_path = args.data_config['question_dict']
    
    
    if os.path.exists(question_path):
        print(f'> Loading {args.dataset} question dictionary'.upper())
        token_to_ix, pretrained_emb = pickle.load(open(question_path, 'rb'))
    else:
        print(f'> Create and save {args.dataset} question dictionary'.upper())
        token_to_ix, pretrained_emb = tokenize(question_list, args)
        pickle.dump([token_to_ix, pretrained_emb], open(question_path, 'wb'))
    return token_to_ix, pretrained_emb


def prep_ans(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


def create_ann_vocal(examples, args):
    ans2tok, tok2ans = {"unanswerable": 0}, {0: "unanswerable"}
    id = {}
    
    for ex in examples:
        ans_str = ex["answer"]
        ans_type = ex["answer_type"]
        if ans_str != "unanswerable":
            if ans_type not in ans2tok:
                ans2tok[ans_type] = {}
                tok2ans[ans_type] = {}
                id[ans_type] = 0
            if ans_str not in ans2tok[ans_type]:
                id_i = id[ans_type]
                ans2tok[ans_type][ans_str] = id_i
                tok2ans[ans_type][id_i] = ans_str
                id[ans_type] += 1
    return ans2tok, tok2ans 


def annotation_vocal(anns, args):
    annotation_path = f"./dataset/ann_vocal_{args.task}_{args.dataset}.json"
    
    if os.path.exists(annotation_path):
        # print("3"*100)
        print(f'> Loading {args.dataset} annotation dictionary'.upper())
        ans_to_ix, ix_to_ans = json.load(open(annotation_path, 'r'))
    else:
        print(f'> Create and save {args.dataset} annotation dictionary'.upper())
        ans_to_ix, ix_to_ans = create_ann_vocal(anns,args)
        json.dump([ans_to_ix, ix_to_ans], open(annotation_path, 'w'))
    return ans_to_ix, ix_to_ans



def rnn_proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break
    return ques_ix

def get_score(occur):
    if occur == 0:
        return 0.0
    elif occur == 1:
        return 0.3
    elif occur == 2:
        return 0.7
    else:
        return 1


def vqa_proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}

    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])
    return ans_score

def simpsonsvqa_proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)

    ans = prep_ans(ans)
    if ans not in ans_to_ix:
        ans_score[0] = 1
        return ans_score
    else:
        ans_score[ans_to_ix[ans]] = 1
        return ans_score


def process_ac_answers(records):
    """
    Process judgements to determine the label for an annotation.

    Args:
        records (dict): A dictionary of judgements for an annotation.

    Returns:
        torch.Tensor: A tensor representing the label based on the counts of answers.

    The function counts the number of judgements with 'answer' as 1 and 0.
    Based on these counts, it returns a label tensor. If there are at least 2 counts of 1,
    it returns [0, 0, 1]; if there are at least 2 counts of 0, it returns [1, 0, 0].
    Otherwise, it returns [0, 1, 0].
    """
    count_1 = 0
    count_0 = 0

    # Count the number of 1s and 0s in the judgements
    for record in records.values():
        if record['answer'] == 1:
            count_1 += 1
        elif record['answer'] == 0:
            count_0 += 1

    # Determine the label based on the counts
    if count_1 >= 2:
        return torch.tensor([0, 0, 1], dtype=torch.float32) 
    elif count_0 >= 2:
        return torch.tensor([1, 0, 0], dtype=torch.float32) 
    else:
        return torch.tensor([0, 1, 0], dtype=torch.float32) 