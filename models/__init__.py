from models.VQA import VQA
from models.SAN import SAN
from models.MLB_MUTAN_Att import MLBAtt, MutanAtt
from models.MLB_MUTAN_NoAtt import MLBNoAtt, MutanNoAtt
from models.SANHieVQA import SANHieVQA
from models.VQAHieVQA import VQAHieVQA


def get_model(args, train_dataset):
    if args.model == "VQA":
        print("CREATING VQA MODEL")
        model = VQA(
            args = args,
            question_vocab_size = train_dataset.token_size,
            ans_vocab_size = train_dataset.ans_size)
        
    elif args.model == "SAN":
        print("CREATING SAN MODEL")
        model = SAN(
            question_vocab_size = train_dataset.token_size,
            ans_vocab_size = train_dataset.ans_size, 
            args = args)
        
    elif args.model == "MutanAtt":
        print("CREATING MUTAN MODEL WITH ATTENTION")
        model = MutanAtt(
            question_vocab_size = train_dataset.token_size,
            ans_vocab_size = train_dataset.ans_size, 
            args = args)
        
    elif args.model == "MutanNoAtt":
        print("CREATING MUTAN WITHOUT ATTENTION")
        model = MutanNoAtt(
            question_vocab_size = train_dataset.token_size,
            ans_vocab_size = train_dataset.ans_size, 
            opt = args)
        
    elif args.model == "MLBAtt":
        print("CREATING MLB MODEL WITH ATTENTION")
        model = MLBAtt(
            question_vocab_size = train_dataset.token_size,
            ans_vocab_size = train_dataset.ans_size, 
            opt = args.model_config)
        
    elif args.model == "MLBNoAtt":
        print("CREATING MLB MODEL")
        model = MLBNoAtt(
            question_vocab_size = train_dataset.token_size,
            ans_vocab_size = train_dataset.ans_size, 
            opt = args)
    elif args.model == "SAN-HieVQA":
        print("CREATING HieVQA")
        model = SANHieVQA(
            question_vocab_size = train_dataset.token_size,
            ans_vocab_type_dict = train_dataset.ans_to_ix,
            idx_to_answer_type = train_dataset.idx_to_answer_type, 
            args = args)
    elif args.model == "VQA" and "hie" in args.task:
        print("CREATING HieVQA")
        model = VQAHieVQA(
            question_vocab_size = train_dataset.token_size,
            ans_vocab_type_dict = train_dataset.ans_to_ix,
            idx_to_answer_type = train_dataset.idx_to_answer_type, 
            args = args)
    return model
