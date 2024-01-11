from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import pandas as pd

def calculate_vqa_accuracy(result_data):
    
    
    accuracies = {}
    for lqt in result_data['answer_type'].unique():
        df_lqt = result_data[result_data['answer_type'] == lqt]
        correct_predictions = ((df_lqt['prediction'] == df_lqt['target'])).sum()
        total_instances = len(df_lqt)
        accuracies[lqt] = correct_predictions / total_instances if total_instances > 0 else 0
    
    # Preparing to save accuracies with specified key names
    
    formatted_accuracies = {
        f'val_{lqt.lower().replace("/","_")}_accuracy(vqa-wo-unans)': accuracy for lqt, accuracy in accuracies.items()
    }
    overall_correct_predictions = ((result_data['prediction'] == result_data['target'])).sum()
    total_instances = len(result_data)
    vqa_accuracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"val_accuracy_vqa(vqa-wo-unans)":vqa_accuracy})
    return formatted_accuracies

def calculate_accuracies(df, dataset):
    with open("./dataset/super_answer_type_simpsons.json", 'r') as file:
        super_type = json.load(file)
        
        
    df["answer_type"] = df["answer_type"].map(dataset.idx_to_ans_type)
    df["answer_type_prediction"] = df["answer_type_prediction"].map(dataset.idx_to_ans_type)
    
    df["target class"] = df[["answer_type", "target"]].apply(lambda x: convert_process(x["answer_type"], x["target"], dataset), axis =1)
    df["prediction class"] = df[["answer_type_prediction", "prediction"]].apply(lambda x: convert_process(x["answer_type_prediction"], x["prediction"], dataset),  axis =1)
    
    # df["prediction class"] = df["prediction"].apply(lambda x: dataset.ix_to_ans[str(x)])
    # df["target class"] = df["target"].apply(lambda x: dataset.ix_to_ans[str(x)])
    # df["answer_type"] = df["id"].apply(lambda x: dataset.idx_to_ann[x]["answer_type"])
    
    df["small_answer_type_target"] = df["target class"].apply(lambda x: super_type[x])
    df["small_answer_type_prediction"] = df["prediction class"].apply(lambda x: super_type[x])
    print(df)
    accuracy_vqa = calculate_vqa_accuracy(df)
    return accuracy_vqa, df

def convert_process(answer_type, ans_idx, dataset):
    idx_to_ans = dataset.ix_to_ans[answer_type]
    answer = idx_to_ans[str(ans_idx)]
    return answer