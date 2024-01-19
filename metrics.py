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
    
    overall_correct_predictions = ((result_data['prediction'] == result_data['target'])).sum()
    total_instances = len(result_data)
    vqa_accuracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    
    overall_correct_predictions = ((result_data['small_answer_type_target'] == result_data['small_answer_type_prediction'])).sum()
    total_instances = len(result_data)
    small_qt_accracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"small_qt_accracy(vqa-wo-unans)":small_qt_accracy})
    
    overall_correct_predictions = ((result_data['answer_type'] == result_data['answer_type_prediction'])).sum()
    total_instances = len(result_data)
    large_qt_accracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"large_qt_accracy(vqa-wo-unans)":large_qt_accracy})
    return formatted_accuracies

def calculate_accuracies(df, dataset):
    with open("./dataset/super_answer_type_simpsons.json", 'r') as file:
        super_type = json.load(file)
    df["answer_type"] = df["id"].apply(lambda x: dataset.idx_to_ann[x]["answer_type"])
    
    df["small_answer_type_target"] = df["target"].apply(lambda x: super_type[x])
    df["small_answer_type_prediction"] = df["prediction"].apply(lambda x: super_type[x])
    
    df["answer_type"] =  df["small_answer_type_target"].apply(lambda x: x if x in ["yes/no", "number"] else "other")
    df["answer_type_prediction"] =  df["small_answer_type_prediction"].apply(lambda x: x if x in ["yes/no", "number"] else "other")

    accuracy_vqa = calculate_vqa_accuracy(df)
    return accuracy_vqa, df