from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import pandas as pd
from sklearn.metrics import classification_report
def calculate_vqa_accuracy(result_data):
    
    ###########################################################
    accuracies = {}
    for lqt in result_data['processed_answer_type'].unique():
        df_lqt = result_data[result_data['processed_answer_type'] == lqt]
        correct_predictions = ((df_lqt['prediction'] == df_lqt['target'])).sum()
        total_instances = len(df_lqt)
        accuracies[lqt] = correct_predictions / total_instances if total_instances > 0 else 0
    
    
    formatted_accuracies = {
        f'val_{lqt.lower().replace("/","_")}_accuracy(vqa-w-unans)': accuracy for lqt, accuracy in accuracies.items()
    }
    ###########################################################
    accuracies = {}
    for lqt in result_data['answer_type'].unique():
        df_lqt = result_data[result_data['answer_type'] == lqt]
        correct_predictions = ((df_lqt['prediction'] == df_lqt['target'])).sum()
        total_instances = len(df_lqt)
        accuracies[lqt] = correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies_1 = {
        f'val_{lqt.lower().replace("/","_")}_accuracy(vqa-w-unans, origin)': accuracy for lqt, accuracy in accuracies.items()
    }
    formatted_accuracies.update(formatted_accuracies_1)
    ###########################################################
    
    overall_correct_predictions = ((result_data['prediction'] == result_data['target'])).sum()
    total_instances = len(result_data)
    vqa_accuracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"val_accuracy_vqa(vqa-w-unans)":vqa_accuracy})
    ###########################################################
    
    report_dict = classification_report(result_data["birary answerable prediction"], result_data["birary answerable tartget"], output_dict=True)
    unanswerable_resuslt = {
        "val_answerable_recall": report_dict["answerable"]["recall"],
        "val_unanswerable_recall": report_dict["unanswerable"]["recall"],
        "val_answerable_precision": report_dict["answerable"]["precision"],
        "val_unanswerable_precision": report_dict["unanswerable"]["precision"]
    }
    formatted_accuracies.update(unanswerable_resuslt)
    ###########################################################
    
    result_data = result_data[result_data["processed_answer_type"] !="unanswerable"]
    accuracies = {}
    for lqt in result_data['answer_type'].unique():
        df_lqt = result_data[result_data['answer_type'] == lqt]
        correct_predictions = ((df_lqt['prediction'] == df_lqt['target'])).sum()
        total_instances = len(df_lqt)
        accuracies[lqt] = correct_predictions / total_instances if total_instances > 0 else 0
    
    formatted_accuracies_2 = {
        f'val_{lqt.lower().replace("/","_")}_accuracy(vqa-wo-unans)': accuracy for lqt, accuracy in accuracies.items()
    }
    formatted_accuracies.update(formatted_accuracies_2)
    ###########################################################
    
    overall_correct_predictions = ((result_data['prediction'] == result_data['target'])).sum()
    total_instances = len(result_data)
    vqa_accuracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"val_accuracy_vqa(vqa-wo-unans)":vqa_accuracy})

        
    return formatted_accuracies

def calculate_accuracies(df, dataset):
    with open("./dataset/super_answer_type_simpsons.json", 'r') as file:
        super_type = json.load(file)
    df["prediction class"] = df["prediction"].apply(lambda x: dataset.ix_to_ans[str(x)])
    df["target class"] = df["target"].apply(lambda x: dataset.ix_to_ans[str(x)])
    df["answer_type"] = df["id"].apply(lambda x: dataset.idx_to_ann[x]["answer_type"])
    df["processed_answer_type"] = df["id"].apply(lambda x: dataset.idx_to_ann[x]["processed_answer_type"])
    
    df["small_answer_type_target"] = df["target class"].apply(lambda x: super_type[x])
    df["small_answer_type_prediction"] = df["prediction class"].apply(lambda x: super_type[x])
    
    df["birary answerable prediction"] = df["prediction class"].apply(lambda x: "unanswerable" if x == "unanswerable" else "answerable")
    df["birary answerable tartget"] = df["target class"].apply(lambda x: "unanswerable" if x == "unanswerable" else "answerable")

    accuracy_vqa = calculate_vqa_accuracy(df)
    return accuracy_vqa, df