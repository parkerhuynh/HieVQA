from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import pandas as pd
from sklearn.metrics import classification_report
def calculate_vqa_accuracy(result_data):
    
    ###########################################################
    accuracies = {}
    for lqt in result_data['answer_type'].unique():
        df_lqt = result_data[result_data['answer_type'] == lqt]
        correct_predictions = ((df_lqt['prediction class'] == df_lqt['target class'])).sum()
        total_instances = len(df_lqt)
        accuracies[lqt] = correct_predictions / total_instances if total_instances > 0 else 0
    
    formatted_accuracies = {
        f'val_{lqt.lower().replace("/","_")}_accuracy(vqa-wo-unans)': accuracy for lqt, accuracy in accuracies.items()
    }
    # formatted_accuracies.update(formatted_accuracies_2)
    ###########################################################
    
    overall_correct_predictions = ((result_data['prediction class'] == result_data['target class'])).sum()
    total_instances = len(result_data)
    vqa_accuracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"val_accuracy_vqa(vqa-wo-unans)":vqa_accuracy})

    overall_correct_predictions = ((result_data['small_answer_type_target'] == result_data['small_answer_type_prediction'])).sum()
    total_instances = len(result_data)
    small_qt_accracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"small_qt_accracy(vqa-wo-unans)":small_qt_accracy})
    ###########################################################

    overall_correct_predictions = ((result_data['small_answer_type_target'] == result_data['small_answer_type_prediction'])).sum()
    total_instances = len(result_data)
    small_qt_accracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"small_qt_accracy(vqa-wo-unans)":small_qt_accracy})
    ###########################################################
    
    overall_correct_predictions = ((result_data['answer_type'] == result_data['answer_type_prediction'])).sum()
    total_instances = len(result_data)
    small_qt_accracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"large_qt_accracy(vqa-wo-unans, originn)":small_qt_accracy})

    overall_correct_predictions = ((result_data['processed_answer_type'] == result_data['processed_answer_type_prediction'])).sum()
    total_instances = len(result_data)
    small_qt_accracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"large_qt_accracy(vqa-wo-unans)":small_qt_accracy})

        
    return formatted_accuracies


def calculate_accuracies(df, dataset):
    
    df["small_answer_type_target"] = df["answer_type"].map(dataset.idx_to_ans_type)
    df["small_answer_type_prediction"] = df["answer_type_prediction"].map(dataset.idx_to_ans_type)
    #########################################################################
    
    df["target class"] = df[["small_answer_type_target", "target"]].apply(lambda x: convert_process(x["small_answer_type_target"], x["target"], dataset), axis =1)
    df["prediction class"] = df[["small_answer_type_prediction", "prediction"]].apply(lambda x: convert_process(x["small_answer_type_prediction"], x["prediction"], dataset),  axis =1)
    
    
    df["binary answerable prediction"] = df["prediction class"].apply(lambda x: "unanswerable" if x == "unanswerable" else "answerable")
    df["binary answerable target"] = df["target class"].apply(lambda x: "unanswerable" if x == "unanswerable" else "answerable")
    
    df["answer_type"] =  df["small_answer_type_target"].apply(lambda x: x if x in ["yes/no", "number"] else "other")
    df["answer_type_prediction"] =  df["small_answer_type_prediction"].apply(lambda x: x if x in ["yes/no", "number"] else "other")
    
    df["processed_answer_type"] =  df["small_answer_type_target"].apply(lambda x: x if x in ["yes/no", "number", "unanswerable"] else "other")
    df["processed_answer_type_prediction"] =  df["small_answer_type_prediction"].apply(lambda x: x if x in ["yes/no", "number", "unanswerable"] else "other")
    
    df["processed_answer_type (special)"] = df["processed_answer_type"]
    df["processed_answer_type_prediction (special)"] = df["processed_answer_type_prediction"]
    df["processed_answer_type"] = df[["processed_answer_type", "target class"]].apply(lambda x: process_small_type_answer(x['processed_answer_type'], x['target class']), axis = 1)
    df["processed_answer_type_prediction"] = df[["processed_answer_type_prediction", "target class"]].apply(lambda x: process_small_type_answer(x['processed_answer_type_prediction'], x['target class']), axis = 1)

    accuracy_vqa = calculate_vqa_accuracy(df)
    return accuracy_vqa, df

def convert_process(answer_type, ans_idx, dataset):
    idx_to_ans = dataset.ix_to_ans[answer_type]
    answer = idx_to_ans[str(ans_idx)]
    return answer

def process_small_type_answer(answer_type_prediction, target_class):
    if target_class == "unanswerable":
        return "unanswerable"
    else:
        return answer_type_prediction