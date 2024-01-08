from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
def calculate_vqa_accuracy(result_data):
    """
    Calculate the accuracy of VQA predictions based on large_question_type.

    Parameters:
    result_data (pd.result_dataFrame): result_dataFrame containing columns 'question_type', 'vqa', 
                         'question_type_label', 'vqa_label', and 'large_question_type'.

    Returns:
    dict: A dictionary with 'large_question_type' as keys and accuracy as values.
    """
    
    
    
    
    accuracies = {}
    for lqt in result_data['answer_type'].unique():
        df_lqt = result_data[result_data['answer_type'] == lqt]
        correct_predictions = ((df_lqt['prediction'] == df_lqt['target'])).sum()
        total_instances = len(df_lqt)
        accuracies[lqt] = correct_predictions / total_instances if total_instances > 0 else 0
    
    # Preparing to save accuracies with specified key names
    
    formatted_accuracies = {
        f'{lqt.lower().replace("/","_")}_accuracy': accuracy for lqt, accuracy in accuracies.items()
    }
    overall_correct_predictions = ((result_data['question_type'] == result_data['question_type_label']) & 
                                   (result_data['vqa'] == result_data['vqa_label'])).sum()
    total_instances = len(result_data)
    vqa_accuracy = overall_correct_predictions / total_instances if total_instances > 0 else 0
    formatted_accuracies.update({"accuracy_vqa":vqa_accuracy})
    return formatted_accuracies

def calculate_accuracies(df, dataset):
    print(df)
    df["prediction class"] = df["prediction"].apply(lambda x: dataset.ix_to_ans[str(x)])
    df["target class"] = df["target"].apply(lambda x: dataset.ix_to_ans[str(x)])
    
    df["answer_type"] = df["id"].apply(lambda x: dataset.idx_to_ann[x]["answer_type"])
    
    accuracy_vqa = calculate_vqa_accuracy(df)
    return accuracy_vqa