import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import copy

def t_roc_threshold(results_probability,results_onehot,labels_onehot,thresholds = np.linspace(0.99, 0, num=100).tolist()):
    fprs = [0.]
    tprs = [0.]
    for th in thresholds:
        th_zero = np.where(np.array(results_probability) > th, 1, 0)
        results_onehot_temp = copy.deepcopy(results_onehot)
        for i in range(len(th_zero)):
            results_onehot_temp[i] = results_onehot_temp[i]*th_zero[i]
            
        fpr, tpr, _ = roc_curve(labels_onehot.ravel(), results_onehot_temp.ravel())
        fprs.append(fpr[1])
        tprs.append(tpr[1])

    quicksort(arr1=fprs, arr2=tprs, low=0, high=len(fprs) - 1)
    fprs.append(1.)
    tprs.append(1.)


    roc_auc = auc(fprs, tprs)
    return fprs, tprs, roc_auc


def quicksort(arr1, arr2, low, high):
    if low < high:
        pivot_index = partition(arr1, arr2, low, high)
        quicksort(arr1, arr2, low, pivot_index - 1)
        quicksort(arr1, arr2, pivot_index + 1, high)

def partition(arr1, arr2, low, high):
    pivot = arr1[high]
    i = low - 1

    for j in range(low, high):
        if arr1[j] <= pivot:
            i += 1
            arr1[i], arr1[j] = arr1[j], arr1[i]
            arr2[i], arr2[j] = arr2[j], arr2[i]

    arr1[i + 1], arr1[high] = arr1[high], arr1[i + 1]
    arr2[i + 1], arr2[high] = arr2[high], arr2[i + 1]

    return i + 1




def analyse_cm(confusion_matrix, verbose=False):
    # Calculate the number of instances per class and the ratio for each class
    class_totals = confusion_matrix.sum(axis=1)
    class_ratios = class_totals / class_totals.sum()

    # Calculate overall accuracy
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()

    # Calculate recall for each class
    recalls = np.diag(confusion_matrix) / np.where(class_totals == 0, 1, class_totals)
    weighted_recall = np.average(recalls, weights=class_ratios)

    # Calculate precision for each class
    column_totals = confusion_matrix.sum(axis=0)
    precisions = np.diag(confusion_matrix) / np.where(column_totals == 0, 1, column_totals)
    weighted_precision = np.average(precisions, weights=class_ratios)

    # Calculate F1 score for each class
    f1_scores = 2 * (precisions * recalls) / np.where((precisions + recalls) == 0, 1, (precisions + recalls))
    weighted_f1_score = np.average(f1_scores, weights=class_ratios)

    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Weighted Precision: {weighted_precision}")
        print(f"Weighted Recall: {weighted_recall}")
        print(f"Weighted F1 Score: {weighted_f1_score}")

    return accuracy, weighted_precision, weighted_recall, weighted_f1_score


def confidence_interveal_scores(scores, k=5):
    mean = np.mean(scores)
    std = np.std(scores)
    ci = (mean - 1.96 * std / np.sqrt(k), mean + 1.96 * std / np.sqrt(k))
    return ci


def bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=False):
    num_classes = 6
    n = len(preds)
    labels, preds, preds_probability = np.array(labels), np.array(preds), np.array(preds_probability)
    stat_acc = []
    stat_precision = []
    stat_recall = []
    stat_f1 = []
    stat_auc = []
    
    for _ in range(num_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        sample_preds = preds[indices]
        sample_labels = labels[indices]
        sample_preds_probability = preds_probability[indices]

        cm = np.zeros((num_classes, num_classes))
        for i in range(len(sample_preds)):
            cm[sample_labels[i]][sample_preds[i]] += 1

        labels_onehot = label_binarize(sample_labels, classes=[i for i in range(num_classes)])
        fpr, tpr, auc = t_roc_threshold(results_probability=sample_preds_probability,labels_onehot=labels_onehot,thresholds=sample_preds_probability.ravel().tolist()+[0])
        acc, prescision, recall, f1 = analyse_cm(cm,talkative=True)
    
        stat_acc.append(acc)
        stat_precision.append(prescision)
        stat_recall.append(recall)
        stat_f1.append(f1)
        stat_auc.append(auc)

 
    stat_acc.sort()
    stat_precision.sort()
    stat_recall.sort()
    stat_f1.sort()
    stat_auc.sort()
    

    lower_bound_index = int(num_iterations * alpha / 2)
    upper_bound_index = int(num_iterations * (1 - alpha / 2))
    

    return [[stat_acc[lower_bound_index],stat_acc[upper_bound_index]],
            [stat_precision[lower_bound_index],stat_precision[upper_bound_index]],
            [stat_recall[lower_bound_index],stat_recall[upper_bound_index]],
            [stat_f1[lower_bound_index],stat_f1[upper_bound_index]],
            [stat_auc[lower_bound_index],stat_auc[upper_bound_index]]]