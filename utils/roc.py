# 引入必要的库
from types import new_class
import numpy as np
np.random.seed(2022)
# import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import copy


def load_thresholds(dir):
    # 从文件中读取字符串并将其转换回列表
    with open(dir, 'r') as f:
        loaded_str = f.read()
        loaded_list = eval(loaded_str)
    return loaded_list



import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score


def compute_metric(datanpGT, datanpPRED, target_names):
    n_class = len(target_names)
    argmaxPRED = np.argmax(datanpPRED, axis=1)
    F1_metric = np.zeros([n_class, 1])
    tn = np.zeros([n_class, 1])
    fp = np.zeros([n_class, 1])
    fn = np.zeros([n_class, 1])
    tp = np.zeros([n_class, 1])

    Accuracy_score = accuracy_score(datanpGT, argmaxPRED)
    ROC_curve = {}
    mAUC = 0

    for i in range(n_class):
        tmp_label = datanpGT == i
        tmp_pred = argmaxPRED == i
        F1_metric[i] = f1_score(tmp_label, tmp_pred)
        tn[i], fp[i], fn[i], tp[i] = confusion_matrix(tmp_label, tmp_pred).ravel()
        outAUROC = roc_auc_score(tmp_label, datanpPRED[:, i])

        mAUC = mAUC + outAUROC
        [roc_fpr, roc_tpr, roc_thresholds] = roc_curve(tmp_label, datanpPRED[:, i])

        ROC_curve.update({'ROC_fpr_'+str(i): roc_fpr,
                          'ROC_tpr_' + str(i): roc_tpr,
                          'ROC_T_' + str(i): roc_thresholds,
                          'AUC_' + str(i): outAUROC})

    mPrecision = sum(tp) / sum(tp + fp)
    mRecall = sum(tp) / sum(tp + fn)
    output = {
        'class_name': target_names,
        'F1': F1_metric,
        'AUC': mAUC / n_class,
        'Accuracy': Accuracy_score,

        'Sensitivity': tp / (tp + fn),
        'Precision': tp / (tp + fp),
        'Specificity': tn / (fp + tn),
        'ROC_curve': ROC_curve,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,

        'micro-Precision': mPrecision,
        'micro-Sensitivity': mRecall,
        'micro-Specificity': sum(tn) / sum(fp + tn),
        'micro-F1': 2*mPrecision * mRecall / (mPrecision + mRecall),
    }

    # return output
    return ROC_curve['ROC_fpr_4'], ROC_curve['ROC_tpr_4'], ROC_curve['AUC_4']


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


def t_roc_threshold_softvote(results_probability,labels_onehot,thresholds):
    fprs = []
    tprs = []
    for th in thresholds:        
        results_probability_temp = copy.deepcopy(results_probability)
        results_probability_temp = np.where(results_probability_temp > th, 1, 0)
        fpr, tpr, _ = roc_curve(labels_onehot.ravel(), results_probability_temp.ravel())
        fprs.append(fpr[1])
        tprs.append(tpr[1])
    quicksort(arr1=fprs, arr2=tprs, low=0, high=len(fprs) - 1)
    roc_auc = auc(fprs, tprs)
    return fprs, tprs, roc_auc

def t_prc_threshold_softvote(results_probability,labels_onehot,thresholds):
    fprs = []
    tprs = []
    for th in thresholds:        
        results_probability_temp = copy.deepcopy(results_probability)
        results_probability_temp = np.where(results_probability_temp > th, 1, 0)
        fpr, tpr, _ = precision_recall_curve(labels_onehot.ravel(), results_probability_temp.ravel())
 
        fprs.append(fpr[1])
        tprs.append(tpr[1])

    quicksort(arr1=fprs, arr2=tprs, low=0, high=len(fprs) - 1)

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


# labels_onehot = label_binarize(labels, classes=[i for i in range(num_classes)])
# fpr, tpr, auc = t_roc(y_test= labels_onehot, y_score=vote_results_onehot)

def t_roc(y_test, y_score):
    # y_test是一个多类别的概率，形状为M×N， M：样本数目，N：类别数, 是对标签进行onehot编码后的结果
    # y_score(numpy) 是一个多类别的概率，形状为M×N， M：样本数目，N：类别数
    _, n_classes = y_score.shape

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # # Compute macro-average ROC curve and ROC area（方法一）
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr

    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # lw=2
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
    #             ''.format(roc_auc["micro"]),
    #         color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #         label='macro-average ROC curve (area = {0:0.2f})'
    #             ''.format(roc_auc["macro"]),
    #         color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(i, roc_auc[i]))



    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()


    return fpr["micro"], tpr["micro"], roc_auc["micro"]

# def t_roc_curve(fpr, tpr):
#     roc_auc = auc(fpr, tpr)
#     plt.figure()
#     plt.plot(fpr, tpr,
#             label='micro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc),
#             color='deeppink', linestyle=':', linewidth=4)

#     lw = 2
#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC曲线')
#     plt.legend(loc="lower right")
#     plt.show()

#     return roc_auc

def t_auc(fpr, tpr):
    return auc(fpr, tpr)


def t_roc_everyClass(y_test, y_score):
    # y_test是一个多类别的概率，形状为M×N， M：样本数目，N：类别数, 是对标签进行onehot编码后的结果
    # y_score(numpy) 是一个多类别的概率，形状为M×N， M：样本数目，N：类别数
    _, n_classes = y_score.shape

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        if len(fpr[i])==2:
            fpr[i]=np.array([0.,0.5,1.0])
            tpr[i]=np.array([0.,0.5,1.0])

    return fpr, tpr, roc_auc


def t_roc_everyClass_threshold_softvote(results_probability,labels_onehot):
    num_classes = 6
    fprs = []
    tprs = []
    aucs = []
    for i in range(num_classes):
        temp_results_probability = results_probability[:, i]
        temp_labels = labels_onehot[:, i]
        fpr, tpr, auc = t_roc_threshold_softvote(temp_results_probability,temp_labels,temp_results_probability.ravel().tolist()+[0])
        if len(fpr)==2:
            print("len(fpr)==2")
            print(fpr)
            print(tpr)
            fpr=np.array([0.,0.5,1.0])
            tpr=np.array([0.,0.5,1.0])
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc)
    return fprs, tprs, aucs





def analyse_cm(cm, talkative=False):
    cm_num = [sum(cm[i]) for i in range(len(cm))]
    cm_ratio = [num / sum(cm_num) for num in cm_num]

    acc = cm.trace() / cm.sum()

    Recalles = []
    for i in range(len(cm)):
        if sum(cm[i]) == 0:
            Recalles.append(0)
        else:
            Recalles.append(cm[i][i] / sum(cm[i]))

    average_recall = 0.0
    for i in range(len(cm)):
        average_recall += cm_ratio[i] * Recalles[i]

    Precisions = []
    for i in range(len(cm)):
        if (sum(cm[i]) == 0) or (sum(cm[:, i]) == 0):
            Precisions.append(0)
        else:
            Precisions.append(cm[i][i] / sum(cm[:, i]))

    average_precision = 0.0
    for i in range(len(cm)):
        average_precision += cm_ratio[i] * Precisions[i]

    F1_macros = []
    for i in range(len(cm)):
        if (sum(cm[i]) == 0) or ((Recalles[i] + Precisions[i]) == 0):
            F1_macros.append(0)
        else:
            F1_macros.append(2 * Recalles[i] * Precisions[i] / (Recalles[i] + Precisions[i]))

    average_f1_macro = 0.0
    for i in range(len(cm)):
        average_f1_macro += cm_ratio[i] * F1_macros[i]



    return acc, average_precision, average_recall, average_f1_macro


def analyse_cm_everyClass(cm, talkative=False):
    num_class = len(cm)
    acces = []
    recalles = []
    precisiones = []
    f1es = []
    for i in range(num_class):
        TP = cm[i][i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN
        acc = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f1 = 2*recall*precision/(recall+precision) if (recall + precision) != 0 else 0
        acces.append(acc)
        recalles.append(recall)
        precisiones.append(precision)
        f1es.append(f1)

    return acces, recalles, precisiones, f1es


def confidence_interveal_scores(scores, k=5):
    mean = np.mean(scores)
    std = np.std(scores)
    ci = (mean - 1.96 * std / np.sqrt(k), mean + 1.96 * std / np.sqrt(k))
    return ci


# def plot_roc_curves(fpr, tpr, roc_auc, names, bounds=None):
#     # 根据roc_auc值进行排序
#     if ['Bilateral'] in names:
#         sorted_indices = [i for i in range(len(names))]
#     if names==[['ALL'], ['Qingdao'], ['PUyang'], ['Guangxi'], ['Anhui']]:
#         sorted_indices = [0,3,2,4,1]
#     else:
#         sorted_indices = sorted(range(len(roc_auc)), key=lambda i: roc_auc[i], reverse=True)
    
#     # # 绘制ROC曲线
#     # lw = 2
#     # for i in sorted_indices:
#     #     plt.plot(fpr[i], tpr[i], lw=lw,
#     #              label='ROC curve of {0} (area = {1:0.3f})'.format(names[i][0], roc_auc[i]))
    

#     plt.figure(figsize=(8, 8))
#     # 绘制ROC曲线
#     lw = 2
#     for i in sorted_indices:
#         plt.plot(fpr[i], tpr[i], lw=lw,
#                 #  label='{0:<{width}},AUC: {1:0.3f}\n95%CI:({2:0.3f},{3:0.3f}) '.format(names[i][0], roc_auc[i], bounds[i][0], bounds[i][1], width=17))
#                  label='{0}, AUC: {1:0.3f}\n95%CI:({2:0.3f},{3:0.3f}) '.format(names[i][0], roc_auc[i], bounds[i][0], bounds[i][1]), color = color_dict[names[i][0]])
    
#     for i in sorted_indices:
#         if ("VCUG" in names[i][0]) or ("ALL" in names[i][0]):
#             plt.plot(fpr[i], tpr[i], lw=lw,color = color_dict[names[i][0]])

#     # 添加随机猜测的虚线
    
#     plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    
#     # 设置坐标轴范围、标签和标题
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
    
#     # 添加图例并显示图形
#     plt.legend(loc="lower right", fontsize=10)
#     plt.show()

# def plot_prc_curves(fpr, tpr, roc_auc, names):
#     # 根据roc_auc值进行排序
#     sorted_indices = sorted(range(len(roc_auc)), key=lambda i: roc_auc[i], reverse=True)
    
#     # 绘制ROC曲线
#     lw = 2
#     for i in sorted_indices:
#         plt.plot(fpr[i], tpr[i], lw=lw,
#                  label='ROC curve of {0} (area = {1:0.2f})'.format(names[i][0], roc_auc[i]))
    
#     # 添加随机猜测的虚线
#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    
#     # 设置坐标轴范围、标签和标题
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('PR Curve')
    
#     # 添加图例并显示图形
#     plt.legend(loc="lower right")
#     plt.show()




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
        if verbose and _%20==0:
            print("正在进行第{}次采样".format(_))
        indices = np.random.choice(n, size=n, replace=True)
        sample_preds = preds[indices]
        sample_labels = labels[indices]
        sample_preds_probability = preds_probability[indices]

        cm = np.zeros((num_classes, num_classes))
        for i in range(len(sample_preds)):
            cm[sample_labels[i]][sample_preds[i]] += 1

        # -----------------------------------刻画roc曲线，计算AUC-----------------------------------------
        labels_onehot = label_binarize(sample_labels, classes=[i for i in range(num_classes)])
        fpr, tpr, auc = t_roc_threshold_softvote(results_probability=sample_preds_probability,labels_onehot=labels_onehot,thresholds=sample_preds_probability.ravel().tolist()+[0])
        acc, prescision, recall, f1 = analyse_cm(cm,talkative=True)

    
        # 计算新样本的统计量，这里是样本均值
        stat_acc.append(acc)
        stat_precision.append(prescision)
        stat_recall.append(recall)
        stat_f1.append(f1)
        stat_auc.append(auc)

    # 对计算得到的统计量排序
    stat_acc.sort()
    stat_precision.sort()
    stat_recall.sort()
    stat_f1.sort()
    stat_auc.sort()
    
    # 确定置信区间边界的位置
    lower_bound_index = int(num_iterations * alpha / 2)
    upper_bound_index = int(num_iterations * (1 - alpha / 2))
    
    # 返回95%置信区间的上下限
    return [[stat_acc[lower_bound_index],stat_acc[upper_bound_index]],
            [stat_precision[lower_bound_index],stat_precision[upper_bound_index]],
            [stat_recall[lower_bound_index],stat_recall[upper_bound_index]],
            [stat_f1[lower_bound_index],stat_f1[upper_bound_index]],
            [stat_auc[lower_bound_index],stat_auc[upper_bound_index]]]




def bootstrap_confidence_interval_class(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=False):
    num_classes = 6
    n = len(preds)
    labels, preds, preds_probability = np.array(labels), np.array(preds), np.array(preds_probability)
    stat_acc = []
    stat_precision = []
    stat_recall = []
    stat_f1 = []
    stat_auc = []
    
    for _ in range(num_iterations):
        if verbose:
            print("正在进行第{}次采样".format(_))
        indices = np.random.choice(n, size=n, replace=True)
        sample_preds = preds[indices]
        sample_labels = labels[indices]
        sample_preds_probability = preds_probability[indices]

        cm = np.zeros((num_classes, num_classes))
        for i in range(len(sample_preds)):
            cm[sample_labels[i]][sample_preds[i]] += 1

        # -----------------------------------刻画roc曲线，计算AUC-----------------------------------------


        labels_onehot = label_binarize(sample_labels, classes=[i for i in range(num_classes)])
        fpr, tpr, auc = t_roc_everyClass_threshold_softvote(results_probability=sample_preds_probability,labels_onehot=labels_onehot)
        acces, recalles, precisiones, f1es = analyse_cm_everyClass(cm,talkative=True)


        # 计算新样本的统计量，这里是样本均值
        stat_acc.append(acces)
        stat_precision.append(precisiones)
        stat_recall.append(recalles)
        stat_f1.append(f1es)
        stat_auc.append(auc)


    stat_acc = np.array(stat_acc)
    stat_precision = np.array(stat_precision)
    stat_recall = np.array(stat_recall)
    stat_f1 = np.array(stat_f1)
    stat_auc = np.array(stat_auc)

    # 对计算得到的统计量排序
    stat_acc = np.sort(stat_acc,axis=0)
    stat_precision = np.sort(stat_precision,axis=0)
    stat_recall = np.sort(stat_recall,axis=0)
    stat_f1 = np.sort(stat_f1,axis=0)
    stat_auc = np.sort(stat_auc,axis=0)
    
    # 确定置信区间边界的位置
    lower_bound_index = int(num_iterations * alpha / 2)
    upper_bound_index = int(num_iterations * (1 - alpha / 2))
    
    # 返回95%置信区间的上下限
    return [[stat_acc[lower_bound_index].tolist(),stat_acc[upper_bound_index].tolist()],
            [stat_precision[lower_bound_index].tolist(),stat_precision[upper_bound_index].tolist()],
            [stat_recall[lower_bound_index].tolist(),stat_recall[upper_bound_index].tolist()],
            [stat_f1[lower_bound_index].tolist(),stat_f1[upper_bound_index].tolist()],
            [stat_auc[lower_bound_index].tolist(),stat_auc[upper_bound_index].tolist()]]




tong2academic = {
    "DeepVCUG":"DeepVCUG",
    "VCUG":"DeepVCUG",
    "googlenet":"GoogLeNet",
    "densenet161":"DenseNet-161",
    "efficientnet_b0":"EfficientNet-B0",
    "resnet101":"ResNet-101",
    "shufflenet_v2_x1_0":"ShuffleNet-V2",
    "RegNetY_400MF":"RegNetY-400MF",
    "efficientnetv2_l":"EfficientNetV2-L",
    "vgg16":"VGG-16",
    "mobildenet_v2":"MobileNet-V2",

    'googlenet_2branch': 'GoogLeNet-2branch',
    'densenet161_2branch': 'DenseNet-161-2branch',
    'efficientnet_b0_2branch': 'EfficientNet-B0-2branch',
    'resnet101_2branch': 'ResNet-101-2branch',
    'shufflenet_v2_x1_0_2branch': 'ShuffleNet-V2-2branch',
    'RegNetY_400MF_2branch': 'RegNetY-400MF-2branch',
    'efficientnetv2_l_2branch': 'EfficientNetV2-L-2branch',
    'vgg16_2branch': 'VGG-16-2branch',
    'mobildenet_v2_2branch': 'MobileNet-V2-2branch',

    'all':"ALL",
    'qingda':"Qingdao",
    'PUyang':"PUyang",
    'guangxi':"Guangxi",
    'anhui':"Anhui",

}



color_dict = {
    'DeepVCUG': 'red',
    'GoogLeNet': 'blue',
    'DenseNet-161': 'green',
    'EfficientNet-B0': 'purple',
    'ResNet-101': 'orange',
    'ShuffleNet-V2': 'cyan',
    'RegNetY-400MF': 'magenta',
    'EfficientNetV2-L': 'yellow',
    'VGG-16': 'brown',
    'MobileNet-V2': 'gray',

    'GoogLeNet-2branch': 'blue',
    'DenseNet-161-2branch': 'green',
    'EfficientNet-B0-2branch': 'purple',
    'ResNet-101-2branch': 'orange',
    'ShuffleNet-V2-2branch': 'cyan',
    'RegNetY-400MF-2branch': 'magenta',
    'EfficientNetV2-L-2branch': 'yellow',
    'VGG-16-2branch': 'brown',
    'MobileNet-V2-2branch': 'gray',

    'Unilateral-0': 'green',
    'Unilateral-1': 'purple',
    'Unilateral-2': 'orange',
    'Unilateral-3': 'cyan',
    'Unilateral-4': 'magenta',
    'Unilateral-5': 'blue',
    'Bilateral': 'brown',

    'ALL':"red",
    'Qingdao':"green",
    'PUyang':"orange",
    'Guangxi':"magenta",
    'Anhui':"blue",




}

