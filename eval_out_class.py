import torch
import torch.nn as nn
from models.getmodels import get_model_2branch_trained_631, get_model_trained_631
from datasets import load_data_vcug_out

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import label_binarize
import numpy as np

from itertools import combinations
from utils.roc import analyse_cm_everyClass, analyse_cm, t_roc_everyClass_threshold_softvote, t_roc_threshold_softvote



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle


def ele(model_names):
    model_names_single = model_names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    batch_size = 1
    vcug_dir_single = '/home/tanzl/data/VCUG/out_new/all/singleBranch/'
    vcug_dir_double = '/home/tanzl/data/VCUG/out_new/all/doubleBranch/'

    validate_loader_single, validate_loader_double = load_data_vcug_out(vcug_dir_single, vcug_dir_double, batch_size, label_model='label.txt')


    vote_results = []
    labels = []
    vote_results_single = []
    labels_single = []
    file_paths_single = []


    # 单分支
    results = []
    num_classes = 6

    for model_name in model_names_single:
        print("正在验证：{}".format(model_name))
        net = get_model_trained_631(model_name=model_name, num_classes=num_classes, model_weight_path_root="/home/tanzl/code/VCUG_retrain/result0725_single_all/")
        result = []
        net.to(device)
        net.eval()
        with torch.no_grad():
            for val_data in validate_loader_single:
                images, labels, file_paths  = val_data  # 左右分支
                outputs = net(images.to(device))
                outputs = torch.exp(outputs)/torch.sum(torch.exp(outputs))
                result.append(outputs[0].tolist())
        results.append(result)

    # labels_single
    for val_data in validate_loader_single:
        images, lb, file_paths  = val_data  # 左右分支
        labels_single.append(lb.item())
        file_paths_single.append(file_paths[0])

    results_softvote_probability_single = np.mean(results, axis=0)
    vote_results_single = np.argmax(results_softvote_probability_single, axis=1).tolist()




    vote_results = vote_results_single
    labels = labels_single
    results_softvote_probability = np.array(results_softvote_probability_single.tolist())

    cm = np.zeros((num_classes, num_classes))
    for i in range(len(vote_results)):
        cm[labels[i]][vote_results[i]] += 1



    # -----------------------------------刻画roc曲线，计算AUC-----------------------------------------
    labels_onehot = label_binarize(labels, classes=[i for i in range(num_classes)])
    fpr, tpr, auc = t_roc_everyClass_threshold_softvote(results_probability=results_softvote_probability,labels_onehot=labels_onehot)
    acces, recalles, precisiones, f1es = analyse_cm_everyClass(cm,talkative=True)
    return  acces, precisiones, recalles, f1es, fpr, tpr, auc, [labels, vote_results, results_softvote_probability]


model_name_all = ["googlenet", "densenet161", "efficientnet_b0","resnet101","shufflenet_v2_x1_0",  "RegNetY_400MF",  "efficientnetv2_l", 'vgg16',"mobildenet_v2"]


count_i = 0
names = []
fpr_es = []
tpr_es = []
auc_es = []
acc_es = []
recall_es = []
precision_es = []
f1_es = []
detailes = []



c = list(combinations(model_name_all,len(model_name_all)))
for model_names in c:
    names.append(model_names)
    count_i = count_i +1

c = list(combinations(model_name_all,1))
for model_names in c:
    names.append(model_names)
    count_i = count_i +1




for model_names in names:
    print("正在验证：-----------------------------{}--------------------------".format(model_names))
    acc, precision, recall, f1, fpr, tpr, auc, detailed_data = ele(model_names=model_names)

    fpr_es.append(fpr)
    tpr_es.append(tpr)
    auc_es.append(auc)
    acc_es.append(acc)
    recall_es.append(recall)
    precision_es.append(precision)
    f1_es.append(f1)
    detailes.append(detailed_data)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
names[0] = ('VCUG',)

import pickle
data_pkl_test = dict()
for i in range(count_i):
    data_pkl_test[names[i][0]] = [acc_es[i], precision_es[i], recall_es[i], f1_es[i], fpr_es[i], tpr_es[i], auc_es[i], detailes[i]]

with open('Out_Class.pkl', 'wb') as file:
    pickle.dump(data_pkl_test, file)
