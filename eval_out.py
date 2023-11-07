import torch
from models.getmodels import get_model_2branch_trained_631, get_model_trained_631
from datasets import load_data_vcug_out

from sklearn.preprocessing import label_binarize
import numpy as np

from utils.roc import  analyse_cm, t_roc_threshold_softvote



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle




# def ele(hospital_name):
#     model_names_single = ["googlenet", "densenet161", "efficientnet_b0","resnet101","shufflenet_v2_x1_0",  "RegNetY_400MF",  "efficientnetv2_l", 'vgg16',"mobildenet_v2"]
#     model_names_double = ["vgg16_2branch","densenet161_2branch","resnet101_2branch","mobildenet_v2_2branch",
#                 "shufflenet_v2_x1_0_2branch","efficientnet_b0_2branch","RegNetY_400MF_2branch","efficientnetv2_l_2branch","googlenet_2branch"]


#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")

#     batch_size = 1
#     vcug_dir_single = '/home/tanzl/data/VCUG/out_new/{}/singleBranch/'.format(hospital_name)
#     vcug_dir_double = '/home/tanzl/data/VCUG/out_new/{}/doubleBranch/'.format(hospital_name)

#     validate_loader_single, validate_loader_double = load_data_vcug_out(vcug_dir_single, vcug_dir_double, batch_size, label_model='label.txt')


#     vote_results = []
#     labels = []
#     vote_results_single = []
#     labels_single = []
#     vote_results_double = []
#     labels_double = []
#     file_paths_single = []
#     file_paths_double = []


#     # 单分支
#     results = []
#     num_classes = 6

#     for model_name in model_names_single:
#         print("正在验证：{}".format(model_name))
#         net = get_model_trained_631(model_name=model_name, num_classes=num_classes, model_weight_path_root="/home/tanzl/code/VCUG_retrain/result0725_single_all/")
#         result = []
#         net.to(device)
#         net.eval()
#         with torch.no_grad():
#             for val_data in validate_loader_single:
#                 images, labels, file_paths  = val_data  # 左右分支
#                 outputs = net(images.to(device))
#                 outputs = torch.exp(outputs)/torch.sum(torch.exp(outputs))
#                 result.append(outputs[0].tolist())
#         results.append(result)

#     # labels_single
#     for val_data in validate_loader_single:
#         images, lb, file_paths  = val_data  # 左右分支
#         labels_single.append(lb.item())
#         file_paths_single.append(file_paths[0])

#     results_softvote_probability_single = np.mean(results, axis=0)
#     vote_results_single = np.argmax(results_softvote_probability_single, axis=1).tolist()


#     # vote_result_single
#     num_pic = len(vote_results_single)
#     for i in range(num_pic):
#         with open( file_paths_single[i] + '/assit_vcug.txt', 'w') as file:
#             file.write('单侧 :'+str(int(vote_results_single[i])))




#     # 双分支
#     results = []
#     results_l = []
#     results_r = []
#     for model_name in model_names_double:
#         print("正在验证：{}".format(model_name))
#         net = get_model_2branch_trained_631(model_name=model_name, model_weight_path_root="/home/tanzl/code/VCUG_retrain/result0725_double_all")

#         result_l = []
#         result_r = []
#         net.to(device)
#         net.eval()
#         with torch.no_grad():
#             for val_data in validate_loader_double:
#                 images, lb_l, lb_r,file_paths  = val_data  # 左右分支
#                 outputs_l, outputs_r = net(images.to(device))
#                 outputs_l = torch.exp(outputs_l)/torch.sum(torch.exp(outputs_l))
#                 outputs_r = torch.exp(outputs_r)/torch.sum(torch.exp(outputs_r))
#                 result_l.append(outputs_l[0].tolist())
#                 result_r.append(outputs_r[0].tolist())
#         result = result_l + result_r
        
#         results_r.append(result_r)
#         results_l.append(result_l)
#         results.append(result)
        

#     results_softvote_probability_double = np.mean(results, axis=0)
#     results_softvote_probability_l = np.mean(results_l, axis=0)
#     results_softvote_probability_r = np.mean(results_r, axis=0)

#     vote_results_double = np.argmax(results_softvote_probability_double, axis=1).tolist()
#     vote_results_l = np.argmax(results_softvote_probability_l, axis=1).tolist()
#     vote_results_r = np.argmax(results_softvote_probability_r, axis=1).tolist()


#     labels_l = []
#     labels_r = []
#     for val_data in validate_loader_double:
#         images, lb_l, lb_r,file_paths  = val_data  # 左右分支
#         labels_l.append(lb_l.item())
#         labels_r.append(lb_r.item())
#         file_paths_double.append(file_paths[0])


#     num_pic = len(vote_results_l)
#     for i in range(num_pic):
#         with open( file_paths_double[i] + '/assit_vcug.txt', 'w') as file:
#             file.write('左侧 :'+str(vote_results_l[i])+'\n'+'右侧 :'+str(vote_results_r[i]))  

#     labels_double = labels_l + labels_r
#     vote_results_double = vote_results_l + vote_results_r

#     vote_results = vote_results_single + vote_results_double
#     labels = labels_single + labels_double
#     results_softvote_probability = np.array(results_softvote_probability_single.tolist() + results_softvote_probability_double.tolist())

#     cm = np.zeros((num_classes, num_classes))
#     for i in range(len(vote_results)):
#         cm[labels[i]][vote_results[i]] += 1



#     # -----------------------------------刻画roc曲线，计算AUC-----------------------------------------
#     labels_onehot = label_binarize(labels, classes=[i for i in range(num_classes)])
#     fpr, tpr, auc = t_roc_threshold_softvote(results_probability=results_softvote_probability,labels_onehot=labels_onehot,thresholds=results_softvote_probability.ravel().tolist()+[0])
#     acc, average_prescision, average_recall, average_f1 = analyse_cm(cm,talkative=True)
#     return  acc, average_prescision, average_recall, average_f1, fpr, tpr, auc, [labels, vote_results, results_softvote_probability]




# hospital_names = ['all', 'qingda', 'PUyang', 'guangxi', 'anhui']

# # hospital_names = ['qingda']
# results = []
# detailes = []

# fpr = []
# tpr = []
# auc = []


# for hospital_name in hospital_names:
#     print("正在验证：-----------------------------{}--------------------------".format(hospital_name))
#     temp_acc, temp_prescision, temp_recall, temp_f1, temp_fpr, temp_tpr, temp_auc, detailed_data = ele(hospital_name=hospital_name)

#     fpr.append(temp_fpr)
#     tpr.append(temp_tpr)
#     auc.append(temp_auc)


#     print("机构：", hospital_name)
#     print("准确率：", temp_acc)
#     print("查准率：", temp_prescision)
#     print("召回率：", temp_recall)
#     print("f1：", temp_f1)
#     print("auc：", temp_auc)
#     results.append([temp_acc, temp_prescision, temp_recall, temp_f1, temp_auc])
#     detailes.append(detailed_data)
# # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# for i in range(len(hospital_names)):
#     print(hospital_names[i],results[i])

# data_pkl = dict()
# for i in range(len(hospital_names)):
#     data_pkl[hospital_names[i]] = [results[i], fpr[i], tpr[i], detailes[i]]

# with open('Out_zongti.pkl', 'wb') as file:
#     pickle.dump(data_pkl, file)





import pickle
import sys
sys.path.append("../..")
from VCUG_retrain.utils.roc import bootstrap_confidence_interval


pkl_name = 'Out_zongti.pkl'

pickle_path = "/home/tanzl/code/VCUG_retrain/{}".format(pkl_name)
with open(pickle_path, 'rb') as file:
    data = pickle.load(file)

print(pickle_path)
bounds = dict()
for k,v in data.items():
    print("-----------------------------------------{}------------------------------------------".format(k))
    results, fprs, tprs, label_preds_prob = v
    labels, preds, preds_probability = label_preds_prob
    bound = bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)
    bounds[k] = bound



with open('/home/tanzl/code/VCUG_retrain/eval_pkls/result_bound/bound_{}'.format(pkl_name), 'wb') as file:
    pickle.dump(bounds, file)