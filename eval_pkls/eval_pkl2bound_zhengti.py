import pickle
import sys
sys.path.append("../..")
from VCUG_retrain.utils.roc import bootstrap_confidence_interval


pickle_path = "/home/tanzl/code/VCUG_retrain/eval_pkls/In_eval.pkl"
pickle_path_double = "/home/tanzl/code/VCUG_retrain/eval_pkls/In_DoubleBranch_eval.pkl"
with open(pickle_path, 'rb') as file:
    data = pickle.load(file)
with open(pickle_path_double, 'rb') as file:
    data_double = pickle.load(file)
print(pickle_path)

bounds = dict()
for k,v in data.items():
    print("-----------------------------------------{}------------------------------------------".format(k))
    results, fprs, tprs, label_preds_prob = v
    labels_single, preds_single, preds_probability_single = label_preds_prob
    
    if "VCUG" in k:
        results, fprs, tprs, label_preds_prob = data_double[k]
        labels_double, preds_double, preds_probability_double = label_preds_prob
    else:
        results, fprs, tprs, label_preds_prob = data_double[k+"_2branch"]
        labels_double, preds_double, preds_probability_double = label_preds_prob
    
    labels = labels_single + labels_double
    preds = preds_single + preds_double
    preds_probability = preds_probability_single.tolist() + preds_probability_double.tolist()
    bound = bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)
    bounds[k] = bound


pkl_name = "In_zongti_eval.pkl"
with open('/home/tanzl/code/VCUG_retrain/eval_pkls/result_bound/bound_{}'.format(pkl_name), 'wb') as file:
    pickle.dump(bounds, file)









# import pickle
# import sys
# sys.path.append("../..")
# from VCUG_retrain.utils.roc import bootstrap_confidence_interval


# pkl_name = 'In_train.pkl'

# pickle_path = "/home/tanzl/code/VCUG_retrain/{}".format(pkl_name)
# with open(pickle_path, 'rb') as file:
#     data = pickle.load(file)

# print(pickle_path)
# bounds = dict()
# for k,v in data.items():
    
#     results, fprs, tprs, label_preds_prob = v
#     labels, preds, preds_probability = label_preds_prob
#     bound = bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)
#     bounds[k] = bound