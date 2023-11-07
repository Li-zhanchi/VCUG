
import pickle
import sys
sys.path.append("../..")
from VCUG_retrain.utils.roc import bootstrap_confidence_interval_class


pickle_path = "/home/tanzl/code/VCUG_retrain/Out_Class.pkl"
with open(pickle_path, 'rb') as file:
    data = pickle.load(file)

print(pickle_path)
bounds = dict()
for k,v in data.items():
    print(k)
    acc_es, precision_es, recall_es, f1_es, fpr_es, tpr_es, auc_es, detailes = v
    labels, preds, preds_probability = detailes
    bound = bootstrap_confidence_interval_class(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)
    bounds[k] = bound

with open('/home/tanzl/code/VCUG_retrain/eval_pkls/result_bound/bound_Out_Class.pkl', 'wb') as file:
    pickle.dump(bounds, file)