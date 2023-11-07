import pickle
import sys
sys.path.append("../..")
from VCUG_retrain.utils.roc import bootstrap_confidence_interval


pickle_path = "/home/tanzl/code/VCUG_retrain/eval_pkls/Out_double.pkl"
with open(pickle_path, 'rb') as file:
    data = pickle.load(file)

print(pickle_path)

bounds = dict()
for k,v in data.items():
    print(k)
    results, fprs, tprs, label_preds_prob = v
    labels_single, preds_single, preds_probability_single = label_preds_prob
    
    labels = labels_single
    preds = preds_single
    preds_probability = preds_probability_single.tolist()
    bound = bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)
    bounds[k] = bound


pkl_name = "Out_double.pkl"
with open('/home/tanzl/code/VCUG_retrain/eval_pkls/result_bound/bound_{}'.format(pkl_name), 'wb') as file:
    pickle.dump(bounds, file)