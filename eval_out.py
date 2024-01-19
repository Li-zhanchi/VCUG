import pickle
import sys
sys.path.append("../..")
from VCUG_retrain.utils.roc import bootstrap_confidence_interval

pkl_name = '..'
pickle_path = ".."
with open(pickle_path, 'rb') as file:
    data = pickle.load(file)

bounds = dict()
for k,v in data.items():
    results, fprs, tprs, label_preds_prob = v
    labels, preds, preds_probability = label_preds_prob
    bound = bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)
    bounds[k] = bound

with open('..', 'wb') as file:
    pickle.dump(bounds, file)