import pickle
import sys
sys.path.append("../..")
from VCUG_retrain.utils.roc import bootstrap_confidence_interval


def load_data(pickle_path):
    with open(pickle_path, 'rb') as file:
        return pickle.load(file)


def calculate_bounds(data):
    bounds = dict()
    for k, v in data.items():
        print(k)
        results, fprs, tprs, label_preds_prob = v
        labels_single, preds_single, preds_probability_single = label_preds_prob

        labels = labels_single
        preds = preds_single
        preds_probability = preds_probability_single.tolist()
        bound = bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)
        bounds[k] = bound
    return bounds


def save_bounds(bounds, pickle_path):
    with open(pickle_path, 'wb') as file:
        pickle.dump(bounds, file)


if __name__ == "__main__":
    pickle_path = ".."
    data = load_data(pickle_path)
    bounds = calculate_bounds(data)
    save_bounds(bounds, "..")