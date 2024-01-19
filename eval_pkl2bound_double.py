import pickle
import sys
sys.path.append("../..")
from VCUG_retrain.utils.roc import bootstrap_confidence_interval


def load_pickle_file(pickle_path):
    with open(pickle_path, 'rb') as file:
        return pickle.load(file)


def save_pickle_file(data, pickle_path):
    with open(pickle_path, 'wb') as file:
        pickle.dump(data, file)


def combine_labels_preds_prob(labels_single, preds_single, preds_probability_single, labels_double, preds_double, preds_probability_double):
    labels = labels_single + labels_double
    preds = preds_single + preds_double
    preds_probability = preds_probability_single.tolist() + preds_probability_double.tolist()
    return labels, preds, preds_probability


def calculate_bootstrap_confidence_interval(labels, preds, preds_probability):
    return bootstrap_confidence_interval(labels, preds, preds_probability, num_iterations=1000, alpha=0.05, verbose=True)


def process_data(data, data_double):
    bounds = dict()
    for k, v in data.items():
        results, fprs, tprs, label_preds_prob = v
        labels_single, preds_single, preds_probability_single = label_preds_prob

        if "VCUG" in k:
            results, fprs, tprs, label_preds_prob = data_double[k]
            labels_double, preds_double, preds_probability_double = label_preds_prob
        else:
            results, fprs, tprs, label_preds_prob = data_double[k+"_2branch"]
            labels_double, preds_double, preds_probability_double = label_preds_prob

        labels, preds, preds_probability = combine_labels_preds_prob(labels_single, preds_single, preds_probability_single, labels_double, preds_double, preds_probability_double)
        bound = calculate_bootstrap_confidence_interval(labels, preds, preds_probability)
        bounds[k] = bound

    return bounds


def main():
    pickle_path = ".."
    pickle_path_double = ".."
    data = load_pickle_file(pickle_path)
    data_double = load_pickle_file(pickle_path_double)
    bounds = process_data(data, data_double)
    save_pickle_file(bounds, '..')


if __name__ == "__main__":
    main()
