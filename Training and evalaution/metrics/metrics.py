import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def convert_to_one_hot(labels, num_classes):
    """
    Convert integer labels to one-hot encoding.
    """
    one_hot_labels = np.eye(num_classes)[labels]
    return one_hot_labels


def calculate_roc_auc(probabilities, all_labels):

    probabilities_np = np.array(probabilities)
    all_labels_np = np.array(all_labels, dtype=int)

    if len(all_labels_np.shape) == 1:  # Convert labels to one-hot if not already
        num_classes = len(np.unique(all_labels_np))
        all_labels_np = convert_to_one_hot(all_labels_np, num_classes)

    num_classes = all_labels_np.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for class_idx in range(num_classes):
        y_true = all_labels_np[:, class_idx]
        y_prob = probabilities_np[:, class_idx]

        fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true, y_prob)
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

    return fpr, tpr, roc_auc


def find_best_threshold(probabilities, all_labels):
    probabilities_np = np.array(probabilities)
    all_labels_np = np.array(all_labels)

    if len(all_labels_np.shape) == 1:  # Convert labels to one-hot if not already
        num_classes = len(np.unique(all_labels_np))
        all_labels_np = convert_to_one_hot(all_labels_np, num_classes)

    num_classes = all_labels_np.shape[1]
    thresholds = dict()

    for class_idx in range(num_classes):
        y_true = all_labels_np[:, class_idx]
        y_prob = probabilities_np[:, class_idx]

        p, r, t = precision_recall_curve(y_true.astype(int), y_prob)
        # Choose the best threshold based on the highest F1 measure
        non_zero_mask = (r + p) != 0
        f1 = np.zeros_like(p)  # Initialize F1 array with zeros
        f1[non_zero_mask] = np.multiply(2, np.divide(np.multiply(
            p[non_zero_mask], r[non_zero_mask]), np.add(r[non_zero_mask], p[non_zero_mask])))

        best_threshold = t[np.argmax(f1)]
        thresholds[class_idx] = best_threshold

    return thresholds


def calculate_fpr_fnr(probabilities, all_labels, thresholds):
    probabilities_np = np.array(probabilities)
    all_labels_np = np.array(all_labels)

    if len(all_labels_np.shape) == 1:  # Convert labels to one-hot if not already
        num_classes = len(np.unique(all_labels_np))
        all_labels_np = convert_to_one_hot(all_labels_np, num_classes)

    num_classes = all_labels_np.shape[1]
    fpr_numbers = dict()
    fnr_numbers = dict()

    for class_idx in range(num_classes):
        y_true = all_labels_np[:, class_idx]
        y_prob = probabilities_np[:, class_idx]
        best_threshold = thresholds[class_idx]

        # Calculate false positive rate for the chosen threshold
        tn = sum((y_true == 0) & (y_prob < best_threshold))  # True Negatives
        fp = sum((y_true == 0) & (y_prob >= best_threshold))  # False Positives
        fpr_numbers[class_idx] = fp / (fp + tn)

        # Calculate false negative rate for the chosen threshold
        fn = sum((y_true == 1) & (y_prob < best_threshold))  # False Negatives
        tp = sum((y_true == 1) & (y_prob >= best_threshold))  # True Positives
        fnr_numbers[class_idx] = fn / (fn + tp)

    return fpr_numbers, fnr_numbers


if __name__ == '__main__':
    # Test
    probabilities_test = np.random.rand(100000, 14)
    labels_test = np.random.randint(0, 2, size=(100000, 14))

    # Test calculate_roc_auc function
    print("Test calculate_roc_auc:")
    fpr_test, tpr_test, roc_auc_test = calculate_roc_auc(
        probabilities_test, labels_test)
    # print("FPR:", fpr_test)
    # print("TPR:", tpr_test)
    print("ROC AUC:", roc_auc_test)
    print()

    # Test find_best_threshold function
    print("Test find_best_threshold:")
    thresholds_test = find_best_threshold(probabilities_test, labels_test)
    print("Best Thresholds:", thresholds_test)
    print()

    # Test calculate_fpr_fnr function
    print("Test calculate_fpr_fnr:")
    fpr_numbers_test, fnr_numbers_test = calculate_fpr_fnr(
        probabilities_test, labels_test, thresholds_test)
    print("FPR Numbers:", fpr_numbers_test)
    print("FNR Numbers:", fnr_numbers_test)
