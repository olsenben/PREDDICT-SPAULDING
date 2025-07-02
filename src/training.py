from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np
from tqdm import tqdm



def learning_curve_auc(X, y, sample_sizes, model, n_splits=5):
    means = []
    stds = []
    
    for frac in tqdm(sample_sizes, desc="Training Data Samples"):
        aucs = []
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=frac, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = model

            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_prob)
            aucs.append(auc)

        means.append(np.mean(aucs))
        stds.append(np.std(aucs))
    
    return means, stds


def learning_curve_pr(X, y, sample_sizes, model, n_splits=5):
    ap_means = []
    ap_stds = []
    pr_curves = []

    for frac in tqdm(sample_sizes, desc="Training Data Samples"):
        ap_scores = []
        pr_points = []

        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=frac, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = model
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # Store AP score
            ap = average_precision_score(y_test, y_pred_prob)
            ap_scores.append(ap)

            # Store PR curve (just for first fold to avoid clutter)
            if len(pr_points) == 0:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
                pr_points.append((precision, recall))

        ap_means.append(np.mean(ap_scores))
        ap_stds.append(np.std(ap_scores))
        pr_curves.append(pr_points[0])  # only one per frac, for plotting

    return ap_means, ap_stds, pr_curves