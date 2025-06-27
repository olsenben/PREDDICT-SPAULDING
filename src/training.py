from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm



def learning_curve_auc(X, y, sample_sizes, n_splits=5):
    means = []
    stds = []
    
    for frac in tqdm(sample_sizes, desc="Training Data Samples"):
        aucs = []
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=frac, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = SGDClassifier(
                loss='log_loss',  # logistic regression
                penalty='l2',
                alpha=0.001,       # regularization strength (like 1/C)
                max_iter=1000,
                class_weight='balanced',
                random_state=42
                )

            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_prob)
            aucs.append(auc)

        means.append(np.mean(aucs))
        stds.append(np.std(aucs))
    
    return means, stds