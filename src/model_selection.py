from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import itertools


def select_best_model_params(X_train, y_train):
    """Tunes hyperparameters using a custom cross-validation loop."""

    # Define Parameter Grids
    param_grid_lr = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "class_weight": ["balanced", None],
    }
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    param_grid_xgb = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "use_label_encoder": [False],
        "class_weight": ["balanced", None],
    }

    # Define Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Function to perform Grid Search on a single model
    def grid_search(X, y, model_type, param_grid):

        best_score = 0
        best_params = {}
        for params in parameter_grid_search(param_grid):

            scores = []
            for train_index, val_index in cv.split(X, y):

                X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
                y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

                if model_type == "lr":
                    model = LogisticRegression(
                        **params, random_state=42, solver="liblinear"
                    )
                elif model_type == "rf":
                    model = RandomForestClassifier(**params, random_state=42)
                elif model_type == "xgb":
                    model = XGBClassifier(
                        **params, random_state=42, eval_metric="logloss"
                    )

                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                scores.append(f1_score(y_val_fold, y_pred))

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        return best_params

    def parameter_grid_search(param_grid):
        keys, values = zip(*param_grid.items())
        for value_tuple in itertools.product(*values):
            yield dict(zip(keys, value_tuple))

    # Perform Grid Search for Each Model
    best_params_lr = grid_search(pd.DataFrame(X_train), y_train, "lr", param_grid_lr)
    best_params_rf = grid_search(pd.DataFrame(X_train), y_train, "rf", param_grid_rf)
    best_params_xgb = grid_search(pd.DataFrame(X_train), y_train, "xgb", param_grid_xgb)

    return best_params_rf, best_params_lr, best_params_xgb
