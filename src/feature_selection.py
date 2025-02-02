import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE


def select_features(X_train, y_train, model_type="rf", k=10):
    """Select top k features based on model feature importance."""
    if model_type == "rf":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "lr":
        model = LogisticRegression(random_state=42, solver="liblinear")
    elif model_type == "xgb":
        model = XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )
    else:
        raise ValueError("Invalid model_type specified")

    rfe = RFE(estimator=model, n_features_to_select=k)
    fit = rfe.fit(X_train, y_train)

    feature_names = pd.DataFrame(X_train).columns
    selected_features = [
        feature_names[i] for i in range(len(feature_names)) if fit.support_[i]
    ]
    X_train_selected = pd.DataFrame(X_train, columns=feature_names)[selected_features]

    return X_train_selected, selected_features
