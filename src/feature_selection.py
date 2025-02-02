import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def select_features(X_train, y_train, model_type="rf", k="all"):
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

    model.fit(X_train, y_train)

    if k == "all":
        k = X_train.shape[1]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise AttributeError(
            "Model does not have feature_importances_ or coef_ attribute"
        )

    feature_names = pd.DataFrame(X_train).columns
    feature_importances = pd.Series(importances, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False)

    if k < len(feature_importances):
        selected_features = feature_importances.head(k).index.tolist()
        X_train_selected = pd.DataFrame(X_train, columns=feature_names)[
            selected_features
        ]
    else:
        X_train_selected = pd.DataFrame(X_train, columns=feature_names)

    return X_train_selected, selected_features
