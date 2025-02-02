from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(X_train, y_train, best_params_rf, best_params_lr, best_params_xgb):
    """Trains Logistic Regression, Random Forest and XGB models."""

    # Logistic Regression
    logistic_model = LogisticRegression(
        **best_params_lr, random_state=42, solver="liblinear"
    )
    logistic_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier(**best_params_rf, random_state=42)
    rf_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = XGBClassifier(**best_params_xgb, random_state=42)
    xgb_model.fit(X_train, y_train)

    return logistic_model, rf_model, xgb_model
