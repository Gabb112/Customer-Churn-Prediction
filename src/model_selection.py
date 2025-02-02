from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def select_best_model_params(X_train, y_train):
    """Tunes hyperparameters using GridSearchCV."""

    # Logistic Regression Hyperparameters
    param_grid_lr = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "class_weight": ["balanced", None],
    }

    # Random Forest Hyperparameters
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    # XGBoost Hyperparameters
    param_grid_xgb = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "use_label_encoder": [False],
        "class_weight": ["balanced", None],
    }

    # Cross-validation settings
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search for Logistic Regression
    grid_search_lr = GridSearchCV(
        LogisticRegression(random_state=42, solver="liblinear"),
        param_grid_lr,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search_lr.fit(X_train, y_train)
    best_params_lr = grid_search_lr.best_params_

    # Grid search for Random Forest
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search_rf.fit(X_train, y_train)
    best_params_rf = grid_search_rf.best_params_

    # Grid search for XGBoost
    grid_search_xgb = GridSearchCV(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        param_grid_xgb,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search_xgb.fit(X_train, y_train)
    best_params_xgb = grid_search_xgb.best_params_

    return best_params_rf, best_params_lr, best_params_xgb
