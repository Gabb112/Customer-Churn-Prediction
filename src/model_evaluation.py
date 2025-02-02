import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def evaluate_models(logistic_model, rf_model, xgb_model, X_test, y_test):
    """Evaluates the trained models."""

    # Predictions
    y_pred_logistic = logistic_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluation Metrics
    print("Logistic Regression Metrics:")
    print(classification_report(y_test, y_pred_logistic))
    print("Accuracy:", accuracy_score(y_test, y_pred_logistic))

    print("\nRandom Forest Metrics:")
    print(classification_report(y_test, y_pred_rf))
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))

    print("\nXGBoost Metrics:")
    print(classification_report(y_test, y_pred_xgb))
    print("Accuracy:", accuracy_score(y_test, y_pred_xgb))

    # Confusion matrix
    cm_logistic = confusion_matrix(y_test, y_pred_logistic)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_logistic, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # ROC curve and AUC
    y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]
    fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_test, y_prob_logistic)
    auc_logistic = roc_auc_score(y_test, y_prob_logistic)

    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_prob_xgb)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_logistic,
        tpr_logistic,
        color="darkorange",
        lw=2,
        label=f"Logistic Regression AUC = {auc_logistic:.2f}",
    )
    plt.plot(
        fpr_rf, tpr_rf, color="green", lw=2, label=f"Random Forest AUC = {auc_rf:.2f}"
    )
    plt.plot(
        fpr_xgb, tpr_xgb, color="purple", lw=2, label=f"XGBoost AUC = {auc_xgb:.2f}"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall curve
    precision_logistic, recall_logistic, _ = precision_recall_curve(
        y_test, y_prob_logistic
    )
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)

    auc_pr_logistic = auc(recall_logistic, precision_logistic)
    auc_pr_rf = auc(recall_rf, precision_rf)
    auc_pr_xgb = auc(recall_xgb, precision_xgb)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall_logistic,
        precision_logistic,
        color="darkorange",
        lw=2,
        label=f"Logistic Regression AUC-PR = {auc_pr_logistic:.2f}",
    )
    plt.plot(
        recall_rf,
        precision_rf,
        color="green",
        lw=2,
        label=f"Random Forest AUC-PR = {auc_pr_rf:.2f}",
    )
    plt.plot(
        recall_xgb,
        precision_xgb,
        color="purple",
        lw=2,
        label=f"XGBoost AUC-PR = {auc_pr_xgb:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()
