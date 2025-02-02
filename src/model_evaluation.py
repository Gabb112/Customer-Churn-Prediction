import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate_models(logistic_model, rf_model, X_test, y_test):
    """Evaluates the trained models."""

    # Predictions
    y_pred_logistic = logistic_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    # Evaluation Metrics
    print("Logistic Regression Metrics:")
    print(classification_report(y_test, y_pred_logistic))
    print("Accuracy:", accuracy_score(y_test, y_pred_logistic))

    print("\nRandom Forest Metrics:")
    print(classification_report(y_test, y_pred_rf))
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))

    # Confusion matrix
    cm_logistic = confusion_matrix(y_test, y_pred_logistic)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

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
    # ROC curve and AUC
    y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]
    fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_test, y_prob_logistic)
    auc_logistic = roc_auc_score(y_test, y_prob_logistic)

    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

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
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
