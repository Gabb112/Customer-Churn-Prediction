from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    """Trains Logistic Regression and Random Forest models."""
    # Logistic Regression
    logistic_model = LogisticRegression(random_state=42, solver='liblinear')
    logistic_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    return logistic_model, rf_model