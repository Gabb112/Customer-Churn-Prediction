import os
import numpy as np
import pandas as pd
from src import (
    utils,
    data_processing,
    visualization,
    model_training,
    model_evaluation,
    model_selection,
    feature_selection,
    resampling,
)


def main():
    # Load the dataset
    data_path = os.path.join("data", "raw", "1", "Customers.csv")
    df = utils.load_data(data_path)

    # Set the thresholds
    spending_score_threshold = df["Spending Score (1-100)"].mean()
    annual_income_threshold = df["Annual Income ($)"].mean()
    work_experience_threshold = df["Work Experience"].mean()

    # Preprocess the data
    X_train, X_test, y_train, y_test, df_processed = data_processing.preprocess_data(
        df, spending_score_threshold, annual_income_threshold, work_experience_threshold
    )
    # Save processed data
    processed_data_path = os.path.join("data", "processed", "processed_data.csv")
    utils.save_data(df_processed, processed_data_path)

    # Univariate Analysis
    numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    visualization.plot_histograms(df_processed, numerical_cols)

    # Bivariate Analysis (scatter plots)
    visualization.plot_pairplot(df_processed, "Gender")

    # Correlation Heatmap
    visualization.plot_correlation_heatmap(df_processed)

    # Resample the data
    X_train_resampled, y_train_resampled = resampling.resample_data(
        X_train, y_train, method="smote"
    )

    # Feature selection
    X_train_selected, selected_features = feature_selection.select_features(
        X_train_resampled, y_train_resampled, model_type="rf", k=10
    )
    X_test_selected = pd.DataFrame(X_test, columns=pd.DataFrame(X_train).columns)[
        selected_features
    ]

    # Model Selection
    best_params_rf, best_params_lr, best_params_xgb = (
        model_selection.select_best_model_params(X_train_selected, y_train_resampled)
    )

    # Model training
    logistic_model, rf_model, xgb_model = model_training.train_models(
        X_train_selected,
        y_train_resampled,
        best_params_rf,
        best_params_lr,
        best_params_xgb,
    )

    # Model evaluation
    model_evaluation.evaluate_models(
        logistic_model, rf_model, xgb_model, X_test_selected, y_test
    )


if __name__ == "__main__":
    main()
