import os
import numpy as np
from src import utils, data_processing, visualization, model_training, model_evaluation


def main():
    # Load the dataset
    data_path = os.path.join("data", "raw", "1", "Customers.csv")
    df = utils.load_data(data_path)

    # Set the spending_score_threshold
    spending_score_threshold = df["Spending Score (1-100)"].mean()

    # Preprocess the data
    X_train, X_test, y_train, y_test, df_processed = data_processing.preprocess_data(
        df, spending_score_threshold
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

    # Model training
    logistic_model, rf_model = model_training.train_models(X_train, y_train)

    # Model evaluation
    model_evaluation.evaluate_models(logistic_model, rf_model, X_test, y_test)


if __name__ == "__main__":
    main()
