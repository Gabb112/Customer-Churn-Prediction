# Customer Churn Prediction Project

This project focuses on predicting customer churn for a fictional shop using machine learning techniques. The goal is to identify customers who are likely to stop using the shop's services, enabling the business to take proactive measures to retain them.

## Overview

This project demonstrates the end-to-end process of building a machine learning model, including:

*   Data loading and preprocessing
*   Exploratory data analysis (EDA)
*   Feature engineering
*   Feature selection
*   Model selection and hyperparameter tuning
*   Model training
*   Model evaluation and visualization

## File Structure
*   `data/raw`: Contains the original, raw data.
*   `data/processed`: Contains the processed dataset used for the training of the model.
*   `src`: Contains the Python modules used in the project.
*   `reports`: This directory is intended for containing the reports for the project.
*   `notebooks`: For any exploratory analysis.
*   `requirements.txt`: A file which has a list of all the libraries.

## Data Description

The dataset used for this project is the "Shop Customer Data", which includes the following features:

*   **Customer ID**: Unique identifier for each customer.
*   **Gender**: Customer's gender (Male/Female).
*   **Age**: Customer's age.
*   **Annual Income**: Customer's annual income.
*   **Spending Score**: A score assigned by the shop based on customer behavior and spending nature.
*   **Profession**: Customer's profession.
*   **Work Experience**: Customer's work experience in years.
*   **Family Size**: Customer's family size.

The dataset is available at: [https://www.kaggle.com/datasets/datascientistanna/customers-dataset](https://www.kaggle.com/datasets/datascientistanna/customers-dataset)

## Data Preprocessing

The `data_processing.py` module handles the following:

*   **Data Loading:** Loads the dataset from the specified CSV file.
*   **Data Cleaning:**
    *   Handles missing values by imputing with the mean for numerical columns and mode for categorical ones.
    *   Removes duplicate rows.
    *   Removes the `customer_id` column.
*   **Feature Transformation:**
    *   Converts the `Gender` column to numerical (0/1).
*   **Target Variable Creation:**
    *   Creates a `churn` variable by setting a threshold for the spending score, where customers with low spending scores are more likely to churn.
*   **Feature Engineering:**
    *   Creates `age_groups` by bucketing the `Age` feature.
*   **Preprocessing:**
    *   Standardizes numerical features using `StandardScaler`.
    *   One-hot encodes categorical features using `OneHotEncoder`.
*   **Train-Test Split:** Splits the data into training and testing sets with an 80/20 split.

## Exploratory Data Analysis (EDA)

The `visualization.py` module performs the following:

*   **Univariate Analysis**: Plots histograms of all the numerical columns.
*   **Bivariate Analysis**: Plots a pairplot to see the correlations between features.
*   **Correlation Heatmap**: Shows the correlation between all numerical features with a heatmap.

## Feature Selection

The `feature_selection.py` module selects important features based on model feature importance. The features are selected based on the selected model which is passed as an argument to the function.

## Model Selection and Training

The `model_selection.py` module performs hyperparameter tuning using a custom cross validation loop and the `model_training.py` module is used to train the model using the hyperparameters.

The following models are trained:

*   **Logistic Regression:**  A linear model to classify customers based on the defined threshold.
*   **Random Forest:** An ensemble method that uses multiple decision trees to get more accurate predictions.
*   **XGBoost:**  A gradient-boosting algorithm.

## Model Evaluation

The `model_evaluation.py` module evaluates the trained models using the test data and generates the following output:

*   **Classification Report:** Shows precision, recall, F1-score, and support for each class.
*   **Accuracy Score:** Shows the accuracy of the prediction.
*   **Confusion Matrix**: Visual representation of the model's prediction accuracy.
*   **ROC Curve and AUC Score:** Visualizes the model's ability to distinguish between the positive and negative classes.
*   **Precision-Recall Curve and AUC-PR Score:** Visualizes the trade-off between precision and recall, useful when dealing with imbalanced datasets.

## How to Run the Project

1.  **Clone the Repository:** Clone the repository to your local machine using `git clone`.
2.  **Set up your environment**:
    *   Create a new environment using `python -m venv churn_venv`
    *   Activate the environment using `source churn_venv/bin/activate`
    *   Install dependencies using `pip install -r requirements.txt`
3.  **Download the Data:** Download the dataset from the provided Kaggle link and place the `Customers.csv` file in the `data/raw/1/` directory.
4.  **Run the Main Script:** Navigate to the root directory of the project and execute the `main.py` script:

    ```bash
    python src/main.py
    ```

## Future Work

*   Explore more advanced feature engineering techniques.
*   Try different machine learning models and hyperparameter tuning strategies.
*   Add external data to improve the predictions.
*   Implement a way to save and load the models.
*   Deploy the model in a more production environment using APIs.

## Dependencies

*   Python (3.7+)
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `kagglehub`
*   `xgboost`
*   `itertools`
