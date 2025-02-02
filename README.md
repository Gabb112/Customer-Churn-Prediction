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
*   Handling class imbalance using resampling

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
    *   Creates a `churn` variable by combining `spending_score`, `annual_income` and `work_experience` with custom thresholds.
*   **Feature Engineering:**
    *   Creates `income_per_family_size` and `age_work_experience` as new features.
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

## Data Resampling

The `resampling.py` module handles the following:

*   **Class Imbalance Handling:** Implements oversampling using `SMOTE` or undersampling using `RandomUnderSampler` to address class imbalance.

## Feature Selection

The `feature_selection.py` module selects important features based on Recursive Feature Elimination (RFE), using different models.

## Model Selection and Training

The `model_selection.py` module performs hyperparameter tuning using a custom cross validation loop and the `model_training.py` module is used to train the model using the hyperparameters.

The following models are trained:

*   **Logistic Regression:** A linear model to classify customers based on the defined threshold.
*   **Random Forest:** An ensemble method that uses multiple decision trees to get more accurate predictions.
*   **XGBoost:** A gradient-boosting algorithm.

## Model Evaluation

The `model_evaluation.py` module evaluates the trained models using the test data and generates the following output:

*   **Classification Report:** Shows precision, recall, F1-score, and support for each class.
*   **Accuracy Score:** Shows the accuracy of the prediction.
*   **Confusion Matrix**: Visual representation of the model's prediction accuracy.
*   **ROC Curve and AUC Score:** Visualizes the model's ability to distinguish between the positive and negative classes.
*    **Precision-Recall Curve and AUC-PR Score:** Visualizes the trade-off between precision and recall, useful when dealing with imbalanced datasets.

### Current Results

After implementing the changes, the models achieved the following performance:

*   **Logistic Regression:**
    ```
    Logistic Regression Metrics:
                  precision    recall  f1-score   support

               0       0.95      0.60      0.73       339
               1       0.27      0.82      0.40        61

        accuracy                           0.63       400
       macro avg       0.61      0.71      0.57       400
    weighted avg       0.84      0.63      0.68       400

    Accuracy: 0.6325
    ```
*   **Random Forest:**
    ```
    Random Forest Metrics:
                  precision    recall  f1-score   support

               0       0.92      0.90      0.91       339
               1       0.52      0.59      0.55        61

        accuracy                           0.85       400
       macro avg       0.72      0.75      0.73       400
    weighted avg       0.86      0.85      0.86       400

    Accuracy: 0.855
    ```
*   **XGBoost:**
    ```
    XGBoost Metrics:
                  precision    recall  f1-score   support

               0       0.93      0.88      0.90       339
               1       0.48      0.61      0.54        61

        accuracy                           0.84       400
       macro avg       0.70      0.74      0.72       400
    weighted avg       0.86      0.84      0.85       400

    Accuracy: 0.84
    ```

    As we can see, the Random Forest is performing the best overall, with a high accuracy score. However, the f1-score for class 1 is still lower compared to class 0, highlighting a class imbalance problem.

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

## Project Report (Optional)

If you have created a project report (`reports/report.md`), include key findings, challenges, and any recommendations.

## Future Work

*   Fine-tune the model further to improve the f1 score for class 1.
*   Explore more advanced feature engineering techniques, and features based on the business logic.
*   Try different machine learning models and hyperparameter tuning strategies, focusing on Random Forest.
*   Add external data to improve the predictions.
*   Implement a way to save and load the models.
*   Deploy the model in a more production environment using APIs.
*  Explore Model interpretability techniques.

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
*   `imbalanced-learn`
