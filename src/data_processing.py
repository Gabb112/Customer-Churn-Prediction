import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(
    df, spending_score_threshold, annual_income_threshold, work_experience_threshold
):
    """Preprocesses data for model training."""

    # Rename the columns for more clarity
    df = df.rename(
        columns={
            "Customer ID": "customer_id",
            "Annual Income ($)": "annual_income",
            "Spending Score (1-100)": "spending_score",
            "Work Experience": "work_experience",
            "Family Size": "family_size",
        }
    )

    # Handle Missing Values
    for column in ["annual_income", "work_experience", "family_size"]:
        mean_value = df[column].mean()
        df[column] = df[column].fillna(mean_value)
    for column in ["Profession", "Gender"]:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

    # Remove duplicates
    df = df.drop_duplicates()
    # Remove Customer ID
    df.drop(columns="CustomerID", inplace=True)

    # Convert Gender into numerical features
    df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

    # Feature Engineering
    df["income_per_family_size"] = df["annual_income"] / (df["family_size"] + 1e-6)
    df["age_work_experience"] = df["Age"] * df["work_experience"]

    # Target Variable Creation (More Complex Logic)
    df["churn"] = (
        (df["spending_score"] < spending_score_threshold)
        & (df["annual_income"] < annual_income_threshold)
        & (df["work_experience"] < work_experience_threshold)
    ).astype(int)

    df.drop(columns=["spending_score"], inplace=True)  # Remove spending score
    df.drop(columns=["annual_income"], inplace=True)
    df.drop(columns=["work_experience"], inplace=True)

    # Feature Engineering: Age Groups
    bins = [18, 25, 35, 50, 65, 80]
    labels = ["18-24", "25-34", "35-49", "50-64", "65+"]
    df["age_group"] = pd.cut(df["Age"], bins, labels=labels, right=False)
    df.drop(columns=["Age"], inplace=True)

    # Define features and target
    X = df.drop(columns="churn")
    y = df["churn"]

    # Define numerical and categorical columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Use ColumnTransformer to apply transformers to appropriate columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Apply the preprocessing to the features (X)
    X_preprocessed = preprocessor.fit_transform(X)

    # Train-test split after feature engineering
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, df
