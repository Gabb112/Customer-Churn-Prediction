import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, numerical_cols):
    """Plots histograms for numerical columns."""
    for column in numerical_cols:
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins=20, edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


def plot_pairplot(df, hue):
    """Plots pairplot of dataframe"""
    sns.pairplot(df, hue=hue)
    plt.show()


def plot_correlation_heatmap(df):
    """Plots correlation heatmap for numerical features"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
    plt.show()