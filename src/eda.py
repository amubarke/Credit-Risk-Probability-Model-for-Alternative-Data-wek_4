import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDAAnalyzer:
    """
    Exploratory Data Analysis (EDA) Analyzer
    --------------------------------------
    Provides structured methods to understand dataset structure,
    summary statistics, distributions, correlations, missing values,
    and outliers.
    """

    def __init__(self, df: pd.DataFrame):
     self.df = df

     self.numeric_cols = ['Amount', 'Value']

     self.categorical_cols = [
        'ChannelId',
        'FraudResult',
        'PricingStrategy',
        'ProductCategory',
        'ProductId',
        'CountryCode',
        'CurrencyCode',
        'ProviderId',
        'ChannelId',
        'BatchId',
        'SubscriptionId',
        'AccountId',
        'TransactionStartTime'
    ]

    # 1️⃣ Dataset Structure
    def dataset_structure(self):
        """Understand rows, columns, and data types."""
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "dtypes": self.df.dtypes
        }

    # 2️⃣ Summary Statistics
    def summary_statistics(self):
        """Central tendency, dispersion, and distribution summary."""
        return self.df.describe(include='all').transpose()

    # 3️⃣ Distribution of Numerical Features
    def plot_numerical_distributions(self, bins: int = 30):
        """Visualize numerical feature distributions."""
        for col in self.numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col].dropna(), bins=bins, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    # 4️⃣ Distribution of Categorical Features
    def plot_categorical_distributions(self, top_n: int = 10):
        """Frequency distribution of categorical features."""
        for col in self.categorical_cols:
            plt.figure(figsize=(7, 4))
            self.df[col].value_counts().head(top_n).plot(kind='bar')
            plt.title(f"Top {top_n} Categories in {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

    # 5️⃣ Correlation Analysis
    def correlation_analysis(self):
        """Correlation matrix for numerical features."""
        corr = self.df[self.numeric_cols].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        return corr

    # 6️⃣ Missing Values
    def missing_values(self):
        """Identify missing values and their percentages."""
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        return pd.DataFrame({
            'missing_count': missing_count,
            'missing_percent': missing_percent
        }).sort_values(by='missing_percent', ascending=False)

    # 7️⃣ Outlier Detection (Box Plots)
    def outlier_detection(self):
        """Detect outliers using box plots for numerical features."""
        for col in self.numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(y=self.df[col])
            plt.title(f"Outlier Detection for {col}")
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()

