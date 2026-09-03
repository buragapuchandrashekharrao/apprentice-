"""
Week 1 Data Preparation Project
Dataset: Titanic passenger survival data
Author: Student
Purpose: Data acquisition, cleaning, preprocessing and EDA.
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
RAW_FILE = RAW_DIR / "titanic.csv"
CLEAN_FILE = PROCESSED_DIR / "titanic_cleaned.csv"
SUMMARY_FILE = PROCESSED_DIR / "cleaning_summary.csv"

sns.set_theme(style="whitegrid")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def acquire_data() -> pd.DataFrame:
    """Download the public CSV and return it as a DataFrame."""
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    df = pd.read_csv(RAW_FILE)
    print(f"Downloaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a compact quality report before cleaning."""
    print("\n--- DATASET INFO ---")
    print(df.info())
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())
    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe(include="all").T)
    print("\n--- MISSING VALUES ---")
    print(df.isna().sum().sort_values(ascending=False))
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=missing.values, y=missing.index)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Column")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_missing_values.png", dpi=180)
    plt.close()


def plot_survival_by_sex(df: pd.DataFrame) -> None:
    rates = df.groupby("sex")["survived"].mean().sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=rates.index, y=rates.values)
    ax.set_title("Survival Rate by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)
    for i, value in enumerate(rates.values):
        ax.text(i, value + 0.03, f"{value:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_survival_by_sex.png", dpi=180)
    plt.close()


def plot_survival_by_class(df: pd.DataFrame) -> None:
    rates = df.groupby("pclass")["survived"].mean()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=rates.index.astype(str), y=rates.values)
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate")
    ax.set_ylim(0, 1)
    for i, value in enumerate(rates.values):
        ax.text(i, value + 0.03, f"{value:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_survival_by_class.png", dpi=180)
    plt.close()


def plot_correlation(df: pd.DataFrame, title: str, filename: str) -> None:
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=180)
    plt.close()


def clean_and_preprocess(df: pd.DataFrame):
    """Clean the raw data and prepare useful numerical features."""
    work = df.copy()
    before_rows = len(work)
    before_missing = int(work.isna().sum().sum())
    duplicates_removed = int(work.duplicated().sum())

    # 1. Remove exact duplicate records.
    work = work.drop_duplicates().copy()

    # 2. Correct numeric data types explicitly.
    numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # 3. Standardize categorical text.
    for col in ["sex", "embarked", "class", "who", "embark_town", "alive"]:
        work[col] = work[col].astype("string").str.strip()

    # 4. Cabin is too sparse as a raw identifier, but its deck letter is useful.
    #    Missing cabins become an explicit "Unknown" deck category.
    work["deck"] = work["deck"].fillna("Unknown").astype("string")
    work["deck"] = work["deck"].str[0]

    # 5. Impute Age with the median of passengers in the same class and sex.
    work["age"] = work.groupby(["pclass", "sex"])["age"].transform(
        lambda s: s.fillna(s.median())
    )
    work["age"] = work["age"].fillna(work["age"].median())

    # 6. Embarked has only a couple of missing values: use the mode.
    work["embarked"] = work["embarked"].fillna(work["embarked"].mode()[0])

    # 7. Detect extreme Fare values using IQR and cap rather than delete rows.
    #    This preserves legitimate passengers while reducing the influence of extremes.
    q1 = work["fare"].quantile(0.25)
    q3 = work["fare"].quantile(0.75)
    iqr = q3 - q1
    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    fare_outliers = int(((work["fare"] < lower) | (work["fare"] > upper)).sum())
    work["fare"] = work["fare"].clip(lower=lower, upper=upper)

    # 8. Remove redundant/leakage-prone columns.
    #    alive duplicates the target survived; class duplicates pclass.
    #    Name/Ticket are identifiers, not directly useful for this Week 1 pipeline.
    drop_cols = [
        "name", "ticket", "cabin", "alive", "class",
        "who", "adult_male", "embark_town", "alone"
    ]
    work = work.drop(columns=drop_cols, errors="ignore")

    # 9. Encode categorical variables.
    work["sex"] = work["sex"].map({"female": 0, "male": 1}).astype("int64")
    work = pd.get_dummies(work, columns=["embarked", "deck"], dtype=int)

    # 10. Scale continuous variables for downstream ML use.
    scaler = StandardScaler()
    work[["age", "fare"]] = scaler.fit_transform(work[["age", "fare"]])

    # Final quality check.
    after_missing = int(work.isna().sum().sum())

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_before",
                "rows_after",
                "columns_before",
                "columns_after",
                "missing_cells_before",
                "missing_cells_after",
                "duplicate_rows_removed",
                "fare_outliers_capped",
            ],
            "value": [
                before_rows,
                len(work),
                df.shape[1],
                work.shape[1],
                before_missing,
                after_missing,
                duplicates_removed,
                fare_outliers,
            ],
        }
    )
    return work, summary


def run():
    df = acquire_data()
    inspect_data(df)

    # EDA on the raw data.
    plot_missing_values(df)
    plot_survival_by_sex(df)
    plot_survival_by_class(df)
    plot_correlation(
        df,
        "Correlation Matrix - Raw Numeric Features",
        "04_correlation_matrix.png",
    )

    # Cleaning and preprocessing.
    cleaned, summary = clean_and_preprocess(df)

    # EDA after preprocessing.
    plot_correlation(
        cleaned,
        "Correlation Matrix - Preprocessed Features",
        "05_cleaned_correlation_matrix.png",
    )

    cleaned.to_csv(CLEAN_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n--- CLEANED DATA ---")
    print(cleaned.head())
    print("\nMissing values after cleaning:")
    print(cleaned.isna().sum().sum())
    print(f"\nSaved cleaned dataset: {CLEAN_FILE}")
    print(f"Saved cleaning summary: {SUMMARY_FILE}")
    print(f"Saved figures in: {FIG_DIR}")


if __name__ == "__main__":
    run()
