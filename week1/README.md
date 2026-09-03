# Week 1 Data Preparation & EDA — Titanic Dataset

## Project objective
This project demonstrates a complete Week 1 data-science workflow:
1. Acquire a public dataset.
2. Inspect data quality.
3. Handle missing values and duplicates.
4. Correct data types and categorical values.
5. Detect and treat outliers.
6. Encode categorical variables.
7. Scale continuous variables.
8. Perform exploratory data analysis (EDA).
9. Export a cleaned dataset and visualizations.

## Dataset
**Titanic passenger survival dataset**, sourced from the public `seaborn-data` GitHub repository:

`https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv`

The dataset contains 891 passenger records and 15 columns in this version. The important fields include survival status, passenger class, sex, age, family counts, fare, embarkation, and cabin/deck information.

## Project structure

```text
week1_titanic_project/
├── data/
│   ├── raw/                 # downloaded CSV
│   └── processed/           # cleaned CSV + cleaning summary
├── outputs/
│   └── figures/             # generated charts
├── reports/
│   └── Week1_Titanic_Data_Preparation_Report.docx
├── src/
│   └── main.py              # complete pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run

### 1. Create an environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python src/main.py
```

The script downloads the raw data automatically and creates:
- `data/raw/titanic.csv`
- `data/processed/titanic_cleaned.csv`
- `data/processed/cleaning_summary.csv`
- five PNG visualizations in `outputs/figures/`

## Cleaning decisions

### Missing values
- **Age:** imputed using the median age within the same passenger class and sex, then a global median fallback.
- **Embarked:** filled with the mode because only two observations are missing.
- **Cabin:** the raw cabin identifier is highly sparse, so it is converted into a `deck` feature and missing values are represented as `Unknown`.

### Duplicates
Exact duplicate rows are removed before analysis.

### Outliers
Fare is checked using the IQR rule. Extreme fares are capped to the IQR upper/lower bounds instead of deleting rows, because high fares can represent genuine first-class passengers.

### Encoding and scaling
- `sex`: female → 0, male → 1.
- `embarked` and `deck`: one-hot encoded.
- `age` and `fare`: standardized using `StandardScaler`.

## EDA questions
The visualizations investigate:
- Where are the missing values?
- Does survival differ by sex?
- Does passenger class relate to survival?
- Which numeric variables are correlated?

## Key findings
- The raw data contains substantial missingness in `deck/cabin` and `age`, while `embarked` is almost complete.
- Survival is substantially higher among female passengers than male passengers.
- First-class passengers have a higher survival rate than second- and third-class passengers.
- Fare has a wide distribution and extreme values, so it is treated carefully rather than blindly deleting observations.

## Reproducibility
The raw dataset is not hard-coded into the project. Running `src/main.py` downloads the public CSV from the source URL, making the workflow reproducible.
