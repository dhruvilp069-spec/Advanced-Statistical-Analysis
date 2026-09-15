# Advanced Statistical Analysis & Hypothesis Testing

## Project Overview

This project performs advanced statistical analysis and hypothesis testing
on an Employee Performance dataset using Python.

The project includes parametric and non-parametric hypothesis tests,
normality tests, One-Way ANOVA, Two-Way ANOVA, and Tukey HSD post-hoc analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## Statistical Tests

### 1. Normality Tests

- Shapiro-Wilk Test
- Kolmogorov-Smirnov Test

### 2. Two-Sample Tests

- Independent Two-Sample t-Test
- Mann-Whitney U Test
- 95% Confidence Interval

### 3. ANOVA

- One-Way ANOVA
- Tukey HSD Post-Hoc Test
- Two-Way ANOVA

## Significance Level

The default significance level is:

α = 0.05

Decision rule:

- p < 0.05 → Reject the Null Hypothesis
- p ≥ 0.05 → Fail to Reject the Null Hypothesis

## Dataset

The dataset contains employee information including:

- Department
- Gender
- Training Method
- Experience Years
- Salary
- Performance Score
- Satisfaction Score

## Project Structure

```text
Advanced_Statistical_Analysis/
│
├── app.py
├── statistical_analysis.py
├── statistical_analysis.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── employee_performance.csv
│
└── results/
    ├── normality_tests.csv
    ├── hypothesis_tests.csv
    ├── anova_results.csv
    └── tukey_results.csv