# ============================================================
# ADVANCED STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# Employee Performance Dataset
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from pathlib import Path


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Get the folder where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# Data and results folders
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Create results folder automatically
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

data_path = DATA_DIR / "employee_performance.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("ADVANCED STATISTICAL ANALYSIS")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Dataset location:", data_path)
print("Dataset shape:", df.shape)


# ============================================================
# 4. DISPLAY FIRST ROWS
# ============================================================

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 5. DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())


# ============================================================
# 7. DEPARTMENT-WISE SUMMARY
# ============================================================

print("\nDepartment-wise Performance Summary:")

department_summary = (
    df.groupby("Department")["Performance_Score"]
    .agg(["count", "mean", "std", "min", "max"])
)

print(department_summary)


# ============================================================
# 8. VISUALIZATION - PERFORMANCE BY DEPARTMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Department",
    y="Performance_Score"
)

plt.title("Performance Score by Department")
plt.xlabel("Department")
plt.ylabel("Performance Score")

plt.tight_layout()
plt.show()


# ============================================================
# 9. VISUALIZATION - TRAINING METHOD
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Training_Method",
    y="Performance_Score"
)

plt.title("Performance Score by Training Method")
plt.xlabel("Training Method")
plt.ylabel("Performance Score")

plt.tight_layout()
plt.show()


# ============================================================
# 10. SIGNIFICANCE LEVEL
# ============================================================

ALPHA = 0.05

print("\nSignificance Level (Alpha):", ALPHA)


# ============================================================
# 11. NORMALITY TESTS
# ============================================================

print("\n" + "=" * 60)
print("NORMALITY TESTS")
print("=" * 60)

performance = df["Performance_Score"].dropna()


# ------------------------------------------------------------
# 11.1 Shapiro-Wilk Test
# ------------------------------------------------------------

shapiro_stat, shapiro_p = stats.shapiro(performance)

print("\nShapiro-Wilk Test")
print("-----------------")
print(f"Statistic: {shapiro_stat:.4f}")
print(f"p-value: {shapiro_p:.4f}")

if shapiro_p < ALPHA:
    print("Decision: Reject H0")
    print("Conclusion: The data is not normally distributed.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: There is no significant evidence of non-normality.")


# ------------------------------------------------------------
# 11.2 Kolmogorov-Smirnov Test
# ------------------------------------------------------------

# Standardize the data before comparing it with standard normal
standardized = (
    performance - performance.mean()
) / performance.std()

ks_stat, ks_p = stats.kstest(
    standardized,
    "norm"
)

print("\nKolmogorov-Smirnov Test")
print("----------------------")
print(f"Statistic: {ks_stat:.4f}")
print(f"p-value: {ks_p:.4f}")

if ks_p < ALPHA:
    print("Decision: Reject H0")
    print("Conclusion: The sample differs significantly from normal.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: There is no significant evidence of non-normality.")


# ============================================================
# 12. SAVE NORMALITY RESULTS
# ============================================================

normality_results = pd.DataFrame({
    "Test": [
        "Shapiro-Wilk",
        "Kolmogorov-Smirnov"
    ],
    "Statistic": [
        shapiro_stat,
        ks_stat
    ],
    "p_value": [
        shapiro_p,
        ks_p
    ],
    "Alpha": [
        ALPHA,
        ALPHA
    ]
})

normality_path = RESULTS_DIR / "normality_tests.csv"

normality_results.to_csv(
    normality_path,
    index=False
)

print("\nCreated:", normality_path)


# ============================================================
# 13. TWO-SAMPLE TESTS
# ============================================================

print("\n" + "=" * 60)
print("TWO-SAMPLE HYPOTHESIS TESTS")
print("=" * 60)


# ------------------------------------------------------------
# 13.1 Create Training Groups
# ------------------------------------------------------------

online = df.loc[
    df["Training_Method"] == "Online",
    "Performance_Score"
].dropna()

classroom = df.loc[
    df["Training_Method"] == "Classroom",
    "Performance_Score"
].dropna()

print("\nOnline group size:", len(online))
print("Classroom group size:", len(classroom))

print("\nOnline mean:", round(online.mean(), 2))
print("Classroom mean:", round(classroom.mean(), 2))


# ============================================================
# 14. INDEPENDENT TWO-SAMPLE T-TEST
# ============================================================

print("\n" + "-" * 60)
print("INDEPENDENT TWO-SAMPLE T-TEST")
print("-" * 60)

print("\nH0: Mean performance of Online and Classroom groups is equal.")
print("H1: Mean performance of Online and Classroom groups is different.")

# Welch's t-test
t_stat, t_p = stats.ttest_ind(
    online,
    classroom,
    equal_var=False
)

print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {t_p:.4f}")

if t_p < ALPHA:
    print("Decision: Reject H0")
    print("Conclusion: There is a statistically significant difference.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: There is no statistically significant difference.")


# ============================================================
# 15. 95% CONFIDENCE INTERVAL
# ============================================================

print("\n" + "-" * 60)
print("95% CONFIDENCE INTERVAL")
print("-" * 60)

mean_online = online.mean()
mean_classroom = classroom.mean()

# Difference in means
difference = mean_online - mean_classroom

# Standard error
se = np.sqrt(
    online.var(ddof=1) / len(online)
    +
    classroom.var(ddof=1) / len(classroom)
)

# Welch-Satterthwaite degrees of freedom
df_welch = (
    (
        online.var(ddof=1) / len(online)
        +
        classroom.var(ddof=1) / len(classroom)
    ) ** 2
    /
    (
        (
            online.var(ddof=1) / len(online)
        ) ** 2
        / (len(online) - 1)
        +
        (
            classroom.var(ddof=1) / len(classroom)
        ) ** 2
        / (len(classroom) - 1)
    )
)

# Critical t value
critical_value = stats.t.ppf(
    1 - ALPHA / 2,
    df_welch
)

# Margin of error
margin = critical_value * se

# Confidence interval
ci_lower = difference - margin
ci_upper = difference + margin

print("Mean difference (Online - Classroom):")
print(round(difference, 4))

print("\n95% Confidence Interval:")
print("Lower:", round(ci_lower, 4))
print("Upper:", round(ci_upper, 4))


# ============================================================
# 16. MANN-WHITNEY U TEST
# ============================================================

print("\n" + "-" * 60)
print("MANN-WHITNEY U TEST")
print("-" * 60)

print("\nH0: The two groups have the same distribution.")
print("H1: The two groups have different distributions.")

u_stat, u_p = stats.mannwhitneyu(
    online,
    classroom,
    alternative="two-sided"
)

print(f"\nU-statistic: {u_stat:.4f}")
print(f"p-value: {u_p:.4f}")

if u_p < ALPHA:
    print("Decision: Reject H0")
    print("Conclusion: The two groups are statistically different.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: No statistically significant difference was detected.")


# ============================================================
# 17. SAVE HYPOTHESIS TEST RESULTS
# ============================================================

hypothesis_results = pd.DataFrame({
    "Test": [
        "Independent t-test",
        "Mann-Whitney U"
    ],
    "Statistic": [
        t_stat,
        u_stat
    ],
    "p_value": [
        t_p,
        u_p
    ],
    "Alpha": [
        ALPHA,
        ALPHA
    ]
})

hypothesis_path = RESULTS_DIR / "hypothesis_tests.csv"

hypothesis_results.to_csv(
    hypothesis_path,
    index=False
)

print("\nCreated:", hypothesis_path)


# ============================================================
# 18. ONE-WAY ANOVA
# ============================================================

print("\n" + "=" * 60)
print("ONE-WAY ANOVA")
print("=" * 60)

print("\nH0: All department means are equal.")
print("H1: At least one department mean is different.")


# Create groups based on Department
groups = [
    group["Performance_Score"].dropna()
    for _, group in df.groupby("Department")
]


# Perform One-Way ANOVA
f_stat, anova_p = stats.f_oneway(*groups)

print(f"\nF-statistic: {f_stat:.4f}")
print(f"p-value: {anova_p:.4f}")

if anova_p < ALPHA:
    print("Decision: Reject H0")
    print("Conclusion: At least one department mean is different.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: No significant difference between department means.")


# ============================================================
# 19. TUKEY HSD POST-HOC TEST
# ============================================================

print("\n" + "=" * 60)
print("TUKEY HSD POST-HOC TEST")
print("=" * 60)

print("\nTukey HSD identifies which specific departments differ.")

tukey = pairwise_tukeyhsd(
    endog=df["Performance_Score"],
    groups=df["Department"],
    alpha=ALPHA
)

print("\n")
print(tukey)


# ============================================================
# 20. CONVERT TUKEY RESULTS TO DATAFRAME
# ============================================================

tukey_results = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

print("\nTukey Results DataFrame:")
print(tukey_results)


# ============================================================
# 21. SAVE TUKEY RESULTS
# ============================================================

tukey_path = RESULTS_DIR / "tukey_results.csv"

tukey_results.to_csv(
    tukey_path,
    index=False
)

print("\nCreated:", tukey_path)


# ============================================================
# 22. TWO-WAY ANOVA
# ============================================================

print("\n" + "=" * 60)
print("TWO-WAY ANOVA")
print("=" * 60)

print("""
Factors:
1. Department
2. Training Method
3. Department × Training Method interaction
""")

print("H0 (Department): Department has no effect on performance.")
print("H0 (Training): Training method has no effect on performance.")
print("H0 (Interaction): There is no interaction effect.")


# Create Two-Way ANOVA model
model = ols(
    "Performance_Score ~ "
    "C(Department) + "
    "C(Training_Method) + "
    "C(Department):C(Training_Method)",
    data=df
).fit()


# ANOVA table
two_way_anova = sm.stats.anova_lm(
    model,
    typ=2
)

print("\nTwo-Way ANOVA Results:")
print(two_way_anova)


# ============================================================
# 23. INTERPRET TWO-WAY ANOVA
# ============================================================

print("\nTwo-Way ANOVA Interpretation:")
print("-" * 60)

for factor in two_way_anova.index:

    p_value = two_way_anova.loc[
        factor,
        "PR(>F)"
    ]

    if p_value < ALPHA:

        print(
            f"{factor}: SIGNIFICANT "
            f"(p = {p_value:.4f})"
        )

    else:

        print(
            f"{factor}: NOT SIGNIFICANT "
            f"(p = {p_value:.4f})"
        )


# ============================================================
# 24. SAVE TWO-WAY ANOVA RESULTS
# ============================================================

anova_path = RESULTS_DIR / "anova_results.csv"

two_way_anova.to_csv(
    anova_path
)

print("\nCreated:", anova_path)


# ============================================================
# 25. TRAINING METHOD SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING METHOD SUMMARY")
print("=" * 60)

training_summary = (
    df.groupby("Training_Method")["Performance_Score"]
    .agg(["count", "mean", "std", "min", "max"])
)

print(training_summary)


# ============================================================
# 26. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL STATISTICAL SUMMARY")
print("=" * 60)

print("\nNormality Tests")
print("----------------")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}")

print("\nTwo-Sample Tests")
print("----------------")
print(f"Independent t-test p-value: {t_p:.4f}")
print(f"Mann-Whitney U p-value: {u_p:.4f}")

print("\nOne-Way ANOVA")
print("----------------")
print(f"ANOVA F-statistic: {f_stat:.4f}")
print(f"ANOVA p-value: {anova_p:.4f}")

print("\n95% Confidence Interval")
print("-----------------------")
print(f"Difference in means: {difference:.4f}")
print(f"Lower: {ci_lower:.4f}")
print(f"Upper: {ci_upper:.4f}")


# ============================================================
# 27. CHECK GENERATED FILES
# ============================================================

print("\n" + "=" * 60)
print("GENERATED RESULT FILES")
print("=" * 60)

for file in sorted(RESULTS_DIR.glob("*.csv")):
    print("✓", file.name)


# ============================================================
# 28. PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nResults saved in:")
print(RESULTS_DIR)