# ============================================================
# STREAMLIT APP
# ADVANCED STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# ============================================================

import streamlit as st
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced Statistical Analysis",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Advanced Statistical Analysis & Hypothesis Testing")

st.markdown("""
### Employee Performance Analysis

This application performs:

- Normality Tests
- Two-Sample Hypothesis Tests
- Confidence Interval
- One-Way ANOVA
- Tukey HSD Post-Hoc Analysis
- Two-Way ANOVA
- Statistical Interpretation
""")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

ALPHA = st.sidebar.number_input(
    "Significance Level (α)",
    min_value=0.001,
    max_value=0.10,
    value=0.05,
    step=0.01
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Decision Rule:\n\n"
    "p < α → Reject H₀\n\n"
    "p ≥ α → Fail to reject H₀"
)


# ============================================================
# LOAD DATASET
# ============================================================

st.header("📂 Dataset")

uploaded_file = st.file_uploader(
    "Upload employee_performance.csv",
    type=["csv"]
)


# Try uploaded file first
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    # Try loading default project dataset
    BASE_DIR = Path(__file__).resolve().parent

    data_path = BASE_DIR / "data" / "employee_performance.csv"

    if data_path.exists():

        df = pd.read_csv(data_path)

        st.success(
            "Default employee_performance.csv loaded successfully."
        )

    else:

        st.warning(
            "Please upload employee_performance.csv to continue."
        )

        st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.success("Dataset loaded successfully!")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("Dataset Preview")

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Department",
    "Gender",
    "Training_Method",
    "Experience_Years",
    "Salary",
    "Performance_Score",
    "Satisfaction_Score"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Descriptive Statistics",
        "🔬 Normality Tests",
        "🧪 Two-Sample Tests",
        "📊 One-Way ANOVA",
        "🔍 Tukey HSD",
        "📊 Two-Way ANOVA"
    ]
)


# ============================================================
# TAB 1
# DESCRIPTIVE STATISTICS
# ============================================================

with tab1:

    st.header("📈 Descriptive Statistics")

    st.subheader("Overall Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Department-wise Performance")

    department_summary = (
        df.groupby("Department")["Performance_Score"]
        .agg(
            Count="count",
            Mean="mean",
            Std="std",
            Minimum="min",
            Maximum="max"
        )
        .round(2)
    )

    st.dataframe(
        department_summary,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Box Plot
    # --------------------------------------------------------

    st.subheader("Performance Score by Department")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Department",
        y="Performance_Score",
        ax=ax
    )

    ax.set_title(
        "Performance Score by Department"
    )

    ax.set_xlabel("Department")

    ax.set_ylabel(
        "Performance Score"
    )

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# TAB 2
# NORMALITY TESTS
# ============================================================

with tab2:

    st.header("🔬 Normality Tests")

    st.markdown("""
    ### Hypotheses

    **Shapiro-Wilk**

    - H₀: Data is normally distributed.
    - H₁: Data is not normally distributed.

    **Kolmogorov-Smirnov**

    - H₀: Data follows the normal distribution.
    - H₁: Data does not follow the normal distribution.
    """)

    performance = (
        df["Performance_Score"]
        .dropna()
    )

    # --------------------------------------------------------
    # Shapiro-Wilk
    # --------------------------------------------------------

    shapiro_stat, shapiro_p = stats.shapiro(
        performance
    )

    # --------------------------------------------------------
    # Kolmogorov-Smirnov
    # --------------------------------------------------------

    standardized = (
        performance - performance.mean()
    ) / performance.std()

    ks_stat, ks_p = stats.kstest(
        standardized,
        "norm"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Shapiro-Wilk")

        st.metric(
            "Statistic",
            f"{shapiro_stat:.4f}"
        )

        st.metric(
            "p-value",
            f"{shapiro_p:.4f}"
        )

        if shapiro_p < ALPHA:

            st.error(
                "Reject H₀: Data is not normally distributed."
            )

        else:

            st.success(
                "Fail to reject H₀: No significant evidence of non-normality."
            )

    with col2:

        st.subheader("Kolmogorov-Smirnov")

        st.metric(
            "Statistic",
            f"{ks_stat:.4f}"
        )

        st.metric(
            "p-value",
            f"{ks_p:.4f}"
        )

        if ks_p < ALPHA:

            st.error(
                "Reject H₀: Data differs significantly from normal."
            )

        else:

            st.success(
                "Fail to reject H₀: No significant evidence of non-normality."
            )


    # --------------------------------------------------------
    # Results Table
    # --------------------------------------------------------

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

    st.subheader("Normality Results")

    st.dataframe(
        normality_results,
        use_container_width=True
    )

    # Download
    st.download_button(
        label="⬇️ Download Normality Results",
        data=normality_results.to_csv(
            index=False
        ),
        file_name="normality_tests.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 3
# TWO-SAMPLE TESTS
# ============================================================

with tab3:

    st.header("🧪 Two-Sample Hypothesis Tests")

    st.markdown("""
    We compare the Performance Score between:

    **Online Training vs Classroom Training**
    """)

    online = df.loc[
        df["Training_Method"] == "Online",
        "Performance_Score"
    ].dropna()

    classroom = df.loc[
        df["Training_Method"] == "Classroom",
        "Performance_Score"
    ].dropna()


    # --------------------------------------------------------
    # Group Statistics
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Online Training")

        st.metric(
            "Sample Size",
            len(online)
        )

        st.metric(
            "Mean Performance",
            f"{online.mean():.2f}"
        )

    with col2:

        st.subheader("Classroom Training")

        st.metric(
            "Sample Size",
            len(classroom)
        )

        st.metric(
            "Mean Performance",
            f"{classroom.mean():.2f}"
        )


    # --------------------------------------------------------
    # T-Test
    # --------------------------------------------------------

    st.subheader(
        "Independent Two-Sample t-Test"
    )

    st.markdown("""
    **H₀:** The two group means are equal.

    **H₁:** The two group means are different.
    """)

    t_stat, t_p = stats.ttest_ind(
        online,
        classroom,
        equal_var=False
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "t-statistic",
            f"{t_stat:.4f}"
        )

    with col2:

        st.metric(
            "p-value",
            f"{t_p:.4f}"
        )

    if t_p < ALPHA:

        st.error(
            f"Reject H₀ because p = {t_p:.4f} < {ALPHA}."
        )

        st.write(
            "There is a statistically significant "
            "difference between the two group means."
        )

    else:

        st.success(
            f"Fail to reject H₀ because p = {t_p:.4f} ≥ {ALPHA}."
        )

        st.write(
            "There is no statistically significant "
            "difference between the two group means."
        )


    # ========================================================
    # CONFIDENCE INTERVAL
    # ========================================================

    st.subheader(
        "95% Confidence Interval"
    )

    mean_online = online.mean()
    mean_classroom = classroom.mean()

    difference = (
        mean_online - mean_classroom
    )

    se = np.sqrt(
        online.var(ddof=1) / len(online)
        +
        classroom.var(ddof=1) / len(classroom)
    )

    df_welch = (
        (
            online.var(ddof=1) / len(online)
            +
            classroom.var(ddof=1) / len(classroom)
        ) ** 2
        /
        (
            (
                online.var(ddof=1)
                / len(online)
            ) ** 2
            / (len(online) - 1)
            +
            (
                classroom.var(ddof=1)
                / len(classroom)
            ) ** 2
            / (len(classroom) - 1)
        )
    )

    critical_value = stats.t.ppf(
        1 - ALPHA / 2,
        df_welch
    )

    margin = (
        critical_value * se
    )

    ci_lower = (
        difference - margin
    )

    ci_upper = (
        difference + margin
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Mean Difference",
            f"{difference:.4f}"
        )

    with col2:

        st.metric(
            "CI Lower",
            f"{ci_lower:.4f}"
        )

    with col3:

        st.metric(
            "CI Upper",
            f"{ci_upper:.4f}"
        )


    # ========================================================
    # MANN-WHITNEY
    # ========================================================

    st.subheader(
        "Mann-Whitney U Test"
    )

    st.markdown("""
    **H₀:** The two groups have the same distribution.

    **H₁:** The two groups have different distributions.
    """)

    u_stat, u_p = stats.mannwhitneyu(
        online,
        classroom,
        alternative="two-sided"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "U-statistic",
            f"{u_stat:.4f}"
        )

    with col2:

        st.metric(
            "p-value",
            f"{u_p:.4f}"
        )

    if u_p < ALPHA:

        st.error(
            f"Reject H₀ because p = {u_p:.4f} < {ALPHA}."
        )

    else:

        st.success(
            f"Fail to reject H₀ because p = {u_p:.4f} ≥ {ALPHA}."
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

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

    st.subheader(
        "Hypothesis Test Results"
    )

    st.dataframe(
        hypothesis_results,
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Download Hypothesis Results",
        data=hypothesis_results.to_csv(
            index=False
        ),
        file_name="hypothesis_tests.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 4
# ONE-WAY ANOVA
# ============================================================

with tab4:

    st.header("📊 One-Way ANOVA")

    st.markdown("""
    ### Hypotheses

    **H₀:** All department means are equal.

    **H₁:** At least one department mean is different.
    """)

    groups = [
        group["Performance_Score"].dropna()
        for _, group
        in df.groupby("Department")
    ]

    f_stat, anova_p = stats.f_oneway(
        *groups
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "F-statistic",
            f"{f_stat:.4f}"
        )

    with col2:

        st.metric(
            "p-value",
            f"{anova_p:.4f}"
        )

    if anova_p < ALPHA:

        st.error(
            f"Reject H₀ because p = {anova_p:.4f} < {ALPHA}."
        )

        st.write(
            "At least one department has a "
            "significantly different mean."
        )

    else:

        st.success(
            f"Fail to reject H₀ because p = {anova_p:.4f} ≥ {ALPHA}."
        )

        st.write(
            "No statistically significant "
            "difference was detected between departments."
        )

    # --------------------------------------------------------
    # ANOVA summary
    # --------------------------------------------------------

    anova_summary = pd.DataFrame({

        "Test": ["One-Way ANOVA"],

        "F_statistic": [f_stat],

        "p_value": [anova_p],

        "Alpha": [ALPHA]

    })

    st.dataframe(
        anova_summary,
        use_container_width=True
    )


# ============================================================
# TAB 5
# TUKEY HSD
# ============================================================

with tab5:

    st.header("🔍 Tukey HSD Post-Hoc Analysis")

    st.markdown("""
    Tukey HSD is used to determine **which specific
    departments differ from each other** after ANOVA.
    """)

    tukey = pairwise_tukeyhsd(
        endog=df["Performance_Score"],
        groups=df["Department"],
        alpha=ALPHA
    )

    st.text(
        str(tukey)
    )

    # Convert Tukey output to DataFrame

    tukey_results = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    st.subheader(
        "Tukey HSD Results"
    )

    st.dataframe(
        tukey_results,
        use_container_width=True
    )

    # Download

    st.download_button(
        label="⬇️ Download Tukey Results",
        data=tukey_results.to_csv(
            index=False
        ),
        file_name="tukey_results.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 6
# TWO-WAY ANOVA
# ============================================================

with tab6:

    st.header("📊 Two-Way ANOVA")

    st.markdown("""
    ### Factors

    1. Department
    2. Training Method
    3. Department × Training Method interaction

    ### Hypotheses

    **Department**

    H₀: Department has no effect on performance.

    **Training Method**

    H₀: Training method has no effect on performance.

    **Interaction**

    H₀: There is no Department × Training Method interaction.
    """)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = ols(
        "Performance_Score ~ "
        "C(Department) + "
        "C(Training_Method) + "
        "C(Department):C(Training_Method)",
        data=df
    ).fit()

    two_way_anova = sm.stats.anova_lm(
        model,
        typ=2
    )

    st.subheader(
        "Two-Way ANOVA Results"
    )

    st.dataframe(
        two_way_anova,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.subheader(
        "Statistical Interpretation"
    )

    for factor in two_way_anova.index:

        p_value = two_way_anova.loc[
            factor,
            "PR(>F)"
        ]

        if p_value < ALPHA:

            st.error(
                f"🔴 {factor}: Significant "
                f"(p = {p_value:.4f})"
            )

        else:

            st.success(
                f"🟢 {factor}: Not significant "
                f"(p = {p_value:.4f})"
            )


# ============================================================
# DOWNLOAD ALL RESULTS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.header("📥 Download Results")

# Normality
normality_csv = normality_results.to_csv(
    index=False
)

st.sidebar.download_button(
    "⬇️ Normality Tests",
    normality_csv,
    "normality_tests.csv",
    "text/csv"
)


# Hypothesis
hypothesis_csv = hypothesis_results.to_csv(
    index=False
)

st.sidebar.download_button(
    "⬇️ Hypothesis Tests",
    hypothesis_csv,
    "hypothesis_tests.csv",
    "text/csv"
)


# Tukey
tukey_csv = tukey_results.to_csv(
    index=False
)

st.sidebar.download_button(
    "⬇️ Tukey HSD",
    tukey_csv,
    "tukey_results.csv",
    "text/csv"
)


# Two-Way ANOVA
anova_csv = two_way_anova.to_csv()

st.sidebar.download_button(
    "⬇️ Two-Way ANOVA",
    anova_csv,
    "anova_results.csv",
    "text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Advanced Statistical Analysis & Hypothesis Testing | "
    "Python + Pandas + SciPy + Statsmodels + Streamlit"
)