import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

####st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Automated EDA & Preprocessing Tool")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ================== SUMMARY STATISTICS ==================
    st.write("## 🔍 Automated Insights")

    # Basic Statistics
    st.write("### Descriptive Statistics (Numeric Columns)")
    st.write(df.describe().T)

    # Outlier Detection using IQR
    st.write("### Outlier Detection")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        outlier_report = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
            outlier_report[col] = len(outliers)
        st.write(pd.DataFrame.from_dict(outlier_report, orient='index', columns=['Outlier Count']))

    # Skewness & Kurtosis
    st.write("### Distribution Shape (Skewness & Kurtosis)")
    if not numeric_df.empty:
        dist_stats = pd.DataFrame({
            "Skewness": numeric_df.skew(),
            "Kurtosis": numeric_df.kurt()
        })
        st.write(dist_stats)

    # Top Categories
    st.write("### Top Categories in Categorical Columns")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.write(f"**{col}**:")
            st.write(df[col].value_counts().head(5))

    # ================== MISSING VALUES ==================
    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    missing_option = st.selectbox("Choose how to handle missing values:", 
                                  ["Do Nothing", "Fill with Mean", "Fill with Median", "Drop Missing Values"])
    if missing_option == "Fill with Mean":
        df.fillna(df.select_dtypes(include='number').mean(), inplace=True)
    elif missing_option == "Fill with Median":
        df.fillna(df.select_dtypes(include='number').median(), inplace=True)
    elif missing_option == "Drop Missing Values":
        df.dropna(inplace=True)
    st.write("Updated Missing Values Count:", df.isnull().sum())

    # ================== DUPLICATES ==================
    if st.checkbox("Remove Duplicate Rows"):
        df.drop_duplicates(inplace=True)
        st.write("Duplicates Removed! Updated Dataset:")
        st.dataframe(df.head())

    # ================== DATE HANDLING ==================
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # ================== CORRELATION ==================
    numeric_df = df.select_dtypes(include=['number'])
    st.write("### Correlation Heatmap")
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation heatmap.")

    # ================== HISTOGRAM ==================
    st.write("### Data Distribution")
    if not numeric_df.empty:
        selected_column = st.selectbox("Select a numeric column", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column].dropna(), kde=True, bins=30, ax=ax)
        st.pyplot(fig)

    # ================== ENCODING ==================
    st.write("### Encode Categorical Columns")
    if len(categorical_cols) > 0:
        selected_cat_col = st.selectbox("Select a categorical column to encode:", categorical_cols)
        encoder = LabelEncoder()
        df[selected_cat_col + "_encoded"] = encoder.fit_transform(df[selected_cat_col].astype(str))
        st.write(f"Encoded Column '{selected_cat_col}':")
        st.dataframe(df.head())

    # ================== SCALING ==================
    st.write("### Feature Scaling (Standardization)")
    numeric_df = df.select_dtypes(include=['number'])
    scale_cols = st.multiselect("Select columns to scale:", numeric_df.columns)
    if len(scale_cols) > 0:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        st.write("Scaled Data Preview:")
        st.dataframe(df.head())

    # ================== DOWNLOAD ==================
    st.write("### Download Processed Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")

else:
    st.write("**Please upload a CSV file to start analysis.**")
