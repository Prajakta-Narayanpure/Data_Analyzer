import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Universal EDA Analyzer",
    layout="wide"
)

st.title("📊 Universal Data Analyzer using EDA")
st.markdown("Upload **any CSV dataset** to perform complete Exploratory Data Analysis")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully")

    # =========================
    # Dataset Preview
    # =========================
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # =========================
    # Dataset Info
    # =========================
    st.subheader("📌 Dataset Information")
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())

    # =========================
    # Data Types
    # =========================
    st.subheader("🧠 Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes,
        "Unique Values": df.nunique()
    })
    st.dataframe(dtype_df)

    # =========================
    # Categorical & Numerical
    # =========================
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("🔎 Column Classification")
    col1, col2 = st.columns(2)
    col1.write("**Categorical Columns**")
    col1.write(categorical_cols)
    col2.write("**Numerical Columns**")
    col2.write(numeric_cols)

    # =========================
    # Missing Value Analysis
    # =========================
    st.subheader("❌ Missing Value Analysis")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2)
    })
    st.dataframe(missing_df)

    # =========================
    # Descriptive Statistics
    # =========================
    st.subheader("📈 Descriptive Statistics")
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe())
    else:
        st.info("No numeric columns found")

    # =========================
    # Univariate Analysis
    # =========================
    st.subheader("📊 Univariate Analysis")

    if numeric_cols:
        num_col = st.selectbox("Select Numeric Column", numeric_cols)
        fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
        st.plotly_chart(fig, use_container_width=True)

    if categorical_cols:
        cat_col = st.selectbox("Select Categorical Column", categorical_cols)
        cat_data = df[cat_col].value_counts().reset_index()
        cat_data.columns = [cat_col, "Count"]
        fig = px.bar(cat_data, x=cat_col, y="Count",
                     labels={cat_col: cat_col, "Count": "Count"},
                     title=f"Value Counts of {cat_col}")
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Dynamic Bar Chart
    # =========================
    st.subheader("📊 Dynamic Bar Chart")

    try:
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis (Categorical)", categorical_cols, key="bar_x")
            with col2:
                y_col = st.selectbox("Y-axis (Numeric)", numeric_cols, key="bar_y")
            with col3:
                chart_type = st.selectbox("Chart Type", ["Vertical Bar", "Horizontal Bar"], key="bar_type")

            bar_data = df.groupby(x_col)[y_col].sum().reset_index()
            
            if chart_type == "Vertical Bar":
                fig = px.bar(bar_data, x=x_col, y=y_col, 
                            title=f"{y_col} by {x_col}", 
                            labels={x_col: x_col, y_col: y_col},
                            color=y_col, 
                            color_continuous_scale="Blues")
            else:
                fig = px.bar(bar_data, y=x_col, x=y_col, 
                            orientation='h',
                            title=f"{y_col} by {x_col}", 
                            color=y_col, 
                            color_continuous_scale="Blues")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Need both categorical and numeric columns for bar chart")
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")

    # =========================
    # Dynamic Pie Chart
    # =========================
    st.subheader("🥧 Dynamic Pie Chart")

    try:
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                pie_cat = st.selectbox("Category", categorical_cols, key="pie_cat")
            with col2:
                pie_num = st.selectbox("Value", numeric_cols, key="pie_num")
            with col3:
                pie_chart_type = st.selectbox("Chart Type", ["Pie Chart", "Donut Chart"], key="pie_type")

            pie_data = df.groupby(pie_cat)[pie_num].sum().reset_index()
            
            if pie_chart_type == "Pie Chart":
                fig = px.pie(pie_data, names=pie_cat, values=pie_num,
                            title=f"{pie_num} Distribution by {pie_cat}")
            else:
                fig = px.pie(pie_data, names=pie_cat, values=pie_num,
                            title=f"{pie_num} Distribution by {pie_cat} (Donut)",
                            hole=0.4)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Need both categorical and numeric columns for pie chart")
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
    
    # =========================
    # Simple Count Pie Chart (for any categorical column)
    # =========================
    st.subheader("🥧 Category Distribution Pie Chart")
    
    try:
        if len(categorical_cols) > 0:
            count_cat = st.selectbox("Select Category to Visualize", categorical_cols, key="count_pie_cat")
            count_data = df[count_cat].value_counts().reset_index()
            count_data.columns = [count_cat, "Count"]
            
            fig = px.pie(count_data, names=count_cat, values="Count",
                        title=f"Count Distribution of {count_cat}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No categorical columns available for distribution chart")
    except Exception as e:
        st.error(f"Error creating distribution chart: {str(e)}")

    # =========================
    # Correlation Heatmap
    # =========================
    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation")

    # =========================
    # Outlier Detection
    # =========================
    st.subheader("🚨 Outlier Detection")

    if numeric_cols:
        out_col = st.selectbox("Select Column for Outlier Analysis", numeric_cols)
        fig = px.box(df, y=out_col, title=f"Outliers in {out_col}")
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Download Cleaned Data
    # =========================
    st.subheader("⬇️ Download Dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "cleaned_dataset.csv",
        "text/csv"
    )

else:
    st.info("👆 Upload a CSV file to start EDA")
