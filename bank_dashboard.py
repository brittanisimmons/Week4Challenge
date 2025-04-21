# streamlit
streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set page config
st.set_page_config(page_title="Bank Data Analysis Dashboard", layout="wide")

# Title
st.title("Bank Data Analysis Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your bank data CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file, low_memory=False)
    
    bank_data = load_data(uploaded_file)
    
    # Sidebar for operations
    st.sidebar.header("Data Processing Options")
    
    # KNN Imputation section
    st.sidebar.subheader("KNN Imputation")
    perform_imputation = st.sidebar.checkbox("Perform KNN Imputation")
    
    if perform_imputation:
        sample_size = st.sidebar.slider("Sample size for KNN fitting", 1000, 10000, 5000)
        n_neighbors = st.sidebar.slider("Number of neighbors (k)", 1, 10, 5)
        
        # Data processing
        features = ['demog_inc', 'demog_homeval', 'demog_age']
        
        # Convert to numeric and handle age flags
        bank_data['age_flag'] = (bank_data['demog_age'] <= 0) | (bank_data['demog_age'].isna())
        bank_data['demog_age'] = bank_data['demog_age'].where(bank_data['demog_age'] > 0, np.nan)
        
        for col in features:
            bank_data[col] = pd.to_numeric(bank_data[col], errors='coerce')
        
        # Perform KNN imputation
        sample = bank_data[features].dropna().sample(n=sample_size, random_state=42)
        scaler = StandardScaler()
        sample_scaled = scaler.fit_transform(sample)
        
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer.fit(sample_scaled)
        
        full_scaled = scaler.transform(bank_data[features])
        imputed_scaled = imputer.transform(full_scaled)
        imputed = scaler.inverse_transform(imputed_scaled)
        
        bank_data['demog_age_imputed'] = imputed[:, features.index('demog_age')]
    
    # Display data overview
    st.header("Data Overview")
    st.write("First few rows of the dataset:")
    st.dataframe(bank_data.head())
    
    # Basic statistics
    st.header("Basic Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Shape")
        st.write(f"Rows: {bank_data.shape[0]}")
        st.write(f"Columns: {bank_data.shape[1]}")
    
    with col2:
        st.subheader("Missing Values")
        st.write(bank_data.isnull().sum())
    
    # Visualizations
    st.header("Visualizations")
    
    # Age distribution
    if perform_imputation:
        fig_age = px.histogram(bank_data, 
                             x="demog_age_imputed",
                             title="Distribution of Imputed Ages",
                             labels={"demog_age_imputed": "Age"})
        st.plotly_chart(fig_age)
    
    # Income vs Home Value scatter plot
    fig_scatter = px.scatter(bank_data,
                           x="demog_inc",
                           y="demog_homeval",
                           title="Income vs Home Value",
                           labels={"demog_inc": "Income",
                                  "demog_homeval": "Home Value"})
    st.plotly_chart(fig_scatter)
    
    # Download processed data
    if st.button("Download Processed Data"):
        csv = bank_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_bank_data.csv",
            mime="text/csv"
        )

"""

# Save the streamlit app
with open('bank_dashboard.py', 'w') as f:
    f.write(streamlit_code)

print("Created bank_dashboard.py with Streamlit dashboard code")
