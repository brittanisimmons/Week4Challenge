# Create streamlit dashboard with 3 interactive visualizations
streamlit_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Basic page config
st.set_page_config(page_title="Bank Analysis", layout="wide")

# Load data
try:
    df = pd.read_csv('bank_data.csv')
    # Convert columns to numeric
    df['demog_homeval'] = pd.to_numeric(df['demog_homeval'], errors='coerce')
    df['demog_inc'] = pd.to_numeric(df['demog_inc'], errors='coerce')
    df['demog_age'] = pd.to_numeric(df['demog_age'], errors='coerce')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title
st.title('Bank Customer Analysis')

# Create three columns for better layout
col1, col2 = st.columns(2)

# Visualization 1: Age Distribution with Income Filter
with col1:
    st.subheader('Customer Age Distribution')
    
    # Income range slider
    income_range = st.slider(
        'Filter by Income Range ($)',
        min_value=float(df['demog_inc'].min()),
        max_value=float(df['demog_inc'].max()),
        value=(float(df['demog_inc'].min()), float(df['demog_inc'].max()))
    )
    
    # Filter data based on income range
    filtered_df = df[
        (df['demog_inc'] >= income_range[0]) & 
        (df['demog_inc'] <= income_range[1])
    ]
    
    age_hist = px.histogram(
        filtered_df,
        x='demog_age',
        title='Age Distribution by Income Range',
        labels={'demog_age': 'Age', 'count': 'Number of Customers'}
    )
    st.plotly_chart(age_hist, use_container_width=True)

# Visualization 2: Income vs Home Value with Age Filter
with col2:
    st.subheader('Income vs Home Value')
    
    # Age group dropdown
    age_ranges = ['All'] + [f'{i}-{i+10}' for i in range(20, 71, 10)] + ['70+']
    selected_age = st.selectbox('Select Age Range', age_ranges)
    
    # Filter data based on age selection
    if selected_age != 'All':
        if selected_age == '70+':
            age_filtered = df[df['demog_age'] >= 70]
        else:
            age_min, age_max = map(int, selected_age.split('-'))
            age_filtered = df[
                (df['demog_age'] >= age_min) & 
                (df['demog_age'] < age_max)
            ]
    else:
        age_filtered = df
    
    scatter = px.scatter(
        age_filtered,
        x='demog_inc',
        y='demog_homeval',
        title=f'Income vs Home Value (Age: {selected_age})',
        labels={
            'demog_inc': 'Income ($)',
            'demog_homeval': 'Home Value ($)'
        }
    )
    st.plotly_chart(scatter, use_container_width=True)

# Visualization 3: RFM Analysis
st.subheader('RFM Metrics Analysis')

# Get RFM columns
rfm_cols = [col for col in df.columns if col.startswith('rfm')]

# Create two columns for controls and visualization
control_col, viz_col = st.columns([1, 3])

with control_col:
    # Metric selection dropdown
    selected_metric = st.selectbox(
        'Select RFM Metric',
        options=rfm_cols,
        key='rfm_select'
    )
    
    # Number of bins slider
    n_bins = st.slider('Number of Bins', 5, 20, 10)

with viz_col:
    # Create distribution plot
    rfm_dist = px.histogram(
        df,
        x=selected_metric,
        nbins=n_bins,
        title=f'Distribution of {selected_metric}',
        labels={selected_metric: 'Metric Value', 'count': 'Number of Customers'}
    )
    
    # Add mean line
    rfm_dist.add_vline(
        x=df[selected_metric].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean"
    )
    
    st.plotly_chart(rfm_dist, use_container_width=True)

# Add a data summary section
st.markdown("---")
st.subheader("Data Summary")
st.write(f"Total number of customers: {len(df):,}")
'''

# Save the dashboard
with open('bank_dashboard.py', 'w') as f:
    f.write(streamlit_code)

print("Created dashboard with 3 interactive visualizations:")
