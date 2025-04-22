# streamlit
fixed_code = """
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Basic page config
st.set_page_config(page_title="Bank Analysis", layout="wide")

# Load data
try:
    df = pd.read_csv('bank_data_knn_imputed.csv')
    # Convert columns to numeric
    df['demog_homeval'] = pd.to_numeric(df['demog_homeval'], errors='coerce')
    df['demog_inc'] = pd.to_numeric(df['demog_inc'], errors='coerce')
    df['demog_age'] = pd.to_numeric(df['demog_age'], errors='coerce')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title
st.title('Bank Customer Analysis')

# Create two columns for better layout
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
st.subheader('RFM Metrics Comparison')  
  
# Get RFM columns (excluding rfm1)  
rfm_cols = [col for col in df.columns if col.startswith('rfm') and col[3:].isdigit() and 2 <= int(col[3:]) <= 12]  
  
# Create two columns for selecting metrics  
col1, col2 = st.columns(2)  
  
with col1:  
    rfm_x = st.selectbox(  
        'Select First RFM Metric (X-axis)',  
        options=rfm_cols,  
        key='rfm_x'  
    )  
  
with col2:  
    # Remove the selected X metric from Y options  
    rfm_y_options = [col for col in rfm_cols if col != rfm_x]  
    rfm_y = st.selectbox(  
        'Select Second RFM Metric (Y-axis)',  
        options=rfm_y_options,  
        key='rfm_y'  
    )  
  
# Calculate correlation coefficient  
correlation = df[rfm_x].corr(df[rfm_y])  
  
# Create scatter plot  
scatter_plot = px.scatter(  
    df,  
    x=rfm_x,  
    y=rfm_y,  
    title=f'Correlation between {rfm_x} and {rfm_y}',  
    labels={  
        rfm_x: f'{rfm_x} Values',  
        rfm_y: f'{rfm_y} Values'  
    },  
    trendline="ols"  # Add trendline directly in the main plot  
)  
  
# Add correlation coefficient annotation  
scatter_plot.add_annotation(  
    text=f'Correlation: {correlation:.2f}',  
    xref='paper',  
    yref='paper',  
    x=0.02,  
    y=0.98,  
    showarrow=False,  
    bgcolor='white',  
    bordercolor='black',  
    borderwidth=1  
)  
  
# Display the plot  
st.plotly_chart(scatter_plot, use_container_width=True)  

# Visualization 4: Target Analysis
st.subheader('Target Analysis')

# Create two columns for the target analysis
target_col1, target_col2 = st.columns(2)

with target_col1:
    # Target distribution by home ownership
    target_ho = px.bar(
        df.groupby('demog_ho')['int_tgt'].mean().reset_index(),
        x='demog_ho',
        y='int_tgt',
        title='Target Rate by Home Ownership',
        labels={
            'demog_ho': 'Home Ownership',
            'int_tgt': 'Target Rate'
        }
    )
    st.plotly_chart(target_ho, use_container_width=True)

with target_col2:
    # Target distribution by age groups
    df['age_group'] = pd.cut(df['demog_age'], 
                            bins=[0, 30, 40, 50, 60, 70, 100],
                            labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+'])
    
    target_age = px.bar(
        df.groupby('age_group')['int_tgt'].mean().reset_index(),
        x='age_group',
        y='int_tgt',
        title='Target Rate by Age Group',
        labels={
            'age_group': 'Age Group',
            'int_tgt': 'Target Rate'
        }
    )
    st.plotly_chart(target_age, use_container_width=True)

# Add a data summary section
st.markdown("---")
st.subheader("Data Summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("Total Customers", f"{len(df):,}")
with summary_col2:
    st.metric("Average Age", f"{df['demog_age'].mean():.1f}")
"""

st.header('test')

# Save the fixed dashboard
with open('bank_dashboard.py', 'w') as f:
    f.write(fixed_code)

print("Created fixed dashboard with 4 interactive visualizations and summary metrics. Key fixes include:")

