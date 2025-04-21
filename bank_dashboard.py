# .py file
streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Bank Customer Analysis Dashboard", layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('bank_data.csv')
    # Convert columns to numeric
    df['demog_homeval'] = pd.to_numeric(df['demog_homeval'], errors='coerce')
    df['demog_inc'] = pd.to_numeric(df['demog_inc'], errors='coerce')
    df['demog_age'] = pd.to_numeric(df['demog_age'], errors='coerce')
    return df

df = load_data()

# Title
st.title('Bank Customer Analysis Dashboard')
st.markdown('Interactive analysis of customer demographics and RFM metrics')
st.markdown('---')

# Create two columns for better layout
col1, col2 = st.columns(2)

# Visualization 1: RFM Analysis by Age Group with dropdown
with col1:
    st.subheader('1. RFM Metrics by Age Group')
    rfm_metric = st.selectbox(
        'Select RFM Metric',
        options=[col for col in df.columns if col.startswith('rfm')],
        key='rfm_select'
    )
    
    # Create age groups
    df['age_group'] = pd.cut(df['demog_age'], 
                            bins=[0, 30, 45, 60, 100],
                            labels=['<30', '30-45', '45-60', '60+'])
    
    fig1 = px.box(df, x='age_group', y=rfm_metric,
                  title=f'{rfm_metric} Distribution by Age Group',
                  color='age_group')
    st.plotly_chart(fig1, use_container_width=True)

# Visualization 2: Income vs Home Value Scatter with RFM color intensity
with col2:
    st.subheader('2. Income vs Home Value Analysis')
    rfm_color = st.selectbox(
        'Color by RFM Metric',
        options=[col for col in df.columns if col.startswith('rfm')],
        key='scatter_color'
    )
    
    fig2 = px.scatter(df, 
                      x='demog_inc',
                      y='demog_homeval',
                      color=rfm_color,
                      title='Income vs Home Value (Colored by RFM)',
                      labels={'demog_inc': 'Income',
                             'demog_homeval': 'Home Value'})
    st.plotly_chart(fig2, use_container_width=True)

# Visualization 3: RFM Correlation Heatmap with metric selector
st.subheader('3. RFM Correlation Analysis')
col3, col4 = st.columns([1, 3])

with col3:
    corr_vars = st.multiselect(
        'Select RFM Metrics to Compare',
        options=[col for col in df.columns if col.startswith('rfm')],
        default=[col for col in df.columns if col.startswith('rfm')][:5]
    )

with col4:
    if corr_vars:
        corr_matrix = df[corr_vars].corr()
        fig3 = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r')
        fig3.update_layout(title='RFM Metrics Correlation Heatmap')
        st.plotly_chart(fig3, use_container_width=True)

# Visualization 4: Customer Segmentation by Income with adjustable thresholds
st.subheader('4. Customer Income Segmentation')
col5, col6 = st.columns([1, 3])

with col5:
    income_range = st.slider(
        'Select Income Range (in thousands)',
        float(df['demog_inc'].min()),
        float(df['demog_inc'].max()),
        (float(df['demog_inc'].min()), float(df['demog_inc'].max()))
    )
    
    n_bins = st.slider('Number of Income Segments', 3, 10, 5)

with col6:
    filtered_df = df[
        (df['demog_inc'] >= income_range[0]) & 
        (df['demog_inc'] <= income_range[1])
    ]
    
    fig4 = px.histogram(filtered_df,
                        x='demog_inc',
                        nbins=n_bins,
                        title='Customer Income Distribution',
                        labels={'demog_inc': 'Income'})
    
    fig4.add_vline(x=filtered_df['demog_inc'].median(),
                   line_dash="dash",
                   annotation_text="Median Income")
    
    st.plotly_chart(fig4, use_container_width=True)

# Add insights section
st.markdown('---')
st.subheader('Key Insights')
st.markdown("""
- Use the dropdown menus and sliders above to explore different aspects of the data
- The RFM metrics show varying patterns across age groups
- Income and home value relationships reveal customer segments
- Correlation analysis helps identify related RFM behaviors
- Income distribution helps in customer segmentation strategies
""")
'''

# Save the dashboard code
with open('bank_dashboard.py', 'w') as f:
    f.write(streamlit_code)

print("Enhanced dashboard code has been saved to bank_dashboard.py")
