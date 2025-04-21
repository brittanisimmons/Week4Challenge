
streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Bank Customer Analysis Dashboard", layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('bank_data.csv')
    
    # Convert columns to numeric and handle errors
    numeric_columns = ['demog_homeval', 'demog_inc', 'demog_age']
    rfm_columns = [col for col in df.columns if col.startswith('rfm')]
    
    # Convert demographic columns
    for col in numeric_columns + rfm_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where all numeric columns are NaN
    df = df.dropna(subset=numeric_columns + rfm_columns, how='all')
    
    return df

df = load_data()

# Title
st.title('Bank Customer Analysis Dashboard')
st.markdown('Interactive analysis of customer demographics and RFM metrics')
st.markdown('---')

# Create two columns for better layout
col1, col2 = st.columns(2)

# Visualization 1: Age Distribution by RFM Segments
with col1:
    st.subheader('1. Customer Age Analysis')
    
    # Create age bins for grouping
    age_bins = [0, 30, 40, 50, 60, 100]
    age_labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df['age_group'] = pd.cut(df['demog_age'], bins=age_bins, labels=age_labels)
    
    # Select RFM metric for analysis
    selected_rfm = st.selectbox(
        'Select RFM Metric',
        options=[col for col in df.columns if col.startswith('rfm')],
        key='viz1_select'
    )
    
    fig1 = px.violin(df.dropna(subset=[selected_rfm, 'age_group']),
                     x='age_group',
                     y=selected_rfm,
                     box=True,
                     title=f'Distribution of {selected_rfm} by Age Group')
    st.plotly_chart(fig1, use_container_width=True)

# Visualization 2: Income vs Home Value Analysis
with col2:
    st.subheader('2. Income vs Home Value')
    
    # Filter for income range
    income_range = st.slider(
        'Filter by Income Range',
        float(df['demog_inc'].min()),
        float(df['demog_inc'].max()),
        (float(df['demog_inc'].min()), float(df['demog_inc'].max())),
        key='viz2_slider'
    )
    
    filtered_df = df[
        (df['demog_inc'] >= income_range[0]) &
        (df['demog_inc'] <= income_range[1])
    ].dropna(subset=['demog_inc', 'demog_homeval'])
    
    fig2 = px.scatter(filtered_df,
                      x='demog_inc',
                      y='demog_homeval',
                      color='age_group',
                      title='Income vs Home Value by Age Group',
                      labels={'demog_inc': 'Income',
                             'demog_homeval': 'Home Value',
                             'age_group': 'Age Group'})
    st.plotly_chart(fig2, use_container_width=True)

# Visualization 3: RFM Metrics Analysis
st.subheader('3. RFM Metrics Comparison')
col3, col4 = st.columns([1, 3])

with col3:
    rfm_metrics = [col for col in df.columns if col.startswith('rfm')]
    selected_metrics = st.multiselect(
        'Select RFM Metrics to Compare',
        options=rfm_metrics,
        default=rfm_metrics[:3]
    )

with col4:
    if selected_metrics:
        clean_df = df[selected_metrics].dropna()
        corr_matrix = clean_df.corr()
        
        fig3 = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r',
                        title='Correlation between Selected RFM Metrics')
        st.plotly_chart(fig3, use_container_width=True)

# Visualization 4: Customer Segmentation
st.subheader('4. Customer Segmentation Analysis')
col5, col6 = st.columns([1, 3])

with col5:
    segment_by = st.selectbox(
        'Segment Customers by',
        options=['demog_inc', 'demog_homeval', 'demog_age'],
        format_func=lambda x: {'demog_inc': 'Income',
                              'demog_homeval': 'Home Value',
                              'demog_age': 'Age'}[x]
    )
    
    n_segments = st.slider('Number of Segments', 3, 10, 5)

with col6:
    # Create segments
    valid_data = df[df[segment_by].notna()]
    segment_edges = np.percentile(valid_data[segment_by],
                                np.linspace(0, 100, n_segments + 1))
    
    valid_data['segment'] = pd.cut(valid_data[segment_by],
                                  bins=segment_edges,
                                  labels=[f'Segment {i+1}' for i in range(n_segments)])
    
    fig4 = px.box(valid_data,
                  x='segment',
                  y=[col for col in rfm_metrics if col in selected_metrics],
                  title=f'RFM Metrics Distribution by {segment_by.replace("demog_", "")} Segments')
    st.plotly_chart(fig4, use_container_width=True)

# Add insights section
st.markdown('---')
st.subheader('Key Insights')
st.markdown("""
- Use the interactive controls above to explore different aspects of the data
- The violin plot shows the distribution of RFM metrics across age groups
- Income vs Home Value scatter plot reveals wealth segments by age
- RFM correlation analysis helps identify related customer behaviors
- Customer segmentation shows how RFM metrics vary across different customer groups
""")
'''

# Save the dashboard code
with open('bank_dashboard.py', 'w') as f:
    f.write(streamlit_code)

print("Enhanced dashboard code with error handling has been saved to bank_dashboard.py")
