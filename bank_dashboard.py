# Add error handling and status messages to the Streamlit dashboard
streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bank Customer Analysis Dashboard", layout="wide")

# Show a spinner while loading data
with st.spinner('Loading data...'):
    try:
        df = pd.read_csv('bank_data.csv')
        st.success('Data loaded successfully!')
        st.write('Preview of data:')
        st.dataframe(df.head())
    except FileNotFoundError:
        st.error('Error: bank_data.csv not found. Please upload the data file.')
        st.stop()
    except Exception as e:
        st.error('Error loading data: ' + str(e))
        st.stop()

# Check for required columns
required_cols = ['demog_homeval', 'demog_inc', 'demog_age']
rfm_cols = [col for col in df.columns if col.startswith('rfm')]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error('Missing required columns: ' + ', '.join(missing_cols))
    st.stop()
if not rfm_cols:
    st.error('No RFM columns found in the data.')
    st.stop()

# Convert columns to numeric
for col in required_cols + rfm_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where all numeric columns are NaN
if df[required_cols + rfm_cols].isnull().all(axis=1).any():
    df = df.dropna(subset=required_cols + rfm_cols, how='all')

st.title('Bank Customer Analysis Dashboard')
st.markdown('Interactive analysis of customer demographics and RFM metrics')
st.markdown('---')

col1, col2 = st.columns(2)

# Visualization 1: Age Distribution by RFM Segments
with col1:
    st.subheader('1. Customer Age Analysis')
    age_bins = [0, 30, 40, 50, 60, 100]
    age_labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df['age_group'] = pd.cut(df['demog_age'], bins=age_bins, labels=age_labels)
    selected_rfm = st.selectbox(
        'Select RFM Metric',
        options=rfm_cols,
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
    selected_metrics = st.multiselect(
        'Select RFM Metrics to Compare',
        options=rfm_cols,
        default=rfm_cols[:3]
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
    valid_data = df[df[segment_by].notna()]
    segment_edges = np.percentile(valid_data[segment_by],
                                np.linspace(0, 100, n_segments + 1))
    valid_data['segment'] = pd.cut(valid_data[segment_by],
                                  bins=segment_edges,
                                  labels=[f'Segment {i+1}' for i in range(n_segments)])
    fig4 = px.box(valid_data,
                  x='segment',
                  y=[col for col in rfm_cols if col in selected_metrics],
                  title=f'RFM Metrics Distribution by {segment_by.replace("demog_", "")} Segments')
    st.plotly_chart(fig4, use_container_width=True)

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

with open('bank_dashboard.py', 'w') as f:
    f.write(streamlit_code)

print('Dashboard code updated with error handling and status messages.')
