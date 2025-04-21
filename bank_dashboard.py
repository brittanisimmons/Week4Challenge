
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load data (assume the cleaned CSV is available)
df = pd.read_csv('bank01 (1).csv')

# Clean columns for numeric analysis
df['demog_homeval'] = pd.to_numeric(df['demog_homeval'], errors='coerce')
df['demog_inc'] = pd.to_numeric(df['demog_inc'], errors='coerce')
df['demog_age'] = pd.to_numeric(df['demog_age'], errors='coerce')

st.title('Bank Data Interactive Dashboard')
st.markdown('---')

# Interactive filter: Age range slider
age_min, age_max = int(df["demog_age"].min()), int(df["demog_age"].max())
age_range = st.slider('Select Age Range', min_value=age_min, max_value=age_max, value=(age_min, age_max))
df_filtered = df[(df['demog_age'] >= age_range[0]) & (df['demog_age'] <= age_range[1])]

st.subheader('1. Correlation Heatmap (Demographics vs RFM)')
demo_vars = ['demog_age', 'demog_homeval', 'demog_inc']
rfm_vars = [col for col in df.columns if col.startswith('rfm')]
corr = df_filtered[demo_vars + rfm_vars].corr().loc[demo_vars, rfm_vars]
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
st.pyplot(fig)

st.subheader('2. Scatter Plot: Home Value vs RFM3')
fig2 = px.scatter(df_filtered, x='demog_homeval', y='rfm3', color='demog_age',
                  title='Home Value vs RFM3 (colored by Age)',
                  labels={'demog_homeval': 'Home Value', 'rfm3': 'RFM3'})
st.plotly_chart(fig2)

st.subheader('3. Distribution of Income')
fig3, ax3 = plt.subplots()
sns.histplot(df_filtered['demog_inc'].dropna(), bins=30, kde=True, ax=ax3)
ax3.set_title('Distribution of Income (Filtered by Age)')
st.pyplot(fig3)

st.markdown('---')
st.write('Use the age slider above to filter the data and see how relationships change!')


streamlit run bank_dashboard.py  
