# .py file for github
streamlit_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px

# Load the imputed data
df = pd.read_csv('bank_data_knn_imputed.csv')

# Rename for convenience
bankdataimputed = df

st.set_page_config(page_title="Bank Dashboard", layout="wide")
st.title('Bank Data Dashboard (Imputed Ages)')

# --- 1. Scatterplot: Age vs. Selectable Numeric Column ---
st.header('1. Scatterplot: Age vs. Numeric Column')
numeric_cols = bankdataimputed.select_dtypes(include='number').columns.tolist()
# Remove age from y options
y_options = [col for col in numeric_cols if col not in ['demog_age', 'demog_age_imputed']]
y_axis = st.selectbox('Select Y-axis (numeric column):', y_options, key='scatter_y')

fig1 = px.scatter(bankdataimputed, x='demog_age_imputed', y=y_axis,
                  title='Scatterplot: Age vs. ' + y_axis,
                  labels={'demog_age_imputed': 'Imputed Age', y_axis: y_axis})
st.plotly_chart(fig1, use_container_width=True)

# --- 2. Categorical Comparison ---
st.header('2. Compare Categorical Columns')
categorical_cols = bankdataimputed.select_dtypes(include='object').columns.tolist()
if len(categorical_cols) < 2:
    st.warning('Not enough categorical columns for comparison.')
else:
    cat1 = st.selectbox('Select first categorical column:', categorical_cols, key='cat1')
    cat2 = st.selectbox('Select second categorical column:', [c for c in categorical_cols if c != cat1], key='cat2')
    # Countplot (bar chart) of their cross-tabulation
    cross_tab = pd.crosstab(bankdataimputed[cat1], bankdataimputed[cat2])
    st.write('Cross-tabulation table:')
    st.dataframe(cross_tab)
    fig2 = px.bar(cross_tab, barmode='group', title=f'{cat1} vs {cat2}')
    st.plotly_chart(fig2, use_container_width=True)

# --- 3. Table with Sliders: Compare Categorical and Numerical Columns ---
st.header('3. Table: Filter and Compare')
# Pick a categorical and a numeric column
cat_col = st.selectbox('Select a categorical column for filtering:', categorical_cols, key='table_cat')
num_col = st.selectbox('Select a numeric column for slider:', y_options, key='table_num')

# Get min/max for slider
min_val = float(bankdataimputed[num_col].min())
max_val = float(bankdataimputed[num_col].max())
val_range = st.slider('Select value range for ' + num_col, min_value=min_val, max_value=max_val, value=(min_val, max_val))

# Filtered table
filtered = bankdataimputed[(bankdataimputed[num_col] >= val_range[0]) & (bankdataimputed[num_col] <= val_range[1])]

# Show groupby summary
summary = filtered.groupby(cat_col)[num_col].agg(['count', 'mean', 'min', 'max']).reset_index()
st.write('Summary Table:')
st.dataframe(summary)
'''

with open('bank_dashboard.py', 'w') as f:
    f.write(streamlit_code)

print('Created new bank_dashboard.py with interactive visualizations and table.')
