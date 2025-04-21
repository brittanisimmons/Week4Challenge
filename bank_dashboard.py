# .py file for github
streamlit_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Bank Dashboard", layout="wide")
st.title('Bank Data Dashboard (Imputed Ages)')

# Add error handling for data loading
try:
    # Check if file exists
    if not os.path.exists('bank_data_knn_imputed.csv'):
        st.error('Error: bank_data_knn_imputed.csv not found. Please ensure the file is in the same directory as the script.')
        st.stop()
    
    # Load the data with error handling
    try:
        df = pd.read_csv('bank_data_knn_imputed.csv')
        bankdataimputed = df
    except Exception as e:
        st.error(f'Error loading data: {str(e)}')
        st.stop()

    # --- 1. Scatterplot with Error Handling ---
    st.header('1. Scatterplot: Age vs. Numeric Column')
    
    # Get numeric columns
    numeric_cols = bankdataimputed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    y_options = [col for col in numeric_cols if col not in ['demog_age', 'demog_age_imputed']]
    
    if not y_options:
        st.warning('No numeric columns found for comparison.')
    else:
        y_axis = st.selectbox('Select Y-axis (numeric column):', y_options, key='scatter_y')
        
        try:
            fig1 = px.scatter(bankdataimputed, x='demog_age_imputed', y=y_axis,
                            title=f'Scatterplot: Age vs. {y_axis}',
                            labels={'demog_age_imputed': 'Imputed Age', y_axis: y_axis})
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f'Error creating scatter plot: {str(e)}')

    # --- 2. Categorical Comparison ---
    st.header('2. Compare Categorical Columns')
    
    # Get categorical columns (including object and category dtypes)
    categorical_cols = bankdataimputed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) < 2:
        st.warning('Not enough categorical columns for comparison.')
    else:
        cat1 = st.selectbox('Select first categorical column:', categorical_cols, key='cat1')
        cat2 = st.selectbox('Select second categorical column:', 
                           [c for c in categorical_cols if c != cat1], key='cat2')
        
        try:
            # Create cross-tabulation
            cross_tab = pd.crosstab(bankdataimputed[cat1], bankdataimputed[cat2])
            st.write('Cross-tabulation table:')
            st.dataframe(cross_tab)
            
            # Create bar plot
            fig2 = px.bar(cross_tab, barmode='group',
                         title=f'Comparison: {cat1} vs {cat2}')
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f'Error creating categorical comparison: {str(e)}')

    # --- 3. Interactive Table with Filters ---
    st.header('3. Table: Filter and Compare')
    
    try:
        # Select columns for filtering
        cat_col = st.selectbox('Select a categorical column for filtering:', 
                             categorical_cols, key='table_cat')
        num_col = st.selectbox('Select a numeric column for slider:', 
                             y_options, key='table_num')

        # Create slider for numeric filtering
        min_val = float(bankdataimputed[num_col].min())
        max_val = float(bankdataimputed[num_col].max())
        val_range = st.slider(f'Select range for {num_col}', 
                            min_value=min_val, max_value=max_val, 
                            value=(min_val, max_val))

        # Filter and display data
        filtered = bankdataimputed[
            (bankdataimputed[num_col] >= val_range[0]) & 
            (bankdataimputed[num_col] <= val_range[1])
        ]

        # Create summary statistics
        summary = filtered.groupby(cat_col)[num_col].agg([
            'count', 'mean', 'min', 'max'
        ]).reset_index()
        
        st.write('Summary Table:')
        st.dataframe(summary)
        
        # Add a count visualization
        fig3 = px.bar(summary, x=cat_col, y='count',
                     title=f'Count by {cat_col} (Filtered by {num_col})')
        st.plotly_chart(fig3, use_container_width=True)
        
    except Exception as e:
        st.error(f'Error in interactive table section: {str(e)}')

except Exception as e:
    st.error(f'Unexpected error: {str(e)}')
    st.write('Please check if all required files and dependencies are available.')
'''

# Save the updated dashboard code
with open('bank_dashboard.py', 'w') as f:
    f.write(streamlit_code)

print('Created updated bank_dashboard.py with error handling and improved visualizations.')
