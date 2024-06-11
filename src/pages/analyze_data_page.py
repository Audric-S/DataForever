import streamlit as st

def analyze_data_page():
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.title("Data Analysis")
        
        st.subheader("Descriptive Statistics")
        st.write(data.describe())
    else:
        st.write("No data loaded yet. Please upload a CSV file first.")