import streamlit as st
from pages.load_data import load_data_page
from pages.analyze_data_page import analyze_data_page 

def main():
    st.set_page_config(page_title="Streamlit App", page_icon="📊", layout="centered")
    
    st.sidebar.title("Navigation")
    
    nav_container = st.sidebar.container()
    nav_container.empty() 
    
    with nav_container:
        page = st.radio("Go to", ["Load Data", "Analyze Data"])
    
    if page == "Load Data":
        load_data_page()
    elif page == "Analyze Data":
        analyze_data_page()

if __name__ == "__main__":
    main()
