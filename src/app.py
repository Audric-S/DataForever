import streamlit as st
from front.pages.load_data import load_data_page
from front.pages.data_visualisation import data_visualization
from front.pages.data_cleaning import data_cleaning

def main():
    st.set_page_config(page_title="Data Forever", page_icon="📊", layout="centered")
    
    st.sidebar.title("Navigation")
    
    nav_container = st.sidebar.container()
    nav_container.empty() 
    
    with nav_container:
        page = st.radio("Go to", ["Load Datas", "Clean Datas", "Visualize Datas"])
    
    if page == "Load Datas":
        load_data_page()
    elif page == "Clean Datas":
        data_cleaning()
    elif page == "Visualize Datas":
        data_visualization()

if __name__ == "__main__":
    main()
