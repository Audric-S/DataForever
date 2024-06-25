import streamlit as st
from pages.load_data import load_data_page
from pages.data_visualisation import data_visualization
from pages.data_cleaning import data_cleaning
from pages.clustering_prediction import main_prediction_clustering
from streamlit_option_menu import option_menu

def main():
    st.set_page_config(page_title="Data Forever", page_icon="📊", layout="centered")
    
    with st.sidebar:
        page = option_menu("Menu", ["Load Datas", "Clean Datas", "Visualize Datas", "Prediction"], 
            icons=['cloud-upload', 'gear', ''], default_index=0)
    
    
    if page == "Load Datas":
        load_data_page()
    elif page == "Clean Datas":
        data_cleaning()
    elif page == "Visualize Datas":
        data_visualization()
    elif page == "Prediction":
        main_prediction_clustering()

if __name__ == "__main__":
    main()
