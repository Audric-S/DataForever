import streamlit as st
from pages.load_data import load_data_page
from pages.data_visualisation import data_visualization

def main():
    st.set_page_config(page_title="Streamlit App", page_icon="📊", layout="centered")
    
    st.sidebar.title("Navigation")
    
    nav_container = st.sidebar.container()
    nav_container.empty() 
    
    with nav_container:
        page = st.radio("Go to", ["Load Data", "Visualize datas"])
    
    if page == "Load Data":
        load_data_page()
    elif page == "Visualize datas":
        data_visualization()

if __name__ == "__main__":
    main()
